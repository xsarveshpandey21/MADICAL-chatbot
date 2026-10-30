"""
category_good.py

Builds a reference dictionary describing what a *healthy* person looks like
for a given age group. The result is used later to categorise people in that
age group as healthy / not healthy.

Pipeline
--------
1. Input      : age range (lower-upper) and sex (1 = male, 2 = female),
                from CLI args or interactive prompt. The range is split into
                groups of AGE_STEP years (1-80 -> 1-5, 5-10, ..., 75-80) and
                steps 2-6 run separately for every group.
2. Ollama    : decides WHICH health parameters matter for that age group and
                sex, and WHAT to search the internet for.
3. Scrape     : runs the web searches (DuckDuckGo) and downloads page text.
4. Retrieval  : chunks + embeds the scraped text (nomic-embed-text) and, for
                each parameter, retrieves the most relevant chunks.
5. Ollama     : extracts the healthy range for each parameter from the
                retrieved context only.
6. Output     : one file per group and sex in Category_Dict/output/
                <lower><upper>_healty_<sex>_dictonary.json
                (e.g. 18-20 male -> 1820_healty_male_dictonary.json).
                Groups that already have a file are skipped unless
                --overwrite is given, so an interrupted run can be resumed.

Usage
-----
    python category_good.py 1-80 1             # male, groups 1-5, 5-10, ..., 75-80
    python category_good.py 1-80 2             # female
    python category_good.py 18-40 1 --step 10  # groups 18-20, 20-30, 30-40
    python category_good.py                    # prompts for age range and sex
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
SEXES = {1: "male", 2: "female"}
AGE_STEP = 5                # age range is split into groups of this size

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

MAX_PARAMETERS = 15         # upper bound on parameters Ollama may propose
RESULTS_PER_QUERY = 4       # search results fetched per query
MAX_PAGE_CHARS = 20000      # text kept per scraped page
CHUNK_SIZE = 800            # characters per chunk
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 6            # chunks retrieved per parameter
REQUEST_TIMEOUT = 15
SEARCH_DELAY = 2.0          # seconds between searches (avoids rate limiting)
SEARCH_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

client = ollama.Client(host=OLLAMA_HOST)


# ---------------------------------------------------------------------------
# Ollama block
# ---------------------------------------------------------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "unit": {"type": "string"},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "category", "unit", "search_queries"],
            },
        }
    },
    "required": ["parameters"],
}

RANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "unit": {"type": "string"},
        "min": {"type": ["number", "null"]},
        "max": {"type": ["number", "null"]},
        "healthy_value": {"type": "string"},
        "notes": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["unit", "min", "max", "healthy_value", "notes", "confidence"],
}


def ask_ollama(system: str, prompt: str, schema: dict) -> dict:
    """Call the local LLM and force a JSON response matching `schema`."""
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        format=schema,
        options={"temperature": 0.1},
    )
    return json.loads(response["message"]["content"])


def plan_parameters(age_low: int, age_high: int, sex: str) -> list[dict]:
    """Let Ollama decide which health parameters to scrape for this age group and sex."""
    system = (
        "You are a medical data analyst. You decide which measurable health "
        "parameters define a healthy person and how to search the web for "
        "their reference ranges. Respond only with JSON."
    )
    prompt = (
        f"Age group: {age_low} to {age_high} years. Sex: {sex}.\n"
        f"List up to {MAX_PARAMETERS} measurable parameters that define a healthy "
        f"{sex} in this age group. Cover vitals (heart rate, blood pressure, "
        "respiratory rate, temperature, SpO2), body composition (BMI, waist "
        "circumference), blood work (glucose, HbA1c, cholesterol, hemoglobin) "
        "and lifestyle (sleep hours, physical activity minutes per week), "
        "choosing what is relevant for this age and sex.\n"
        "For each parameter give: name, category (vitals / body / blood / "
        "lifestyle), the usual unit, and 2 short web search queries that "
        f"would find the normal range for {sex}s aged {age_low}-{age_high}."
    )
    plan = ask_ollama(system, prompt, PLAN_SCHEMA)

    # De-duplicate and clean up what the small model returns.
    seen, params = set(), []
    for p in plan.get("parameters", []):
        name = p.get("name", "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        queries = [q.strip() for q in p.get("search_queries", []) if q.strip()]
        if not queries:
            queries = [f"normal {name} range {sex} age {age_low} to {age_high}"]
        params.append({**p, "name": name, "search_queries": queries[:2]})
    return params[:MAX_PARAMETERS]


def extract_range(param: dict, context: list[dict], age_low: int, age_high: int, sex: str) -> dict:
    """Ask Ollama for the healthy range of one parameter using retrieved context."""
    context_text = "\n\n".join(
        f"[Source {i + 1}: {c['url']}]\n{c['text']}" for i, c in enumerate(context)
    )
    system = (
        "You extract healthy reference ranges from provided web text. Use the "
        "context as your primary source. If the context is insufficient, use "
        "well-established medical guidelines and set confidence to 'low'. "
        "The context may mention other measurements; only use numbers that "
        "explicitly refer to the requested parameter. Respond only with JSON."
    )
    prompt = (
        f"Parameter: {param['name']} (expected unit: {param.get('unit', '')})\n"
        f"Age group: {age_low}-{age_high} years\n"
        f"Sex: {sex}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Return the healthy range for this parameter for a {sex} in this age "
        "group: unit, numeric min and max (null if open-ended or not numeric), "
        "healthy_value as a short human readable range, notes (caveats), and "
        "confidence. If the source gives separate male and female ranges, use "
        f"the {sex} one."
    )
    return ask_ollama(system, prompt, RANGE_SCHEMA)


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = RESULTS_PER_QUERY) -> list[dict]:
    """Return [{"href", "body"}] search results, retrying on rate limits."""
    for attempt in range(1, SEARCH_RETRIES + 1):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
            return [r for r in results if r.get("href")]
        except Exception as exc:
            if attempt == SEARCH_RETRIES:
                print(f"    ! search failed for '{query}': {exc}")
                return []
            time.sleep(SEARCH_DELAY * attempt * 2)
    return []


def fetch_page_text(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return ""
    except requests.RequestException as exc:
        print(f"    ! fetch failed {url}: {exc.__class__.__name__}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_PAGE_CHARS]


def scrape(params: list[dict]) -> dict[str, str]:
    """Run every planned query and return {url: page_text}.

    Also records on each parameter (p["urls"]) which pages its own queries
    found, so retrieval can stay focused on that parameter. If a page blocks
    scraping, the search snippet is kept instead.
    """
    pages: dict[str, str] = {}
    for p in params:
        p["urls"] = set()
        for query in p["search_queries"]:
            print(f"  search: {query}")
            for result in web_search(query):
                url = result["href"]
                p["urls"].add(url)
                if url in pages:
                    continue
                text = fetch_page_text(url)
                if len(text) <= 200:
                    text = result.get("body", "")
                if text:
                    pages[url] = text
            time.sleep(SEARCH_DELAY)
    return pages


# ---------------------------------------------------------------------------
# Input based retrieval (embeddings)
# ---------------------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, max(len(text) - CHUNK_OVERLAP, 1), step)]


def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), 32):
        resp = client.embed(model=EMBED_MODEL, input=texts[i:i + 32])
        vectors.extend(resp["embeddings"])
    arr = np.array(vectors, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10)


class Retriever:
    def __init__(self, pages: dict[str, str]):
        self.chunks = [
            {"url": url, "text": chunk}
            for url, text in pages.items()
            for chunk in chunk_text(text)
        ]
        print(f"  embedding {len(self.chunks)} chunks ...")
        self.vectors = embed([c["text"] for c in self.chunks]) if self.chunks else np.empty((0, 0))

    def query(self, question: str, urls: set[str] | None = None, k: int = TOP_K_CHUNKS) -> list[dict]:
        """Top-k chunks for `question`, preferring chunks from `urls`."""
        if not self.chunks:
            return []
        q = embed([question])[0]
        scores = self.vectors @ q
        if urls:
            # Boost pages found by this parameter's own searches.
            boost = np.array([0.15 if c["url"] in urls else 0.0 for c in self.chunks])
            scores = scores + boost
        top = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_dictionary(age_low: int, age_high: int, sex: str) -> dict:
    print(f"[1/4] Asking Ollama which parameters define health for {sex}s aged {age_low}-{age_high} ...")
    params = plan_parameters(age_low, age_high, sex)
    print(f"  -> {len(params)} parameters: {', '.join(p['name'] for p in params)}")

    print("[2/4] Scraping the web ...")
    pages = scrape(params)
    print(f"  -> {len(pages)} pages scraped")

    print("[3/4] Building retrieval index ...")
    retriever = Retriever(pages)

    print("[4/4] Extracting healthy ranges ...")
    parameters = {}
    for p in params:
        question = f"normal healthy {p['name']} range for {sex}s aged {age_low} to {age_high}"
        context = retriever.query(question, p.get("urls"))
        try:
            result = extract_range(p, context, age_low, age_high, sex)
        except Exception as exc:
            print(f"  ! {p['name']}: extraction failed ({exc})")
            continue
        parameters[p["name"]] = {
            "category": p.get("category", ""),
            **result,
            "sources": sorted({c["url"] for c in context}),
        }
        print(f"  {p['name']}: {result.get('healthy_value')} [{result.get('confidence')}]")

    return {
        "age_lower": age_low,
        "age_upper": age_high,
        "sex": sex,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL,
        "parameters": parameters,
    }


def output_path(age_low: int, age_high: int, sex: str) -> Path:
    """e.g. (18, 20, "male") -> output/1820_healty_male_dictonary.json"""
    return OUTPUT_DIR / f"{age_low}{age_high}_healty_{sex}_dictonary.json"


def save(entry: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = output_path(entry["age_lower"], entry["age_upper"], entry["sex"])
    output_file.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_file


def split_age_range(lower: int, upper: int, step: int = AGE_STEP) -> list[tuple[int, int]]:
    """Split into groups on multiples of `step`: 1-80 -> 1-5, 5-10, ..., 75-80."""
    groups, start = [], lower
    while start < upper:
        end = min((start // step + 1) * step, upper)
        groups.append((start, end))
        start = end
    return groups or [(lower, upper)]


def parse_age_group(text: str) -> tuple[int, int]:
    """Parse '18-20' (or '18 20') into (18, 20)."""
    match = re.fullmatch(r"\s*(\d+)\s*[-\s]\s*(\d+)\s*", text)
    if not match:
        sys.exit(f"Invalid age group '{text}'. Use the form lower-upper, e.g. 18-20.")
    lower, upper = int(match.group(1)), int(match.group(2))
    if upper < lower:
        sys.exit("Invalid age group: lower must be <= upper.")
    return lower, upper


def read_inputs() -> tuple[int, int, str, int, bool]:
    parser = argparse.ArgumentParser(description="Build healthy-person dictionaries for an age range.")
    parser.add_argument("age_group", nargs="?", help="age range as lower-upper, e.g. 1-80")
    parser.add_argument("sex", type=int, nargs="?", choices=SEXES, help="1 = male, 2 = female")
    parser.add_argument("--step", type=int, default=AGE_STEP,
                        help=f"size of each age group (default {AGE_STEP})")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild age groups that already have an output file")
    args = parser.parse_args()
    if args.step < 1:
        sys.exit("--step must be at least 1.")

    age_group = args.age_group or input("Age group (lower-upper, e.g. 18-20): ")
    lower, upper = parse_age_group(age_group)

    sex_code = args.sex
    while sex_code not in SEXES:
        choice = input("Sex (1 = male, 2 = female): ").strip()
        sex_code = int(choice) if choice.isdigit() else None
    return lower, upper, SEXES[sex_code], args.step, args.overwrite


def main():
    age_low, age_high, sex, step, overwrite = read_inputs()
    try:
        client.list()
    except Exception:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_HOST}. Is it running?")

    # Each age group has its own health standards, so every group gets its
    # own planning, scraping, retrieval and output file.
    groups = split_age_range(age_low, age_high, step)
    print(f"Age groups ({sex}): {', '.join(f'{lo}-{hi}' for lo, hi in groups)}")

    saved, skipped, failed = [], [], []
    for i, (lo, hi) in enumerate(groups, 1):
        print(f"\n=== Age group {lo}-{hi} ({i}/{len(groups)}) ===")
        if not overwrite and output_path(lo, hi, sex).exists():
            print(f"  already saved: {output_path(lo, hi, sex).name} (use --overwrite to rebuild)")
            skipped.append(f"{lo}-{hi}")
            continue
        try:
            entry = build_dictionary(lo, hi, sex)
            path = save(entry)
        except Exception as exc:
            print(f"  ! age group {lo}-{hi} failed: {exc}")
            failed.append(f"{lo}-{hi}")
            continue
        print(f"  Saved {len(entry['parameters'])} parameters to {path.name}")
        saved.append(f"{lo}-{hi}")

    print(f"\nDone. Saved: {len(saved)}, skipped: {len(skipped)}, failed: {len(failed)}")
    if failed:
        print(f"Failed groups (re-run to retry): {', '.join(failed)}")


if __name__ == "__main__":
    main()
