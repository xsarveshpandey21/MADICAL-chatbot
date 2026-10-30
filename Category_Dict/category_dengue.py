"""
category_dengue.py

Builds a reference dictionary describing what a person WITH DENGUE looks like
for a given age group and sex - the dengue counterpart of category_good.py.
The result is used later to generate / categorise dengue patients.

The parameters are based on the Dengue screening app (BulkData/app.py):
the ML features it uses (temperature, symptoms, platelet count, WBC count,
hematocrit) and the WHO warning signs it lists. Ollama may add a few more
dengue-relevant parameters (e.g. heart rate, blood pressure).

Two kinds of parameter
----------------------
numeric : a measured value, with the typical range seen in dengue patients
          (min / max), e.g. Platelet Count 20000-100000 cells/uL
symptom : yes/no finding, with the share of dengue patients who have it
          (prevalence_percent), e.g. Headache 75 %

Pipeline (same as category_good.py)
-----------------------------------
1. Input      : age range (lower-upper) and sex (1 = male, 2 = female). The
                range is split into groups of AGE_STEP years and steps 2-6 run
                separately for every group.
2. Ollama     : writes the web search queries for every parameter and may
                propose extra parameters.
3. Scrape     : DuckDuckGo search + page download.
4. Retrieval  : embeddings (nomic-embed-text), top chunks per parameter.
5. Ollama     : extracts the dengue values for each parameter.
6. Output     : Category_Dict/output_dengue/<lower><upper>_dengue_<sex>_dictonary.json
                Groups that already have a file are skipped unless --overwrite.

Usage
-----
    python category_dengue.py 1-80 1             # male, groups 1-5, 5-10, ..., 75-80
    python category_dengue.py 1-80 2             # female
    python category_dengue.py 18-40 1 --step 10
    python category_dengue.py                    # prompts for age range and sex
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from category_good import (
    AGE_STEP,
    LLM_MODEL,
    OLLAMA_HOST,
    SEXES,
    Retriever,
    ask_ollama,
    client,
    parse_age_group,
    scrape,
    split_age_range,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_dengue"

MAX_EXTRA_PARAMETERS = 4    # extra parameters Ollama may add to the core list

# Parameters from the Dengue app: its ML features + the WHO warning signs.
CORE_PARAMETERS = [
    {"name": "Body Temperature", "kind": "numeric", "category": "vitals", "unit": "°C"},
    {"name": "Platelet Count", "kind": "numeric", "category": "blood", "unit": "cells/µL"},
    {"name": "WBC Count", "kind": "numeric", "category": "blood", "unit": "cells/µL"},
    {"name": "Hematocrit", "kind": "numeric", "category": "blood", "unit": "%"},
    {"name": "Headache", "kind": "symptom", "category": "symptom", "unit": ""},
    {"name": "Retro-orbital Pain", "kind": "symptom", "category": "symptom", "unit": ""},
    {"name": "Muscle and Joint Pain", "kind": "symptom", "category": "symptom", "unit": ""},
    {"name": "Nausea or Vomiting", "kind": "symptom", "category": "symptom", "unit": ""},
    {"name": "Skin Rash", "kind": "symptom", "category": "symptom", "unit": ""},
    {"name": "Severe Abdominal Pain", "kind": "symptom", "category": "warning sign", "unit": ""},
    {"name": "Persistent Vomiting", "kind": "symptom", "category": "warning sign", "unit": ""},
    {"name": "Mucosal Bleeding", "kind": "symptom", "category": "warning sign", "unit": ""},
    {"name": "Lethargy or Restlessness", "kind": "symptom", "category": "warning sign", "unit": ""},
]


# ---------------------------------------------------------------------------
# Ollama block
# ---------------------------------------------------------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "search_queries"],
            },
        },
        "extra_parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["numeric", "symptom"]},
                    "category": {"type": "string"},
                    "unit": {"type": "string"},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "kind", "category", "unit", "search_queries"],
            },
        },
    },
    "required": ["queries", "extra_parameters"],
}

NUMERIC_SCHEMA = {
    "type": "object",
    "properties": {
        "unit": {"type": "string"},
        "min": {"type": ["number", "null"]},
        "max": {"type": ["number", "null"]},
        "dengue_value": {"type": "string"},
        "severe_value": {"type": "string"},
        "notes": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["unit", "min", "max", "dengue_value", "severe_value", "notes", "confidence"],
}

SYMPTOM_SCHEMA = {
    "type": "object",
    "properties": {
        "prevalence_percent": {"type": ["number", "null"]},
        "dengue_value": {"type": "string"},
        "notes": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["prevalence_percent", "dengue_value", "notes", "confidence"],
}


def plan_parameters(age_low: int, age_high: int, sex: str) -> list[dict]:
    """Ollama writes search queries for the core parameters and may add extras."""
    core_names = ", ".join(p["name"] for p in CORE_PARAMETERS)
    system = (
        "You are a clinical researcher specialising in dengue fever. You plan web "
        "searches to find how dengue presents in patients. Respond only with JSON."
    )
    prompt = (
        f"Patients: {sex}s aged {age_low} to {age_high} years with dengue fever.\n"
        f"Core parameters: {core_names}.\n"
        "1. In 'queries', give every core parameter 2 short web search queries that "
        f"would find its typical value or frequency in dengue patients aged {age_low}-{age_high}.\n"
        f"2. In 'extra_parameters', add up to {MAX_EXTRA_PARAMETERS} other parameters that "
        "change in dengue (for example heart rate, blood pressure, liver enzymes), each "
        "with kind (numeric or symptom), category, unit and 2 search queries."
    )
    plan = ask_ollama(system, prompt, PLAN_SCHEMA)

    planned = {q.get("name", "").strip().lower(): q for q in plan.get("queries", [])}
    params = []
    for core in CORE_PARAMETERS:
        q = planned.get(core["name"].lower(), {})
        queries = [s.strip() for s in q.get("search_queries", []) if s.strip()]
        if not queries:
            queries = [f"{core['name']} in dengue fever patients age {age_low} to {age_high}"]
        params.append({**core, "search_queries": queries[:2]})

    seen = {p["name"].lower() for p in params}
    for extra in plan.get("extra_parameters", [])[:MAX_EXTRA_PARAMETERS]:
        name = extra.get("name", "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        queries = [s.strip() for s in extra.get("search_queries", []) if s.strip()]
        if not queries:
            queries = [f"{name} in dengue fever patients"]
        params.append({**extra, "name": name, "search_queries": queries[:2]})
    return params


def extract_dengue_values(param: dict, context: list[dict], age_low: int, age_high: int, sex: str) -> dict:
    """Ask Ollama for the dengue value of one parameter using retrieved context."""
    context_text = "\n\n".join(
        f"[Source {i + 1}: {c['url']}]\n{c['text']}" for i, c in enumerate(context)
    )
    system = (
        "You extract how dengue fever presents in patients from provided web text. "
        "Use the context as your primary source. If it is insufficient, use WHO dengue "
        "guidelines and set confidence to 'low'. Give values seen IN DENGUE PATIENTS, "
        "not normal healthy values. The context may mention other measurements; only use "
        "numbers that refer to the requested parameter. Respond only with JSON."
    )
    header = (
        f"Parameter: {param['name']}\n"
        f"Patients: {sex}s aged {age_low}-{age_high} years with dengue fever\n\n"
        f"Context:\n{context_text}\n\n"
    )
    if param["kind"] == "symptom":
        prompt = header + (
            "Return prevalence_percent (0-100, the share of dengue patients who have this "
            "symptom), dengue_value (short description of how it presents), notes, and confidence."
        )
        return ask_ollama(system, prompt, SYMPTOM_SCHEMA)

    prompt = header + (
        f"Expected unit: {param.get('unit', '')}\n"
        "Return the typical range in dengue patients: unit, numeric min and max, "
        "dengue_value as a short readable range, severe_value (the level that indicates "
        "severe dengue or a warning sign), notes, and confidence."
    )
    return ask_ollama(system, prompt, NUMERIC_SCHEMA)


def normalise(param: dict, result: dict) -> dict:
    """Fix common unit slips from the small model."""
    if param["kind"] == "symptom":
        p = result.get("prevalence_percent")
        if isinstance(p, (int, float)) and 0 < p <= 1:
            result["prevalence_percent"] = round(p * 100, 1)
        return result

    # Counts reported as x10^3/uL or x10^9/L (e.g. "100") -> cells/uL (100000).
    if param["name"] in ("Platelet Count", "WBC Count"):
        for side in ("min", "max"):
            v = result.get(side)
            if isinstance(v, (int, float)) and 0 < v < 1000:
                result[side] = int(v * 1000)
        result["unit"] = "cells/µL"
    lo, hi = result.get("min"), result.get("max")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
        result["min"], result["max"] = hi, lo
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_dictionary(age_low: int, age_high: int, sex: str) -> dict:
    print(f"[1/4] Asking Ollama what to search for dengue in {sex}s aged {age_low}-{age_high} ...")
    params = plan_parameters(age_low, age_high, sex)
    print(f"  -> {len(params)} parameters: {', '.join(p['name'] for p in params)}")

    print("[2/4] Scraping the web ...")
    pages = scrape(params)
    print(f"  -> {len(pages)} pages scraped")

    print("[3/4] Building retrieval index ...")
    retriever = Retriever(pages)

    print("[4/4] Extracting dengue values ...")
    parameters = {}
    for p in params:
        question = f"{p['name']} in dengue fever patients, {sex}s aged {age_low} to {age_high}"
        context = retriever.query(question, p.get("urls"))
        try:
            result = normalise(p, extract_dengue_values(p, context, age_low, age_high, sex))
        except Exception as exc:
            print(f"  ! {p['name']}: extraction failed ({exc})")
            continue
        parameters[p["name"]] = {
            "kind": p["kind"],
            "category": p.get("category", ""),
            **result,
            "sources": sorted({c["url"] for c in context}),
        }
        shown = (f"{result.get('prevalence_percent')} %" if p["kind"] == "symptom"
                 else result.get("dengue_value"))
        print(f"  {p['name']}: {shown} [{result.get('confidence')}]")

    return {
        "age_lower": age_low,
        "age_upper": age_high,
        "sex": sex,
        "condition": "dengue",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL,
        "parameters": parameters,
    }


def output_path(age_low: int, age_high: int, sex: str) -> Path:
    """e.g. (18, 20, "male") -> output_dengue/1820_dengue_male_dictonary.json"""
    return OUTPUT_DIR / f"{age_low}{age_high}_dengue_{sex}_dictonary.json"


def save(entry: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path(entry["age_lower"], entry["age_upper"], entry["sex"])
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_inputs() -> tuple[int, int, str, int, bool]:
    parser = argparse.ArgumentParser(description="Build dengue-patient dictionaries for an age range.")
    parser.add_argument("age_group", nargs="?", help="age range as lower-upper, e.g. 1-80")
    parser.add_argument("sex", type=int, nargs="?", choices=SEXES, help="1 = male, 2 = female")
    parser.add_argument("--step", type=int, default=AGE_STEP,
                        help=f"size of each age group (default {AGE_STEP})")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild age groups that already have an output file")
    args = parser.parse_args()
    if args.step < 1:
        sys.exit("--step must be at least 1.")

    lower, upper = parse_age_group(args.age_group or input("Age group (lower-upper, e.g. 18-20): "))
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

    # Dengue presents differently by age (e.g. children have more vomiting and
    # shock, adults more joint pain), so every group is researched separately.
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
