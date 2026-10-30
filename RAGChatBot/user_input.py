"""
user_input.py

Step 1 of the RAG pipeline: analyse the user's question, find the
patient(s) it is about in the generated files in BulkOutput/output, and
decide which route the question takes.

Matching (regular expressions, case-insensitive)
------------------------------------------------
1. Full name : every known full name is put in one regex; any full name
               found in the question is a match ("How is Maria Lopez?").
2. Partial   : if no full name is found, single first or last names are
               tried ("How is Lopez doing?"). They must be capitalised and
               common English words are ignored, so "white blood cells",
               "Will" or "May" do not match by accident.

Routes
------
patient   : a name was found -> insight on those patients ("compare her with
            Aidan Kim" also includes the patient in focus).
follow_up : no name, but the question refers back to earlier patients:
            "what about her platelets?" (patient in focus), "is the second
            one okay?" (last list), "what are their symptoms?" (last list).
conversation : questions about the chat itself ("what is my name?", "which
            patients have we discussed?") and small talk -> answered from the
            chat history, no search.
dataset   : no name -> Ollama searches all records in BulkOutput/output
            ("How many dengue patients are over 60?"); "how many of them are
            female?" narrows the previous search.

The session (new_session) remembers all of this until the chat is cleared.

The files are re-read automatically when a new output file is generated.

Usage
    python user_input.py "How is Maria Lopez doing?"
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RECORDS_DIR = BASE_DIR.parent / "BulkOutput" / "output"

MIN_PART_LENGTH = 3         # shortest first/last name used for partial matches
COMMON_WORDS = {
    "all", "and", "any", "are", "art", "bad", "bill", "can", "dawn", "does", "for",
    "from", "god", "grace", "has", "her", "him", "his", "hope", "how", "joy", "king",
    "long", "low", "mark", "may", "much", "new", "not", "now", "patient", "rich",
    "rose", "say", "she", "show", "summer", "the", "their", "they", "was", "what",
    "who", "will", "with", "young",
}

MAX_LIST_PATIENTS = 5      # patients taken from the last list for "they / them"

# Words that point back to the patient in focus ...
FOLLOW_UP_WORDS = re.compile(
    r"\b(he|she|him|her|his|hers|this patient|that patient|the patient|this person)\b", re.IGNORECASE)
# ... or to the patients of the last dataset search ...
PLURAL_WORDS = re.compile(r"\b(they|them|their|those|these)\b", re.IGNORECASE)
# ... unless the question is clearly about the whole dataset.
DATASET_WORDS = re.compile(
    r"\b(how many|which|who|list|all|every(one|body)?|patients|people|records|average|mean|"
    r"total|count|most|least|highest|lowest|top|dataset)\b", re.IGNORECASE)
# Narrowing the last dataset search ("how many of them are female?").
REFINE_WORDS = re.compile(
    r"\b(of|among|from|out of) (them|those|these)\b|^\s*(and|only|now|what about|how about|just)\b",
    re.IGNORECASE)
# Questions about the conversation itself ...
MEMORY_WORDS = re.compile(
    r"\b(my name|who am i|(have|did) we (discuss|talk about|cover)|discussed|so far|"
    r"you (said|told|mentioned)|i (said|told you|asked|mentioned)|remind me|recap|"
    r"summari[sz]e (our|this|the) (chat|conversation|discussion))\b", re.IGNORECASE)
# ... and small talk (only when it is not also a data question).
SMALL_TALK = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok(ay)?|good (morning|afternoon|evening))\b|"
    r"\b(my name is|call me|i am dr|i'm dr|remember that|keep (your )?answers)\b", re.IGNORECASE)
# A patient from the last list ("the second one", "#3", "number 2", "the last patient").
ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9, "tenth": 10, "10th": 10, "last": -1,
}
ORDINAL_WORDS = re.compile(
    rf"\b({'|'.join(ORDINALS)})(?:\s+(?:one|patient|person|record|name))?\b|#\s?(\d+)\b|\bnumber\s+(\d+)\b",
    re.IGNORECASE)

_cache = {"key": None, "records": [], "full": None, "parts": None}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def _normalise(name: str) -> str:
    return " ".join(name.lower().split())


def _name_pattern(names: set[str]) -> re.Pattern | None:
    """One regex for all names; longest first so 'Ann Lee' wins over 'Ann'."""
    if not names:
        return None
    alternatives = sorted(names, key=len, reverse=True)
    body = "|".join(r"\s+".join(map(re.escape, n.split())) for n in alternatives)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def load_records() -> list[dict]:
    """All records from BulkOutput/output, each tagged with its source file."""
    files = sorted(RECORDS_DIR.glob("*.json"))
    key = tuple((f.name, f.stat().st_mtime) for f in files)
    if key == _cache["key"]:
        return _cache["records"]

    records = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! Skipping {path.name}: {exc}")
            continue
        for r in data.get("records", []):
            records.append({**r, "source_file": path.name})

    full_names = {_normalise(r["name"]) for r in records if r.get("name")}
    parts = {p for n in full_names for p in n.split()
             if len(p) >= MIN_PART_LENGTH and p not in COMMON_WORDS}
    _cache.update(key=key, records=records,
                  full=_name_pattern(full_names), parts=_name_pattern(parts))
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def find_names(text: str, records: list[dict]) -> dict:
    """{"match_type" ("full" | "partial" | "none"), "names_found", "matches"}"""
    result = {"match_type": "none", "names_found": [], "matches": []}
    if not text or not records:
        return result

    if _cache["full"]:
        found = {_normalise(m.group(0)) for m in _cache["full"].finditer(text)}
        if found:
            result.update(match_type="full", names_found=sorted(found),
                          matches=[r for r in records if _normalise(r["name"]) in found])
            return result

    if _cache["parts"]:
        # Single names must be capitalised: "white blood cells" is not Mr White.
        found = {_normalise(m.group(0)) for m in _cache["parts"].finditer(text) if m.group(0)[0].isupper()}
        matches = [r for r in records if found & set(_normalise(r["name"]).split())]
        if matches:
            result.update(match_type="partial", names_found=sorted(found), matches=matches)
    return result


def new_session() -> dict:
    """
    What the chat remembers until it is cleared.

    focus      : patients the conversation is about (for he / she / her)
    dataset    : last dataset search {"plan", "shown"} (for them / the second one)
    discussed  : names of every patient discussed, in order
    searches   : every dataset search run, as text
    summary    : summary of older turns that no longer fit in the prompt
    summarized : number of chat messages already folded into the summary
    """
    return {"focus": [], "dataset": None, "discussed": [], "searches": [],
            "summary": "", "summarized": 0}


def ordinal_pick(text: str, shown: list[dict]) -> list[dict]:
    m = ORDINAL_WORDS.search(text)
    if not m or not shown:
        return []
    n = ORDINALS[m[1].lower()] if m[1] else int(m[2] or m[3])
    if n == -1:
        return [shown[-1]]
    return [shown[n - 1]] if 1 <= n <= len(shown) else []


def analyze(text: str, session: dict | None = None) -> dict:
    """
    Find the patients the question is about and pick the route.

    Returns {"query", "route" ("patient" | "follow_up" | "conversation" | "dataset"),
             "reference", "refine", "match_type", "names_found", "matches"}.
    `reference` says how follow-ups were resolved; `refine` is True when a
    dataset question narrows the previous dataset search.
    """
    session = session or new_session()
    text = text.strip()
    focus = session["focus"]
    shown = (session["dataset"] or {}).get("shown", [])
    singular = bool(FOLLOW_UP_WORDS.search(text))
    plural = bool(PLURAL_WORDS.search(text))
    about_dataset = bool(DATASET_WORDS.search(text))

    result = {"query": text, "route": "dataset", "reference": None, "refine": False,
              **find_names(text, load_records())}
    if result["matches"]:
        result["route"] = "patient"
        # Only compare with previous patient if question explicitly expresses a comparison
        is_comparison = bool(re.search(r"\b(compare|versus|vs|both|along with|with)\b", text, re.IGNORECASE))
        if singular and focus and is_comparison:          # "compare her with Aidan Kim"
            new = [r for r in result["matches"] if r not in focus]
            result.update(matches=focus + new, reference="named + previous patient")
    elif picked := ordinal_pick(text, shown):
        result.update(route="follow_up", matches=picked, reference="from the last list")
    elif singular and focus and not about_dataset:
        result.update(route="follow_up", matches=focus, reference="previous patient")
    elif MEMORY_WORDS.search(text) or (SMALL_TALK.search(text) and not about_dataset):
        result.update(route="conversation", reference="the conversation so far")
    elif shown and REFINE_WORDS.search(text):
        result.update(refine=True, reference="narrowing the last search")
    elif plural and shown and not about_dataset:
        result.update(route="follow_up", matches=shown[:MAX_LIST_PATIENTS], reference="the last list")
    return result


def patient_names() -> list[str]:
    return sorted({r["name"] for r in load_records() if r.get("name")})


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    res = analyze(question)
    print(f"route: {res['route']}  match: {res['match_type']}  names: {res['names_found']}  "
          f"reference: {res['reference']}")
    for r in res["matches"]:
        print(f"  {r['name']} ({r['sex']}, {r['age']}) - {r.get('condition', '?')} "
              f"[{r['source_file']} #{r['id']}]")
