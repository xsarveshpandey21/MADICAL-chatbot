"""
ollama_engine.py

Step 2 of the RAG pipeline: turn the records found by user_input.py into
context and ask Ollama for an answer.

Patient route (a name was found)
--------------------------------
Context per patient:
- name, sex, age, age group, condition on record, source file
- every vital with its healthy range (Data_Distl/output, the same ranges
  bulkData.py used) and a LOW / HIGH / normal flag
- symptoms present / absent and the dengue warning signs found

Dataset route (fallback when no name was found)
-----------------------------------------------
1. Plan     : Ollama turns the question into a structured query (condition,
              sex, age range, symptoms, LOW/HIGH vitals, sort, limit), then
              ground_plan() corrects it with what the question states
              explicitly (regex), so a small model cannot add or drop filters.
2. Retrieve : the query is run in Python over every record in
              BulkOutput/output, so counts and sorting are exact.
3. Answer   : Ollama answers from an overview of the whole dataset plus the
              matching records.

Conversation memory
-------------------
Every answer is sent with the earlier turns of the chat. When those get
longer than HISTORY_CHAR_BUDGET the oldest turns are folded into a running
summary (prepare_history), so nothing is lost until the chat is cleared.

Usage
    python ollama_engine.py "How is Maria Lopez doing?"
    python ollama_engine.py "Which dengue patients have the lowest platelets?"
"""

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import ollama

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "BulkOutput"))
from bulkData import (CHILD_AGE, CHILD_WBC, HEALTHY_DIR, HEALTHY_LABS,  # noqa: E402
                      baseline_dictionary, load_dictionaries)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
client = ollama.Client(host=OLLAMA_HOST)

def _detect_model():
    try:
        m = [x["model"] for x in client.list()["models"]]
        return "llama3.2:3b" if "llama3.2:3b" in m else (m[0] if m else "llama3.2:3b")
    except Exception:
        return "llama3.2:3b"

LLM_MODEL = _detect_model()
TEMPERATURE = 0.3
NUM_CTX = 8192              # Ollama context window (its default is too small for chat history)
HISTORY_CHAR_BUDGET = 12000 # earlier turns sent as-is; older ones are folded into a summary
MAX_PATIENTS = 5            # patients put in the context at most (patient route)
MAX_ROWS = 30               # matching records put in the context at most (dataset route)
NARROW_PULSE_PRESSURE = 20  # mmHg; at or below this is a dengue shock sign
LOW_PLATELETS = 100000      # cells/µL; WHO dengue warning threshold
WARNING_SIGN_SYMPTOMS = {   # symptoms with category "warning sign" in Data_Distl/output_dengue
    "Severe Abdominal Pain", "Persistent Vomiting", "Mucosal Bleeding", "Lethargy or Restlessness",
}

SYSTEM_PROMPT = (
    "You are a clinical assistant reviewing synthetic patient records. "
    "Answer health questions only from the patient data in the context; answer "
    "questions about the conversation itself (the user's name, what was said "
    "earlier) from the earlier turns and the summary. For each patient give a "
    "short health insight: the overall picture, the values flagged LOW or HIGH and "
    "what they may indicate, and sensible next steps. Quote the actual values. "
    "The dengue warning signs are exactly those in the line 'Dengue warning signs "
    "found'; do not add others. Never describe a symptom from 'Symptoms ABSENT' as "
    "present, and do not guess values that are not given. "
    "If the question asks something the data cannot answer, say so. "
    "You remember the whole conversation: use the earlier turns and the conversation "
    "summary to resolve follow-up questions and to recall what was said before, "
    "including anything the user told you about themselves or asked you to remember. "
    "Patient values must come from the patient data of the current message. "
    "End with one line noting this is not medical advice."
)

CONVERSATION_PROMPT = (
    "You are a clinical assistant chatting about synthetic patient records. The user's "
    "message is about the conversation itself or is small talk. Answer from the earlier "
    "turns, the conversation summary and the conversation log. The log is the exact list "
    "of patients discussed and searches run; never name a patient that is not in it. "
    "If the user has not told you something (for example their name), say so. "
    "Keep the reply short; you may suggest what they can ask about patients."
)

SUMMARY_PROMPT = (
    "You keep a running summary of a conversation between a user and a clinical "
    "assistant about synthetic patient records. Merge the new turns into the current "
    "summary. Start with a line 'About the user:' holding what the user said about "
    "themselves (name, role, preferences), or 'unknown'. Then keep every patient name "
    "with the key values and findings discussed, the searches asked for and their "
    "results, and anything the user asked to remember. Write short bullet points, at "
    "most 250 words. Reply with the summary only."
)

PLAN_PROMPT = (
    "You turn a question about a dataset of patient records into a search query. "
    "Fill every field. Use 'any' for condition or sex, 0 for age_min, 100 for age_max "
    "and empty lists when the question does not restrict them. "
    "'symptoms' are symptoms the patients must have. "
    "'abnormal' lists vitals that must be LOW or HIGH (use ANY when the question just "
    "says abnormal, out of range or bad). Set 'warning_signs_only' to true only when "
    "the question asks about dengue warning signs or severe cases. "
    "Set 'sort_by' when the question asks for highest, lowest, oldest, youngest or top; "
    "otherwise 'none'. 'limit' is how many patients the question asks for, 0 if not stated."
)

DATASET_PROMPT = (
    "You answer questions about a dataset of synthetic patient records. For the data, "
    "use only the dataset overview and the matching records in the context; answer "
    "questions about the conversation itself (the user's name, what was said earlier) "
    "from the earlier turns and the summary. The search was run by "
    "code, so the line 'Matching records: N' is the exact count; use it for 'how many' "
    "questions. When naming patients, use names and values from the matching records "
    "only; never invent patients or values. The matching records are already sorted "
    "and numbered; list them in exactly that order. If only part of the matches is shown, say "
    "so. If the data cannot answer the question, say so. "
    "You remember the whole conversation: use the earlier turns and the conversation "
    "summary to resolve follow-up questions and to recall what was said before, "
    "including anything the user told you about themselves or asked you to remember. "
    "Counts and values must come from the context of the current message. "
    "End with one line noting this is not medical advice."
)

# Words in a question that refer to a vital (its full name always counts too).
VITAL_KEYWORDS = {
    "Systolic Blood Pressure": ["systolic", "blood pressure", "bp"],
    "Diastolic Blood Pressure": ["diastolic"],
    "Heart Rate": ["heart rate", "pulse"],
    "Respiratory Rate": ["respiratory", "breathing"],
    "Body Temperature": ["temperature", "fever"],
    "SpO2": ["spo2", "oxygen", "saturation"],
    "Fasting Blood Glucose": ["glucose", "blood sugar"],
    "Total Cholesterol": ["cholesterol"],
    "LDL Cholesterol": ["ldl"],
    "HDL Cholesterol": ["hdl"],
    "Hemoglobin": ["haemoglobin"],
    "Waist Circumference": ["waist"],
    "Sleep Hours": ["sleep"],
    "Physical Activity Minutes per Week": ["physical activity", "exercise"],
    "Platelet Count": ["platelet", "platelets"],
    "WBC Count": ["wbc", "white blood cell", "white blood cells", "leukocytes"],
    "Hematocrit": ["haematocrit", "hct"],
}
GENERIC_SYMPTOM_WORDS = {"pain", "severe", "persistent", "or", "and"}

LOW_WORDS = r"low|lower|below|decreased|reduced|dropping|dropped"
HIGH_WORDS = r"high|higher|above|elevated|raised|increased"
ABNORMAL_WORDS = r"abnormal|out of range|bad"
SORT_ASC = r"lowest|least|minimum|smallest|youngest"
SORT_DESC = r"highest|most|maximum|largest|oldest|worst"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
AGE_NOUN = (r"(?:patients?|people|persons?|m[ae]n|wom[ae]n|males?|females?|boys|girls|"
            r"adults|children|kids|those|anyone|aged?|ages)")

DEFAULT_PLAN = {
    "condition": "any", "sex": "any", "age_min": 0, "age_max": 100, "symptoms": [],
    "abnormal": [], "warning_signs_only": False, "sort_by": "none", "order": "desc", "limit": 0,
}

client = ollama.Client(host=OLLAMA_HOST)
_healthy = None


# ---------------------------------------------------------------------------
# Reference ranges
# ---------------------------------------------------------------------------
def healthy_dictionaries() -> list[dict]:
    global _healthy
    if _healthy is None:
        try:
            _healthy = load_dictionaries(HEALTHY_DIR, "*_healty_*_dictonary.json")
        except SystemExit:
            _healthy = []
    return _healthy


def reference_ranges(record: dict) -> dict:
    """name -> {"min", "max"} healthy range for this patient."""
    ranges = {}
    base = baseline_dictionary(healthy_dictionaries(), record["age"], record["sex"])
    if base:
        for name, spec in base["parameters"].items():
            ranges[name] = {"min": spec.get("min"), "max": spec.get("max")}

    labs = dict(HEALTHY_LABS)
    if record["age"] < CHILD_AGE:
        labs["WBC Count"] = CHILD_WBC
    for name, spec in labs.items():
        ranges[name] = {"min": spec["min"], "max": spec["max"]}

    hb = ranges.get("Hemoglobin")       # hematocrit is about 3 x hemoglobin
    if hb and hb["min"] is not None and hb["max"] is not None:
        ranges["Hematocrit"] = {"min": round(hb["min"] * 3, 1), "max": round(hb["max"] * 3, 1)}
    return ranges


def flag(value: float | None, rng: dict | None) -> str:
    if value is None or not rng:
        return "no reference"
    if rng.get("min") is not None and value < rng["min"]:
        return "LOW"
    if rng.get("max") is not None and value > rng["max"]:
        return "HIGH"
    return "normal"


def range_text(rng: dict | None, unit: str) -> str:
    if not rng:
        return "-"
    lo, hi = rng.get("min"), rng.get("max")
    if lo is not None and hi is not None:
        return f"{lo}-{hi} {unit}"
    return f"<= {hi} {unit}" if lo is None else f">= {lo} {unit}"


def pulse_pressure(record: dict) -> float | None:
    vitals = record.get("vitals", {})
    if "Systolic Blood Pressure" in vitals and "Diastolic Blood Pressure" in vitals:
        sys_val = vitals["Systolic Blood Pressure"].get("value")
        dia_val = vitals["Diastolic Blood Pressure"].get("value")
        if sys_val is not None and dia_val is not None:
            return round(sys_val - dia_val, 1)
    return None


def warning_signs(record: dict) -> list[str]:
    signs = []
    pp = pulse_pressure(record)
    if pp is not None and pp <= NARROW_PULSE_PRESSURE:
        signs.append(f"narrow pulse pressure ({pp} mmHg)")
    platelets = record.get("vitals", {}).get("Platelet Count")
    if platelets and platelets.get("value") is not None and platelets["value"] < LOW_PLATELETS:
        signs.append(f"low platelet count ({platelets['value']} {platelets.get('unit', 'cells/µL')})")
    signs += [s for s, on in record.get("symptoms", {}).items() if on and s in WARNING_SIGN_SYMPTOMS]
    return signs


def vital_flags(record: dict) -> dict:
    """name -> LOW / HIGH / normal / no reference"""
    ranges = reference_ranges(record)
    return {name: flag(v["value"], ranges.get(name)) for name, v in record.get("vitals", {}).items()}


# ---------------------------------------------------------------------------
# Patient route
# ---------------------------------------------------------------------------
def patient_context(record: dict) -> str:
    ranges = reference_ranges(record)
    lines = [
        f"Patient: {record['name']} | {record['sex']} | age {record['age']} "
        f"(group {record.get('age_group', '?')}) | condition on record: "
        f"{record.get('condition', 'unknown')} | source: {record['source_file']} #{record['id']}",
        "Vitals (value | healthy range | flag):",
    ]
    for name, v in record.get("vitals", {}).items():
        rng = ranges.get(name)
        lines.append(f"- {name}: {v['value']} {v['unit']} | {range_text(rng, v['unit'])} | {flag(v['value'], rng)}")

    pp = pulse_pressure(record)
    if pp is not None:
        lines.append(f"- Pulse Pressure: {pp} mmHg | > {NARROW_PULSE_PRESSURE} mmHg | "
                     f"{'NARROW' if pp <= NARROW_PULSE_PRESSURE else 'normal'}")

    symptoms = record.get("symptoms", {})
    present = [s for s, on in symptoms.items() if on]
    absent = [s for s, on in symptoms.items() if not on]
    signs = warning_signs(record)
    lines.append(f"Symptoms PRESENT: {', '.join(present) if present else 'none'}")
    lines.append(f"Symptoms ABSENT: {', '.join(absent) if absent else 'none'}")
    lines.append(f"Dengue warning signs found: {', '.join(signs) if signs else 'none'}")
    return "\n".join(lines)


def build_context(matches: list[dict]) -> str:
    parts = [patient_context(r) for r in matches[:MAX_PATIENTS]]
    if len(matches) > MAX_PATIENTS:
        parts.append(f"({len(matches) - MAX_PATIENTS} more matching patients not shown.)")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset route: plan
# ---------------------------------------------------------------------------
def plan_schema(vitals: list[str], symptoms: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "condition": {"type": "string", "enum": ["any", "healthy", "dengue"]},
            "sex": {"type": "string", "enum": ["any", "male", "female"]},
            "age_min": {"type": "integer"},
            "age_max": {"type": "integer"},
            "symptoms": {"type": "array", "items": {"type": "string", "enum": symptoms}},
            "abnormal": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "vital": {"type": "string", "enum": vitals},
                        "flag": {"type": "string", "enum": ["LOW", "HIGH", "ANY"]},
                    },
                    "required": ["vital", "flag"],
                },
            },
            "warning_signs_only": {"type": "boolean"},
            "sort_by": {"type": "string", "enum": ["none", "age"] + vitals},
            "order": {"type": "string", "enum": ["asc", "desc"]},
            "limit": {"type": "integer"},
        },
        "required": list(DEFAULT_PLAN),
    }


def clean_plan(raw: dict, vitals: list[str], symptoms: list[str]) -> dict:
    """Keep only valid values; anything else falls back to DEFAULT_PLAN."""
    plan = json.loads(json.dumps(DEFAULT_PLAN))
    if raw.get("condition") in ("any", "healthy", "dengue"):
        plan["condition"] = raw["condition"]
    if raw.get("sex") in ("any", "male", "female"):
        plan["sex"] = raw["sex"]
    for key in ("age_min", "age_max", "limit"):
        if isinstance(raw.get(key), int) and raw[key] >= 0:
            plan[key] = raw[key]
    if plan["age_max"] > 100:
        plan["age_max"] = DEFAULT_PLAN["age_max"]
    if plan["age_min"] > plan["age_max"]:
        plan["age_min"], plan["age_max"] = DEFAULT_PLAN["age_min"], DEFAULT_PLAN["age_max"]
    plan["symptoms"] = [s for s in raw.get("symptoms", []) if s in symptoms]
    plan["abnormal"] = [a for a in raw.get("abnormal", [])
                        if isinstance(a, dict) and a.get("vital") in vitals
                        and a.get("flag") in ("LOW", "HIGH", "ANY")]
    plan["warning_signs_only"] = raw.get("warning_signs_only") is True
    if raw.get("sort_by") in ["age"] + vitals:
        plan["sort_by"] = raw["sort_by"]
    if raw.get("order") in ("asc", "desc"):
        plan["order"] = raw["order"]
    return plan


def age_range(q: str) -> tuple[int, int] | None:
    """Age limits stated in the question (lower-case), or None."""
    lo, hi = DEFAULT_PLAN["age_min"], DEFAULT_PLAN["age_max"]
    if m := re.search(r"\b(?:between|aged?|ages)\s+(\d{1,3})\s*(?:and|to|-)\s*(\d{1,3})\b", q):
        lo, hi = sorted((int(m[1]), int(m[2])))
    elif m := re.search(r"\baged?\s+(\d{1,3})\b", q):
        lo = hi = int(m[1])
    else:
        over = (re.search(rf"\b{AGE_NOUN}\s+(?:(?:who\s+)?(?:are|is)\s+)?(?:over|above|older than)\s+(\d{{1,3}})\b", q)
                or re.search(r"\b(?:over|above|older than)\s+(\d{1,3})\s*(?:years|yrs|y/?o)?\b", q)
                or re.search(r"\b(\d{1,3})\s*(?:\+|years? (?:and|or) (?:older|over|above))", q))
        under = (re.search(rf"\b{AGE_NOUN}\s+(?:(?:who\s+)?(?:are|is)\s+)?(?:under|below|younger than)\s+(\d{{1,3}})\b", q)
                 or re.search(r"\b(?:under|below|younger than)\s+(\d{1,3})\s*(?:years|yrs|y/?o)?\b", q))
        if over:
            lo = int(over[1]) + (0 if "+" in over[0] or "and" in over[0] or "or" in over[0] else 1)
        if under:
            hi = int(under[1]) - 1
        if re.search(r"\b(children|kids|child|paediatric|pediatric)\b", q):
            hi = min(hi, 17)
        if re.search(r"\b(teenagers?|teens?)\b", q):
            lo, hi = max(lo, 13), min(hi, 19)
        if re.search(r"\badults?\b", q) and not re.search(r"\bolder adults?\b", q):
            lo = max(lo, 18)
        if re.search(r"\b(elderly|seniors?|older adults?)\b", q):
            lo = max(lo, 65)
    hi = min(hi, 100)
    if lo > 100 or lo > hi or (lo, hi) == (DEFAULT_PLAN["age_min"], DEFAULT_PLAN["age_max"]):
        return None
    return lo, hi


def vital_mentions(q: str, vitals: list[str]) -> list[tuple[str, int, int]]:
    """(vital, start, end) for every vital named in the question, in order."""
    keywords = sorted(((kw, v) for v in vitals
                       for kw in [v.lower()] + VITAL_KEYWORDS.get(v, [])), key=lambda x: -len(x[0]))
    found, taken = [], [False] * len(q)
    for kw, vital in keywords:     # longest first: "hdl cholesterol" before "cholesterol"
        for m in re.finditer(rf"\b{re.escape(kw)}\b", q):
            if not any(taken[m.start():m.end()]):
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
                found.append((vital, m.start(), m.end()))
    seen, ordered = set(), []
    for vital, s, e in sorted(found, key=lambda x: x[1]):
        if vital not in seen:
            seen.add(vital)
            ordered.append((vital, s, e))
    return ordered


def flag_near(q: str, start: int, end: int) -> str | None:
    """LOW / HIGH / ANY from words just before or after a vital mention."""
    window = q[max(0, start - 30):start] + " | " + q[end:end + 30]
    if re.search(rf"\b({ABNORMAL_WORDS})\b", window):
        return "ANY"
    if re.search(rf"\b({LOW_WORDS})\b", window):
        return "LOW"
    if re.search(rf"\b({HIGH_WORDS})\b", window):
        return "HIGH"
    return None


def ground_plan(question: str, plan: dict, vitals: list[str], symptoms: list[str],
                base: dict | None = None) -> dict:
    """
    Correct the plan with what the question states explicitly. Anything the
    question states (age, sex, dengue / healthy, sorting, top N, low / high
    next to a vital) wins; the planner's filters are only kept when the vital
    or symptom they refer to is mentioned.

    `base` is the previous search when the question narrows it ("how many of
    them are female?"): whatever the question does not state is taken from it.
    """
    q = question.lower()
    plan = json.loads(json.dumps(plan))
    start = json.loads(json.dumps(base or DEFAULT_PLAN))

    rest = re.sub(r"dengue warning signs?", "", q)
    if re.search(r"\b(non-dengue|without dengue|no dengue|healthy)\b", q):
        plan["condition"] = "healthy"
    elif re.search(r"\bdengue\b", rest):
        plan["condition"] = "dengue"
    else:
        plan["condition"] = start["condition"]

    female = re.search(r"\b(females?|wom[ae]n|girls?|ladies)\b", q)
    male = re.search(r"\b(males?|m[ae]n|boys?|gentlemen)\b", q)
    plan["sex"] = "female" if female and not male else "male" if male and not female else start["sex"]

    plan["age_min"], plan["age_max"] = age_range(q) or (start["age_min"], start["age_max"])
    plan["warning_signs_only"] = bool(re.search(r"\bwarning signs?\b|\bsevere\b", q)) or start["warning_signs_only"]

    grounded = [s for s in plan["symptoms"]
                if any(re.search(rf"\b{re.escape(w)}", q)
                       for w in re.findall(r"[a-z]+", s.lower()) if w not in GENERIC_SYMPTOM_WORDS)]
    plan["symptoms"] = start["symptoms"] + [s for s in grounded if s not in start["symptoms"]]

    mentions = vital_mentions(q, vitals)
    plan["sort_by"], plan["order"] = start["sort_by"], start["order"]
    if re.search(rf"\b({SORT_ASC}|{SORT_DESC}|top)\b", q):
        if re.search(r"\b(oldest|youngest)\b", q):
            plan["sort_by"] = "age"
        elif mentions:
            plan["sort_by"] = mentions[0][0]
        plan["order"] = "asc" if re.search(rf"\b({SORT_ASC})\b", q) else "desc"

    planner_flags = {a["vital"]: a["flag"] for a in plan["abnormal"]}
    abnormal = {a["vital"]: a["flag"] for a in start["abnormal"]}
    for vital, s, e in mentions:
        if vital == plan["sort_by"]:
            continue
        f = flag_near(q, s, e) or planner_flags.get(vital)
        if f:
            abnormal[vital] = f
    plan["abnormal"] = [{"vital": v, "flag": f} for v, f in abnormal.items()]

    plan["limit"] = start["limit"] if base else 0
    if m := re.search(r"\b(?:top|first|bottom)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\b", q) \
            or re.search(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(?:patients|people|records|persons)\b", q):
        plan["limit"] = int(m[1]) if m[1].isdigit() else NUMBER_WORDS[m[1]]
    elif re.search(r"\bwho (?:has|have|is) the\b", q) and plan["sort_by"] != "none":
        plan["limit"] = 1
    elif base and re.search(r"\bhow many\b", q):
        plan["limit"] = 0       # counting: all of the narrowed matches
    return plan


def plan_query(question: str, records: list[dict], model: str, base: dict | None = None) -> dict:
    vitals = sorted({v for r in records for v in r.get("vitals", {})})
    symptoms = sorted({s for r in records for s in r.get("symptoms", {})})
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "system", "content": PLAN_PROMPT},
                      {"role": "user", "content": question}],
            format=plan_schema(vitals, symptoms),
            options={"temperature": 0, "num_ctx": NUM_CTX},
        )
        raw = json.loads(response["message"]["content"])
    except Exception as exc:
        print(f"  ! Query planning failed, searching all records: {exc}")
        raw = {}
    return ground_plan(question, clean_plan(raw, vitals, symptoms), vitals, symptoms, base)


def plan_text(plan: dict) -> str:
    parts = []
    if plan["condition"] != "any":
        parts.append(f"condition = {plan['condition']}")
    if plan["sex"] != "any":
        parts.append(f"sex = {plan['sex']}")
    if (plan["age_min"], plan["age_max"]) != (DEFAULT_PLAN["age_min"], DEFAULT_PLAN["age_max"]):
        parts.append(f"age {plan['age_min']}-{plan['age_max']}")
    if plan["symptoms"]:
        parts.append("symptoms: " + ", ".join(plan["symptoms"]))
    for a in plan["abnormal"]:
        parts.append(f"{a['vital']} {'LOW or HIGH' if a['flag'] == 'ANY' else a['flag']}")
    if plan["warning_signs_only"]:
        parts.append("has dengue warning signs")
    if plan["sort_by"] != "none":
        parts.append(f"sorted by {plan['sort_by']} {'lowest' if plan['order'] == 'asc' else 'highest'} first")
    if plan["limit"]:
        parts.append(f"limit {plan['limit']}")
    return "; ".join(parts) or "all records"


# ---------------------------------------------------------------------------
# Dataset route: retrieve
# ---------------------------------------------------------------------------
def sort_value(record: dict, key: str):
    if key == "age":
        return record["age"]
    v = record.get("vitals", {}).get(key)
    return v["value"] if v else None


def run_plan(plan: dict, records: list[dict], flags: dict) -> list[dict]:
    hits = []
    for r in records:
        f = flags.get(id(r), {})
        if plan["condition"] != "any" and str(r.get("condition", "")).lower() != plan["condition"].lower():
            continue
        if plan["sex"] != "any" and str(r.get("sex", "")).lower() != plan["sex"].lower():
            continue
        if not plan["age_min"] <= r.get("age", 0) <= plan["age_max"]:
            continue
        if not all(r.get("symptoms", {}).get(s) for s in plan["symptoms"]):
            continue
        if not all(f.get(a["vital"]) in (("LOW", "HIGH") if a["flag"] == "ANY" else (a["flag"],))
                   for a in plan["abnormal"]):
            continue
        if plan["warning_signs_only"] and not warning_signs(r):
            continue
        hits.append(r)

    if plan["sort_by"] != "none":
        hits = [r for r in hits if sort_value(r, plan["sort_by"]) is not None]
        hits.sort(key=lambda r: sort_value(r, plan["sort_by"]), reverse=plan["order"] == "desc")
    return hits


def median(values: list[float]) -> str:
    return f"{statistics.median(values):g}" if values else "-"


def dataset_overview(records: list[dict], flags: dict) -> str:
    if not records:
        return "Dataset: 0 records."
    files = sorted({r["source_file"] for r in records if "source_file" in r})
    conditions = Counter(r.get("condition", "unknown") for r in records)
    sexes = Counter(r.get("sex", "unknown") for r in records)
    ages = [r["age"] for r in records if "age" in r]
    age_str = f"{min(ages)}-{max(ages)} (median {median(ages)})" if ages else "none"
    lines = [
        f"Dataset: {len(records)} records in {len(files)} file(s): {', '.join(files)}",
        "Conditions: " + ", ".join(f"{k} {v}" for k, v in sorted(conditions.items()))
        + " | Sex: " + ", ".join(f"{k} {v}" for k, v in sorted(sexes.items()))
        + f" | Age: {age_str}",
        f"Patients with dengue warning signs: {sum(1 for r in records if warning_signs(r))}",
        "Vitals (median healthy / median dengue | patients LOW | patients HIGH):",
    ]
    units, by_condition, lows, highs = {}, {}, Counter(), Counter()
    for r in records:
        for name, v in r.get("vitals", {}).items():
            units[name] = v["unit"]
            by_condition.setdefault((name, r.get("condition")), []).append(v["value"])
            lows[name] += flags[id(r)].get(name) == "LOW"
            highs[name] += flags[id(r)].get(name) == "HIGH"
    for name, unit in units.items():
        lines.append(f"- {name}: {median(by_condition.get((name, 'healthy'), []))} / "
                     f"{median(by_condition.get((name, 'dengue'), []))} {unit} | {lows[name]} | {highs[name]}")

    symptom_counts = Counter()
    for r in records:
        for s, on in r.get("symptoms", {}).items():
            symptom_counts[(s, r.get("condition"))] += bool(on)
    names = sorted({s for s, _ in symptom_counts})
    if names:
        lines.append("Symptoms present (dengue patients / healthy patients): " + ", ".join(
            f"{s} {symptom_counts[(s, 'dengue')]}/{symptom_counts[(s, 'healthy')]}" for s in names))
    return "\n".join(lines)


def record_row(n: int, record: dict, flags: dict, plan: dict) -> str:
    vitals = record.get("vitals", {})
    shown = [plan["sort_by"]] if plan["sort_by"] not in ("none", "age") else []
    shown += [a["vital"] for a in plan["abnormal"] if a["vital"] not in shown]
    key_values = [f"{k} {vitals[k].get('value')} {vitals[k].get('unit', '')} ({flags.get(k, 'normal')})"
                  for k in shown if k in vitals]
    abnormal = [f"{k} {v.get('value')} {flags.get(k, 'abnormal')}" for k, v in vitals.items()
                if flags.get(k) in ("LOW", "HIGH") and k not in shown]
    present = [s for s, on in record.get("symptoms", {}).items() if on]
    signs = warning_signs(record)
    parts = [f"{n}. {record.get('name', 'Unknown')}", f"{record.get('sex', '?')} {record.get('age', '?')}", record.get("condition", "?")]
    if key_values:
        parts.append("; ".join(key_values))
    parts += [f"other abnormal: {', '.join(abnormal) or 'none'}",
              f"symptoms: {', '.join(present) or 'none'}",
              f"warning signs: {', '.join(signs) or 'none'}"]
    return " | ".join(parts)


def record_key(record: dict) -> tuple:
    return record["source_file"], record["id"]


def retrieve_dataset(question: str, model: str = LLM_MODEL, base: dict | None = None,
                     within: list[dict] | None = None) -> dict:
    """
    Plan the query with Ollama, run it on all records, build the context.
    `base` is the previous search plan when the question narrows it; `within`
    limits the search to the patients of the previous list ("how many of
    them are female?" after a top 3).
    """
    from user_input import load_records

    records = load_records()
    if not records:
        return {"plan": DEFAULT_PLAN, "matches": [], "shown": [], "context": ""}
    flags = {id(r): vital_flags(r) for r in records}
    plan = plan_query(question, records, model, base)
    pool = records
    if within:
        keys = {record_key(r) for r in within}
        pool = [r for r in records if record_key(r) in keys]
        plan["limit"] = 0
    matches = run_plan(plan, pool, flags)
    shown = matches[:min(plan["limit"] or MAX_ROWS, MAX_ROWS)]

    narrowed = ""
    if within:
        narrowed = f" (only among the {len(within)} patients of the previous list)"
    elif base:
        narrowed = " (narrowed from the previous search)"
    lines = [dataset_overview(records, flags), "",
             f"Search: {plan_text(plan)}{narrowed}",
             f"Matching records: {len(matches)}"
             + (f" (showing {len(shown)})" if len(shown) < len(matches) else "")]
    lines += [record_row(i, r, flags[id(r)], plan) for i, r in enumerate(shown, 1)]
    return {"plan": plan, "matches": matches, "shown": shown, "context": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
def message_text(content) -> str:
    """Chat message content as plain text (Gradio may send a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(message_text(c) for c in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content or "")


def chat_turns(history: list) -> list[dict]:
    turns = []
    for m in (history or []):
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            turns.append({"role": m["role"], "content": message_text(m.get("content", ""))})
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            u, a = m
            if u:
                turns.append({"role": "user", "content": message_text(u)})
            if a:
                turns.append({"role": "assistant", "content": message_text(a)})
    return turns


def summarize_turns(summary: str, turns: list[dict], model: str) -> str:
    """Fold `turns` into the running conversation summary."""
    transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "system", "content": SUMMARY_PROMPT},
                      {"role": "user", "content": f"Current summary:\n{summary or '(none)'}\n\n"
                                                  f"New conversation turns:\n{transcript}"}],
            options={"temperature": 0, "num_ctx": NUM_CTX},
        )
        text = response["message"]["content"].strip()
        return re.sub(r"^\s*(current |updated )?summary:?\s*\n", "", text, flags=re.IGNORECASE)
    except Exception as exc:
        print(f"  ! Summarising failed, keeping the turns as text: {exc}")
        return (summary + "\n" + transcript)[-HISTORY_CHAR_BUDGET:]


def prepare_history(history: list, session: dict, model: str = LLM_MODEL) -> list[dict]:
    """
    Earlier turns to send with the next question. Nothing is dropped: when the
    turns get longer than HISTORY_CHAR_BUDGET, the oldest ones are folded into
    session["summary"] (session["summarized"] counts the folded messages).
    """
    turns = chat_turns(history)
    recent = turns[session["summarized"]:]
    while sum(len(t["content"]) for t in recent) > HISTORY_CHAR_BUDGET and len(recent) > 2:
        fold = max(2, len(recent) // 2 // 2 * 2)        # whole question/answer pairs
        session["summary"] = summarize_turns(session["summary"], recent[:fold], model)
        session["summarized"] += fold
        recent = turns[session["summarized"]:]
    return recent


def build_messages(system: str, user: str, history: list[dict] | None, summary: str) -> list[dict]:
    """System prompt, recent turns, then the question (with the summary of older turns)."""
    if summary:
        user = f"Summary of the earlier conversation (the user is the person asking):\n{summary}\n\n{user}"
    return [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def list_models() -> list[str]:
    try:
        return [m["model"] for m in client.list()["models"]]
    except Exception:
        return []


def stream_chat(messages: list[dict], model: str):
    """Yields the answer as it grows (for streaming in the UI)."""
    answer = ""
    try:
        for chunk in client.chat(model=model, messages=messages, stream=True,
                                 options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX}):
            answer += chunk["message"]["content"]
            yield answer
    except Exception as exc:
        yield f"{answer}\n\n**Ollama error:** {exc} (is Ollama running at {OLLAMA_HOST}?)"


def generate_insight(question: str, matches: list[dict], model: str = LLM_MODEL,
                     history: list[dict] | None = None, summary: str = ""):
    """Patient route: insight on the named patients, aware of the earlier conversation."""
    if not matches:
        yield ("I could not find that patient in the generated records. "
               "Please include the patient's full name, e.g. *How is Maria Lopez doing?*")
        return
    user = f"Patient data:\n{build_context(matches)}\n\nQuestion: {question}"
    yield from stream_chat(build_messages(SYSTEM_PROMPT, user, history, summary), model)


def answer_dataset(question: str, context: str, model: str = LLM_MODEL,
                   history: list[dict] | None = None, summary: str = ""):
    """Dataset route: answer from the retrieved context, aware of the earlier conversation."""
    if not context:
        yield "There are no records in BulkOutput/output yet. Run BulkOutput/bulkData.py first."
        return
    user = f"{context}\n\nQuestion: {question}"
    yield from stream_chat(build_messages(DATASET_PROMPT, user, history, summary), model)


def conversation_context(session: dict) -> str:
    return "\n".join([
        f"Patients discussed so far: {', '.join(session['discussed']) or 'none'}",
        f"Patient currently in focus: {', '.join(r['name'] for r in session['focus']) or 'none'}",
        "Searches run so far: " + ("; ".join(session["searches"]) or "none"),
    ])


def answer_conversation(question: str, session: dict, model: str = LLM_MODEL,
                        history: list[dict] | None = None):
    """Conversation route: questions about the chat itself and small talk."""
    user = f"Conversation log:\n{conversation_context(session)}\n\nMessage: {question}"
    yield from stream_chat(build_messages(CONVERSATION_PROMPT, user, history, session["summary"]), model)


if __name__ == "__main__":
    from user_input import analyze

    question = " ".join(sys.argv[1:]) or input("Question: ")
    found = analyze(question)
    print(f"route: {found['route']}\n")
    if found["route"] == "dataset":
        context = retrieve_dataset(question)["context"]
        answer = answer_dataset(question, context)
    else:
        context = build_context(found["matches"])
        answer = generate_insight(question, found["matches"])
    print(context, "\n")
    text = ""
    for text in answer:
        pass
    print(text)
