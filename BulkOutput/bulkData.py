"""
bulkData.py

Generates random people with vital data, mixing healthy people and dengue
patients, using the cleaned dictionaries in
    Data_Distl/output         (healthy ranges)
    Data_Distl/output_dengue  (dengue ranges and symptom prevalence)

Steps
-----
1. Input   : how many records to create.
2. Ollama  : generates that many unique names, each with a sex and an age.
             Ages are kept inside the age range covered by the dictionaries
             (re-drawn at random if Ollama gives one that is not covered).
3. Vitals  : every person gets a healthy baseline from the healthy dictionary
             for their age group and sex (the other sex of the same age group
             is used if there is none, e.g. young girls). Most values fall
             inside the healthy range; UNHEALTHY_RATE of them fall outside it.
             Healthy people also get normal Platelet / WBC / Hematocrit values
             and no dengue symptoms.
4. Dengue  : DENGUE_RATE of the people are dengue patients. For them the
             dengue dictionary for their age group and sex replaces the
             matching values (temperature, heart rate, systolic pressure,
             platelets, WBC, hematocrit) and each symptom is present with its
             prevalence in that dictionary.
5. Output  : BulkOutput/output/<DDMMYY>_vital_<HH>_<count>.json and .csv
             e.g. 10 records at 11:xx on 23 Oct 2026 -> 231026_vital_11_10.json (.csv)

Usage
    python bulkData.py
    python bulkData.py 100
    python bulkData.py 100 --unhealthy-rate 0.3
    python bulkData.py 100 --dengue-rate 0.5
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
HEALTHY_DIR = BASE_DIR.parent / "Data_Distl" / "output"
DENGUE_DIR = BASE_DIR.parent / "Data_Distl" / "output_dengue"
OUTPUT_DIR = BASE_DIR / "output"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from export_csv import export_records_to_csv

OLLAMA_HOST = "http://localhost:11434"
client = ollama.Client(host=OLLAMA_HOST)
def _detect_model():
    try:
        m = [x["model"] for x in client.list()["models"]]
        return "llama3.2:3b" if "llama3.2:3b" in m else (m[0] if m else "llama3.2:3b")
    except Exception:
        return "llama3.2:3b"

LLM_MODEL = _detect_model()
NAMES_PER_CALL = 25         # names requested from Ollama per call
MAX_OLLAMA_CALLS = 200      # safety stop if Ollama keeps returning duplicates

UNHEALTHY_RATE = 0.15       # chance that a single value falls outside its range
OUT_OF_RANGE_SPREAD = 0.4   # how far outside, as a fraction of the range width
DENGUE_RATE = 0.3           # share of people who are dengue patients

# Used when a healthy range is open on one side ("<= 200 mg/dL", ">= 150 min/week")
# to get a realistic value to draw from.
OPEN_LOW_RATIO = {          # min = max * ratio
    "Total Cholesterol": 0.65,
    "LDL Cholesterol": 0.5,
    "Waist Circumference": 0.75,
}
OPEN_HIGH_RATIO = {         # max = min * ratio
    "HDL Cholesterol": 1.8,
    "Physical Activity Minutes per Week": 2.5,
}

# Normal blood counts for people without dengue (the healthy dictionaries do
# not have them). Hematocrit is derived from Hemoglobin (about 3 x Hb).
HEALTHY_LABS = {
    "Platelet Count": {"unit": "cells/µL", "min": 150000, "max": 400000},
    "WBC Count": {"unit": "cells/µL", "min": 4500, "max": 11000},
}
CHILD_WBC = {"unit": "cells/µL", "min": 5000, "max": 15000}   # age < 10
CHILD_AGE = 10

# Hard limits so out-of-range values stay physically possible.
VALUE_LIMITS = {            # name: (lowest, highest)
    "SpO2": (None, 100),
    "Platelet Count": (5000, None),
    "WBC Count": (800, None),
}

DECIMALS = {
    "Body Temperature": 1,
    "BMI": 1,
    "HbA1c": 1,
    "Hemoglobin": 1,
    "Hematocrit": 1,
    "Sleep Hours": 1,
    "Waist Circumference": 1,
}




# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------
def load_dictionaries(folder: Path, pattern: str) -> list[dict]:
    dicts = []
    for path in folder.glob(pattern):
        data = json.loads(path.read_text(encoding="utf-8"))
        dicts.append(data)
    if not dicts:
        sys.exit(f"No cleaned dictionaries found in {folder}. Run Data_Distl/dataCleaner.py first.")
    return sorted(dicts, key=lambda d: (d["sex"], d["age_lower"]))


def find_dictionary(dicts: list[dict], age: int, sex: str) -> dict | None:
    """Group with lower <= age < upper; the oldest group also includes its upper age."""
    groups = [d for d in dicts if d["sex"] == sex]
    for d in groups:
        if d["age_lower"] <= age < d["age_upper"]:
            return d
    for d in groups:
        if age == d["age_upper"]:
            return d
    return None


def baseline_dictionary(healthy: list[dict], age: int, sex: str) -> dict | None:
    """Healthy dictionary for the person; falls back to the other sex of the same age."""
    other = "female" if sex == "male" else "male"
    return find_dictionary(healthy, age, sex) or find_dictionary(healthy, age, other)


def covered_ages(healthy: list[dict], sex: str) -> list[int]:
    """Ages that have a healthy baseline for this sex (directly or via fallback)."""
    low = min(d["age_lower"] for d in healthy)
    high = max(d["age_upper"] for d in healthy)
    return [a for a in range(low, high + 1) if baseline_dictionary(healthy, a, sex)]


def symptom_names(dengue: list[dict]) -> list[str]:
    names = {}
    for d in dengue:
        for name, spec in d["parameters"].items():
            if spec.get("kind") == "symptom":
                names[name] = None
    return list(names)


# ---------------------------------------------------------------------------
# Ollama: names and ages
# ---------------------------------------------------------------------------
PEOPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sex": {"type": "string", "enum": ["male", "female"]},
                    "age": {"type": "integer"},
                },
                "required": ["name", "sex", "age"],
            },
        }
    },
    "required": ["people"],
}


def ask_people(count: int, age_low: int, age_high: int) -> list[dict]:
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You generate realistic fictional people. Respond only with JSON."},
            {"role": "user", "content": (
                f"Generate {count} different people. Each has a realistic full name "
                "(first and last name) from a mix of cultures and countries, a sex "
                "(male or female) that matches the first name, and an age between "
                f"{age_low} and {age_high}. Use roughly equal numbers of males and "
                "females and spread the ages across the whole range. Do not repeat names."
            )},
        ],
        format=PEOPLE_SCHEMA,
        options={"temperature": 1.0, "seed": random.randint(0, 2**31 - 1)},
    )
    return json.loads(response["message"]["content"]).get("people", [])


def generate_people(total: int, healthy: list[dict]) -> list[dict]:
    ages_by_sex = {sex: covered_ages(healthy, sex) for sex in ("male", "female")}
    all_ages = ages_by_sex["male"] + ages_by_sex["female"]
    age_low, age_high = min(all_ages), max(all_ages)

    people, seen, calls = [], set(), 0
    while len(people) < total and calls < MAX_OLLAMA_CALLS:
        calls += 1
        want = min(NAMES_PER_CALL, total - len(people))
        try:
            batch = ask_people(want, age_low, age_high)
        except Exception as exc:
            print(f"  ! Ollama call failed: {exc}")
            continue
        for p in batch:
            name = " ".join(str(p.get("name", "")).split()).title()
            sex = str(p.get("sex", "")).lower()
            if len(name.split()) < 2 or name.lower() in seen or sex not in ages_by_sex:
                continue
            if not ages_by_sex[sex]:
                continue
            age = p.get("age")
            if not isinstance(age, int) or age not in ages_by_sex[sex]:
                age = random.choice(ages_by_sex[sex])
            seen.add(name.lower())
            people.append({"name": name, "sex": sex, "age": age})
            if len(people) == total:
                break
        print(f"  names: {len(people)}/{total}")

    if len(people) < total:
        sys.exit(f"Ollama only produced {len(people)} unique names after {calls} calls.")
    random.shuffle(people)
    return people


# ---------------------------------------------------------------------------
# Vital data
# ---------------------------------------------------------------------------
def draw_value(name: str, spec: dict, unhealthy_rate: float) -> float:
    lo, hi = spec.get("min"), spec.get("max")
    if lo is None and hi is None:
        return None
    if lo is None:
        lo = hi * OPEN_LOW_RATIO.get(name, 0.7)
    if hi is None:
        hi = lo * OPEN_HIGH_RATIO.get(name, 1.5)

    width = hi - lo
    if random.random() < unhealthy_rate:
        # Outside the range, below or above.
        offset = random.uniform(0.05, OUT_OF_RANGE_SPREAD) * width
        value = lo - offset if random.random() < 0.5 else hi + offset
    else:
        value = random.triangular(lo, hi)

    floor, ceiling = VALUE_LIMITS.get(name, (None, None))
    if floor is not None:
        value = max(value, floor)
    if ceiling is not None:
        value = min(value, ceiling)
    value = max(value, 0)
    decimals = DECIMALS.get(name, 0)
    return round(value, decimals) if decimals else int(round(value))


def draw_numeric(parameters: dict, unhealthy_rate: float) -> dict:
    """A value for every numeric parameter (symptoms are skipped)."""
    vitals = {}
    for name, spec in parameters.items():
        if spec.get("kind", "numeric") != "numeric":
            continue
        value = draw_value(name, spec, unhealthy_rate)
        if value is not None:
            vitals[name] = {"value": value, "unit": spec["unit"]}
    return vitals


def healthy_labs(age: int, vitals: dict, unhealthy_rate: float) -> dict:
    specs = dict(HEALTHY_LABS)
    if age < CHILD_AGE:
        specs["WBC Count"] = CHILD_WBC
    labs = draw_numeric(specs, unhealthy_rate)

    hb = vitals.get("Hemoglobin")
    if hb:
        hct = hb["value"] * 3 + random.uniform(-1.5, 1.5)
        labs["Hematocrit"] = {"value": round(hct, DECIMALS["Hematocrit"]), "unit": "%"}
    return labs


def fix_blood_pressure(vitals: dict):
    """Diastolic pressure must stay well below systolic."""
    sys_bp, dia_bp = vitals.get("Systolic Blood Pressure"), vitals.get("Diastolic Blood Pressure")
    if sys_bp and dia_bp and dia_bp["value"] > sys_bp["value"] - 20:
        dia_bp["value"] = max(30, sys_bp["value"] - random.randint(20, 45))


def generate_record(person: dict, healthy: list[dict], dengue: list[dict], symptoms: list[str],
                    unhealthy_rate: float, dengue_rate: float) -> dict:
    base = baseline_dictionary(healthy, person["age"], person["sex"])
    vitals = draw_numeric(base["parameters"], unhealthy_rate)
    vitals.update(healthy_labs(person["age"], vitals, unhealthy_rate))
    condition = "healthy"
    present = {name: False for name in symptoms}

    dengue_dict = find_dictionary(dengue, person["age"], person["sex"])
    if dengue_dict and random.random() < dengue_rate:
        condition = "dengue"
        vitals.update(draw_numeric(dengue_dict["parameters"], unhealthy_rate))
        for name, spec in dengue_dict["parameters"].items():
            if spec.get("kind") == "symptom":
                present[name] = random.random() < spec["prevalence_percent"] / 100

    fix_blood_pressure(vitals)
    return {
        **person,
        "age_group": f"{base['age_lower']}-{base['age_upper']}",
        "condition": condition,
        "vitals": vitals,
        "symptoms": present,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def output_path(count: int, now: datetime) -> Path:
    """<DDMMYY>_vital_<HH>_<count>.json; adds _2, _3 ... instead of overwriting."""
    stem = f"{now:%d%m%y}_vital_{now:%H}_{count}"
    path = OUTPUT_DIR / f"{stem}.json"
    n = 2
    while path.exists() or path.with_suffix(".csv").exists():
        path = OUTPUT_DIR / f"{stem}_{n}.json"
        n += 1
    return path


def read_count(arg: int | None) -> int:
    if arg is None:
        text = input("How many records do you want to create? ").strip()
        if not text.isdigit():
            sys.exit(f"Invalid number: '{text}'")
        arg = int(text)
    if arg < 1:
        sys.exit("Number of records must be at least 1.")
    return arg


def main():
    parser = argparse.ArgumentParser(description="Generate random people with vital data.")
    parser.add_argument("count", type=int, nargs="?", help="number of records to create")
    parser.add_argument("--unhealthy-rate", type=float, default=UNHEALTHY_RATE,
                        help=f"chance each value is outside its range (default {UNHEALTHY_RATE})")
    parser.add_argument("--dengue-rate", type=float, default=DENGUE_RATE,
                        help=f"share of people who are dengue patients (default {DENGUE_RATE})")
    args = parser.parse_args()
    if not 0 <= args.unhealthy_rate <= 1:
        sys.exit("--unhealthy-rate must be between 0 and 1.")
    if not 0 <= args.dengue_rate <= 1:
        sys.exit("--dengue-rate must be between 0 and 1.")

    count = read_count(args.count)
    healthy = load_dictionaries(HEALTHY_DIR, "*_healty_*_dictonary.json")
    dengue = load_dictionaries(DENGUE_DIR, "*_dengue_*_dictonary.json") if args.dengue_rate else []
    symptoms = symptom_names(dengue)
    try:
        client.list()
    except Exception:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_HOST}. Is it running?")

    print(f"[1/2] Generating {count} names and ages with Ollama ...")
    people = generate_people(count, healthy)

    print("[2/2] Generating vital data ...")
    records = []
    for i, person in enumerate(people, 1):
        record = generate_record(person, healthy, dengue, symptoms,
                                 args.unhealthy_rate, args.dengue_rate)
        records.append({"id": i, **record})

    conditions = {"healthy": 0, "dengue": 0}
    for r in records:
        conditions[r["condition"]] += 1

    now = datetime.now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = output_path(count, now)
    path.write_text(json.dumps({
        "generated_at": now.isoformat(timespec="seconds"),
        "count": len(records),
        "conditions": conditions,
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    csv_records = [{**r, "source_file": path.name} for r in records]
    export_records_to_csv(csv_records, output_path=csv_path)

    print(f"\nSaved {len(records)} records ({conditions['healthy']} healthy, "
          f"{conditions['dengue']} dengue) to {path}")
    print(f"Saved CSV export to {csv_path}")


if __name__ == "__main__":
    main()
