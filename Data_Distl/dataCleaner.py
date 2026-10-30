"""
dataCleaner.py

Data distillation pipeline for the scraped dictionaries in Category_Dict.
Every value is checked against the reference in base_dictonary.py and fixed
when it is wrong. Two datasets:

    healthy : Category_Dict/output        -> Data_Distl/output
              (category_good.py, checked against normal human levels)
    dengue  : Category_Dict/output_dengue -> Data_Distl/output_dengue
              (category_dengue.py, checked against WHO dengue values; see
              clean_dengue_file below)

For each healthy file
---------------------
1. Map parameter names to one canonical name ("BMI" / "Body Mass Index (BMI)"
   -> "BMI"). The combined "Blood Pressure" is split into systolic/diastolic.
2. Normalise units: °F -> °C, inches -> cm, mmol/L -> mg/dL, g/L -> g/dL.
   Missing min/max are read from the healthy_value text ("70-99 mg/dL").
3. Check every bound:
     - outside the humanly possible limits     -> rejected, replaced
     - min > max                               -> rejected, replaced
     - further than `tolerance` from reference -> replaced
     - missing                                 -> filled from reference
     - within tolerance                        -> kept (scraped value)
   Low-confidence values go through the same check; if they agree with the
   reference they are kept and marked "verified".
4. Parameters the scraper never found are added from the reference.
5. Save to Data_Distl/output/<same file name>, with the original value of
   every changed parameter kept under "original" for auditing.

Usage
-----
    python dataCleaner.py                          # asks which dataset and how many files
    python dataCleaner.py --dataset dengue --batch 5
    python dataCleaner.py --dataset healthy --batch all
    python dataCleaner.py --overwrite              # also re-clean already cleaned files
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from base_dictonary import (
    BLOOD_PRESSURE,
    DENGUE_PARAMETERS,
    DENGUE_SPECIFIC,
    PARAMETERS,
    format_range,
    get_dengue_reference,
    get_reference,
    resolve_dengue_parameter,
    resolve_parameter,
)

BASE_DIR = Path(__file__).resolve().parent
CATEGORY_DIR = BASE_DIR.parent / "Category_Dict"
DATASETS = {
    "healthy": {
        "input": CATEGORY_DIR / "output",
        "pattern": "*_healty_*_dictonary.json",
        "output": BASE_DIR / "output",
    },
    "dengue": {
        "input": CATEGORY_DIR / "output_dengue",
        "pattern": "*_dengue_*_dictonary.json",
        "output": BASE_DIR / "output_dengue",
    },
}

NUMBER = r"(\d+(?:\.\d+)?)"
DASH = r"\s*(?:-|–|—|to)\s*"


# ---------------------------------------------------------------------------
# Parsing the scraped text
# ---------------------------------------------------------------------------
def parse_range_text(text: str) -> tuple[float | None, float | None]:
    """'70-99 mg/dL' -> (70, 99); 'Below 5.7%' -> (None, 5.7); 'at least 150' -> (150, None)."""
    if not text:
        return None, None
    t = text.lower().replace(",", "")
    m = re.search(NUMBER + DASH + NUMBER, t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(?:below|less than|under|<=?|up to|no more than)\s*" + NUMBER, t)
    if m:
        return None, float(m.group(1))
    m = re.search(r"(?:at least|above|more than|over|>=?|minimum of)\s*" + NUMBER, t)
    if m:
        return float(m.group(1)), None
    return None, None


def parse_blood_pressure(entry: dict) -> dict[str, tuple]:
    """Split a combined blood pressure entry into systolic and diastolic ranges."""
    text = (entry.get("healthy_value") or "").lower()
    sys_rng, dia_rng = (None, None), (None, None)

    # "110-116 / 70-73"
    m = re.search(NUMBER + DASH + NUMBER + r"\s*/\s*" + NUMBER + DASH + NUMBER, text)
    if m:
        sys_rng = (float(m.group(1)), float(m.group(2)))
        dia_rng = (float(m.group(3)), float(m.group(4)))
    else:
        # "91-119/61-79" without spaces is covered above; "below 120/80", "< 130/85"
        m = re.search(NUMBER + r"\s*/\s*" + NUMBER, text)
        if m:
            sys_rng = (None, float(m.group(1)))
            dia_rng = (None, float(m.group(2)))
        else:
            # "110-116 mmHg" -> systolic only
            lo, hi = parse_range_text(text)
            if lo is not None or hi is not None:
                sys_rng = (lo, hi)

    # The min/max fields often hold a systolic/diastolic pair (min 120, max 80).
    if sys_rng == (None, None):
        lo, hi = entry.get("min"), entry.get("max")
        if lo is not None and hi is not None and lo > hi:
            sys_rng, dia_rng = (None, lo), (None, hi)
        else:
            sys_rng = (lo, hi)
    return {"systolic_bp": sys_rng, "diastolic_bp": dia_rng}


def scraped_range(entry: dict) -> tuple[float | None, float | None]:
    """min/max from the JSON fields, with gaps filled from the healthy_value text."""
    lo, hi = entry.get("min"), entry.get("max")
    t_lo, t_hi = parse_range_text(entry.get("healthy_value", ""))
    if lo is None:
        lo = t_lo
    if hi is None:
        hi = t_hi
    return lo, hi


# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------
def to_canonical_unit(key: str, entry: dict, value: float | None, issues: list) -> float | None:
    """Convert one scraped value into the canonical unit of `key`."""
    if value is None:
        return None
    unit_text = f"{entry.get('unit', '')} {entry.get('healthy_value', '')}".lower()

    if key == "body_temperature" and ("°f" in unit_text or "fahrenheit" in unit_text or value > 50):
        issues.append(f"converted {value} °F to °C")
        return round((value - 32) * 5 / 9, 1)

    if key == "waist_circumference" and ("inch" in unit_text or value < 50):
        issues.append(f"converted {value} inches to cm")
        return round(value * 2.54, 1)

    if key in ("total_cholesterol", "ldl_cholesterol", "hdl_cholesterol") and (
        "mmol" in unit_text or value < 20
    ):
        issues.append(f"converted {value} mmol/L to mg/dL")
        return round(value * 38.67)

    if key == "fasting_glucose" and ("mmol" in unit_text or value < 20):
        issues.append(f"converted {value} mmol/L to mg/dL")
        return round(value * 18)

    if key == "hemoglobin" and ("g/l" in unit_text.replace("g/dl", "") or value > 30):
        issues.append(f"converted {value} g/L to g/dL")
        return round(value / 10, 1)

    return value


def wrong_metric(key: str, entry: dict) -> str | None:
    """Detect values that measure something else entirely."""
    unit = (entry.get("unit") or "").lower()
    if key == "bmi" and ("pound" in unit or re.fullmatch(r"\s*kg\s*", unit)):
        return f"value is a body weight ({entry.get('unit')}), not a BMI"
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def check_bound(value, ref_value, side: str, param: dict, issues: list):
    """Return (final_value, action) for one bound. action: kept / filled / corrected."""
    low_limit, high_limit = param["hard_limits"]
    unit = param["unit"]

    if ref_value is None:
        if value is not None:
            issues.append(f"{side} {value} dropped: guideline sets no {side} limit")
            return None, "corrected"
        return None, "kept"
    if value is None:
        issues.append(f"{side} missing, filled with {ref_value} {unit}")
        return ref_value, "filled"
    if not (low_limit <= value <= high_limit):
        issues.append(
            f"{side} {value} {unit} is outside plausible human limits "
            f"({low_limit}-{high_limit}), replaced with {ref_value}"
        )
        return ref_value, "corrected"
    if abs(value - ref_value) > param["tolerance"]:
        issues.append(
            f"{side} {value} {unit} differs from reference {ref_value} by more than "
            f"{param['tolerance']}, replaced"
        )
        return ref_value, "corrected"
    return value, "kept"


def tidy(value):
    """120.0 -> 120, so ranges read naturally."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


def clean_parameter(key: str, rng: tuple, entry: dict | None, age_low: int, age_high: int,
                    sex: str, issues: list) -> dict:
    param = PARAMETERS[key]
    ref = get_reference(key, age_low, age_high, sex)

    if entry is None:
        status, lo, hi = "added", ref["min"], ref["max"]
        issues.append("not found by the scraper, added from reference")
    else:
        bad = wrong_metric(key, entry)
        if bad:
            issues.append(f"{bad}, replaced with reference")
            rng = (None, None)
            forced = True
        else:
            forced = False
        lo = to_canonical_unit(key, entry, rng[0], issues)
        hi = to_canonical_unit(key, entry, rng[1], issues)

        if lo is not None and hi is not None and lo > hi:
            issues.append(f"min {lo} > max {hi}, replaced with reference")
            lo, hi, forced = None, None, True

        lo, a_lo = check_bound(lo, ref["min"], "min", param, issues)
        hi, a_hi = check_bound(hi, ref["max"], "max", param, issues)
        actions = {a_lo, a_hi}
        if forced or "corrected" in actions:
            status = "corrected"
        elif "filled" in actions:
            status = "filled"
        else:
            status = "kept"

    lo, hi = tidy(lo), tidy(hi)

    # Every value now either agrees with the reference within tolerance
    # ("verified") or is the reference itself.
    confidence = "reference" if status != "kept" and (lo, hi) == (ref["min"], ref["max"]) else "verified"

    out = {
        "category": param["category"],
        "unit": param["unit"],
        "min": lo,
        "max": hi,
        "healthy_value": format_range(lo, hi, param["unit"]),
        "confidence": confidence,
        "status": status,
        "scraped_confidence": (entry or {}).get("confidence"),
        "reference": {"min": ref["min"], "max": ref["max"], "source": param["source"]},
    }
    if status == "kept" and entry:
        out["notes"] = entry.get("notes", "")
    if issues:
        out["issues"] = issues
    if entry is not None and status != "kept":
        out["original"] = {k: entry.get(k) for k in ("unit", "min", "max", "healthy_value", "confidence")}
    if entry and entry.get("sources"):
        out["sources"] = entry["sources"]
    return out


def clean_file(data: dict) -> dict:
    age_low, age_high, sex = data["age_lower"], data["age_upper"], data["sex"]

    # 1. Group scraped entries by canonical key. Duplicates (e.g. "Cholesterol"
    #    and "Total Cholesterol") keep the higher-confidence one.
    rank = {"high": 3, "medium": 2, "low": 1}
    found: dict[str, tuple[str, dict, tuple]] = {}
    unmapped = {}

    def add(key, name, entry, rng):
        if key in found and rank.get(found[key][1].get("confidence"), 0) >= rank.get(entry.get("confidence"), 0):
            return
        found[key] = (name, entry, rng)

    for name, entry in data.get("parameters", {}).items():
        key = resolve_parameter(name)
        if key == BLOOD_PRESSURE:
            for bp_key, rng in parse_blood_pressure(entry).items():
                add(bp_key, name, entry, rng)
        elif key:
            add(key, name, entry, scraped_range(entry))
        else:
            unmapped[name] = entry

    # 2-4. Validate every reference parameter.
    parameters = {}
    for key, param in PARAMETERS.items():
        issues = []
        if key in found:
            src_name, entry, rng = found[key]
            if src_name != param["display_name"]:
                issues.append(f"renamed from '{src_name}'")
        else:
            entry, rng = None, (None, None)
        parameters[param["display_name"]] = clean_parameter(key, rng, entry, age_low, age_high, sex, issues)

    summary = {s: 0 for s in ("kept", "filled", "corrected", "added")}
    for p in parameters.values():
        summary[p["status"]] += 1
    summary["low_confidence_in"] = sum(
        1 for e in data.get("parameters", {}).values() if e.get("confidence") == "low"
    )

    out = {
        "age_lower": age_low,
        "age_upper": age_high,
        "sex": sex,
        "generated_at": data.get("generated_at"),
        "cleaned_at": datetime.now().isoformat(timespec="seconds"),
        "cleaning_summary": summary,
        "parameters": parameters,
    }
    if unmapped:
        out["unmapped_parameters"] = unmapped
    return out


# ---------------------------------------------------------------------------
# Dengue
# ---------------------------------------------------------------------------
# Same idea as the healthy cleaner, against DENGUE_PARAMETERS:
#   numeric : min/max checked with check_bound (hard limits, tolerance, gaps);
#             severe_value always comes from the WHO reference.
#   symptom : prevalence_percent kept if inside the reference prevalence range,
#             moved to the nearest edge if outside, reference default if missing.
# Core parameters (from the Dengue app) missing from a file are added.
# A file with no dengue-specific parameter at all (e.g. a healthy dictionary
# saved under a dengue name) is skipped instead of being filled with reference.
class NotDengueData(Exception):
    pass


def dengue_numeric_value(key: str, entry: dict, value, issues: list):
    """Convert one scraped dengue value into the canonical unit."""
    if value is None:
        return None
    text = f"{entry.get('unit', '')} {entry.get('dengue_value', '')}".lower()
    if key == "body_temperature" and ("°f" in text or "fahrenheit" in text or value > 50):
        issues.append(f"converted {value} °F to °C")
        return round((value - 32) * 5 / 9, 1)
    if key in ("platelet_count", "wbc_count") and 0 < value < 1000:
        issues.append(f"converted {value} x10^3/µL to {int(value * 1000)} cells/µL")
        return int(value * 1000)
    if key == "hematocrit" and 0 < value <= 1:
        issues.append(f"converted fraction {value} to {value * 100:.0f} %")
        return round(value * 100, 1)
    return value


def dengue_scraped_range(key: str, entry: dict) -> tuple:
    if key == "systolic_bp":
        return parse_blood_pressure(entry)["systolic_bp"]
    lo, hi = entry.get("min"), entry.get("max")
    t_lo, t_hi = parse_range_text(entry.get("dengue_value", ""))
    return (t_lo if lo is None else lo), (t_hi if hi is None else hi)


def clean_dengue_numeric(key: str, entry: dict | None, ref: dict, issues: list) -> dict:
    param = DENGUE_PARAMETERS[key]
    if entry is None:
        status, lo, hi = "added", ref["min"], ref["max"]
        issues.append("not found by the scraper, added from reference")
    else:
        rng = dengue_scraped_range(key, entry)
        lo = dengue_numeric_value(key, entry, rng[0], issues)
        hi = dengue_numeric_value(key, entry, rng[1], issues)
        forced = False
        if lo is not None and hi is not None and lo > hi:
            issues.append(f"min {lo} > max {hi}, replaced with reference")
            lo, hi, forced = None, None, True
        lo, a_lo = check_bound(lo, ref["min"], "min", param, issues)
        hi, a_hi = check_bound(hi, ref["max"], "max", param, issues)
        actions = {a_lo, a_hi}
        status = ("corrected" if forced or "corrected" in actions
                  else "filled" if "filled" in actions else "kept")

    lo, hi = tidy(lo), tidy(hi)
    out = {
        "kind": "numeric",
        "category": param["category"],
        "unit": param["unit"],
        "min": lo,
        "max": hi,
        "dengue_value": format_range(lo, hi, param["unit"]),
        "severe_value": param["severe_value"],
        "confidence": "reference" if status != "kept" and (lo, hi) == (ref["min"], ref["max"]) else "verified",
        "status": status,
        "reference": {"min": ref["min"], "max": ref["max"], "source": param["source"]},
    }
    if entry and entry.get("severe_value") and entry["severe_value"] != param["severe_value"]:
        out["scraped_severe_value"] = entry["severe_value"]
    return out


def clean_dengue_symptom(key: str, entry: dict | None, ref: dict, issues: list) -> dict:
    param = DENGUE_PARAMETERS[key]
    low, high = ref["prevalence"]
    if entry is None:
        status, value = "added", ref["default"]
        issues.append("not found by the scraper, added from reference")
    else:
        value = entry.get("prevalence_percent")
        if isinstance(value, (int, float)) and 0 < value <= 1:
            issues.append(f"converted fraction {value} to {value * 100:g} %")
            value = round(value * 100, 1)
        if not isinstance(value, (int, float)):
            issues.append(f"prevalence missing, filled with {ref['default']} %")
            status, value = "filled", ref["default"]
        elif not (low <= value <= high):
            nearest = low if value < low else high
            issues.append(f"prevalence {value} % outside expected {low}-{high} %, set to {nearest} %")
            status, value = "corrected", nearest
        else:
            status = "kept"

    value = tidy(value)
    return {
        "kind": "symptom",
        "category": param["category"],
        "prevalence_percent": value,
        "dengue_value": f"present in about {value} % of dengue patients",
        "confidence": "verified" if status == "kept" else "reference",
        "status": status,
        "reference": {"prevalence_range": [low, high], "default": ref["default"], "source": param["source"]},
    }


def clean_dengue_file(data: dict) -> dict:
    age_low, age_high, sex = data["age_lower"], data["age_upper"], data["sex"]

    rank = {"high": 3, "medium": 2, "low": 1}
    found: dict[str, tuple[str, dict]] = {}
    unmapped = {}
    for name, entry in data.get("parameters", {}).items():
        key = resolve_dengue_parameter(name)
        if not key:
            unmapped[name] = entry
        elif key not in found or rank.get(entry.get("confidence"), 0) > rank.get(found[key][1].get("confidence"), 0):
            found[key] = (name, entry)

    if not DENGUE_SPECIFIC & found.keys():
        raise NotDengueData(
            "no dengue parameters (platelets, WBC, hematocrit, symptoms) - "
            "looks like a healthy dictionary; regenerate with category_dengue.py --overwrite"
        )

    parameters = {}
    for key, param in DENGUE_PARAMETERS.items():
        if key not in found and not param["core"]:
            continue
        issues = []
        src_name, entry = found.get(key, (None, None))
        if src_name and src_name != param["display_name"]:
            issues.append(f"renamed from '{src_name}'")
        ref = get_dengue_reference(key, age_low, age_high, sex)
        clean = clean_dengue_symptom if param["kind"] == "symptom" else clean_dengue_numeric
        out = clean(key, entry, ref, issues)
        out["scraped_confidence"] = (entry or {}).get("confidence")
        if out["status"] == "kept" and entry:
            out["notes"] = entry.get("notes", "")
        if issues:
            out["issues"] = issues
        if entry is not None and out["status"] != "kept":
            out["original"] = {k: entry.get(k) for k in
                               ("unit", "min", "max", "prevalence_percent", "dengue_value", "confidence")
                               if k in entry}
        if entry and entry.get("sources"):
            out["sources"] = entry["sources"]
        parameters[param["display_name"]] = out

    summary = {s: 0 for s in ("kept", "filled", "corrected", "added")}
    for p in parameters.values():
        summary[p["status"]] += 1
    summary["low_confidence_in"] = sum(
        1 for e in data.get("parameters", {}).values() if e.get("confidence") == "low"
    )

    out = {
        "age_lower": age_low,
        "age_upper": age_high,
        "sex": sex,
        "condition": "dengue",
        "generated_at": data.get("generated_at"),
        "cleaned_at": datetime.now().isoformat(timespec="seconds"),
        "cleaning_summary": summary,
        "parameters": parameters,
    }
    if unmapped:
        out["unmapped_parameters"] = unmapped
    return out


CLEANERS = {"healthy": clean_file, "dengue": clean_dengue_file}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def age_sort_key(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("age_lower", 0), data.get("sex", "")


def read_batch_size(arg: str | None, pending: int) -> int:
    text = arg if arg is not None else input(f"How many files to process in this run? (1-{pending} or 'all'): ")
    text = text.strip().lower()
    if text in ("all", "a", ""):
        return pending
    if not text.isdigit() or int(text) < 1:
        sys.exit(f"Invalid number of files: '{text}'")
    return min(int(text), pending)


def read_dataset(arg: str | None) -> str:
    if arg:
        return arg
    while True:
        choice = input("Which data to clean? (1 = healthy, 2 = dengue): ").strip()
        if choice in ("1", "2"):
            return "healthy" if choice == "1" else "dengue"


def main():
    parser = argparse.ArgumentParser(description="Check and fix scraped healthy / dengue dictionaries.")
    parser.add_argument("--dataset", choices=DATASETS, help="which dictionaries to clean")
    parser.add_argument("--batch", help="number of files to process in this run, or 'all'")
    parser.add_argument("--overwrite", action="store_true", help="re-clean files already in the output folder")
    parser.add_argument("--input", type=Path, help="folder with the scraped JSON files (overrides the default)")
    args = parser.parse_args()

    dataset = read_dataset(args.dataset)
    config = DATASETS[dataset]
    input_dir, output_dir, pattern = args.input or config["input"], config["output"], config["pattern"]
    cleaner = CLEANERS[dataset]
    print(f"Dataset: {dataset}  ({input_dir} -> {output_dir})")

    files = sorted(input_dir.glob(pattern), key=age_sort_key)
    if not files:
        sys.exit(f"No files matching {pattern} in {input_dir}")
    pending = [f for f in files if args.overwrite or not (output_dir / f.name).exists()]
    print(f"Found {len(files)} files, {len(files) - len(pending)} already cleaned, {len(pending)} to do.")
    if not pending:
        print("Nothing to do (use --overwrite to re-clean).")
        return

    batch = pending[:read_batch_size(args.batch, len(pending))]
    output_dir.mkdir(parents=True, exist_ok=True)

    totals = {"kept": 0, "filled": 0, "corrected": 0, "added": 0}
    skipped = []
    for i, path in enumerate(batch, 1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cleaned = cleaner(data)
        except NotDengueData as exc:
            print(f"[{i}/{len(batch)}] {path.name}: SKIPPED ({exc})")
            skipped.append(path.name)
            continue
        except Exception as exc:
            print(f"[{i}/{len(batch)}] {path.name}: FAILED ({exc})")
            continue
        (output_dir / path.name).write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
        s = cleaned["cleaning_summary"]
        for k in totals:
            totals[k] += s[k]
        print(f"[{i}/{len(batch)}] {path.name}: kept {s['kept']}, filled {s['filled']}, "
              f"corrected {s['corrected']}, added {s['added']}")

    print(f"\nDone: {len(batch) - len(skipped)} files -> {output_dir}")
    print(f"Totals: kept {totals['kept']}, filled {totals['filled']}, "
          f"corrected {totals['corrected']}, added {totals['added']}")
    if skipped:
        print(f"Skipped {len(skipped)} files that contain no dengue data.")
    if len(pending) > len(batch):
        print(f"{len(pending) - len(batch)} files left; run again to continue.")


if __name__ == "__main__":
    main()
