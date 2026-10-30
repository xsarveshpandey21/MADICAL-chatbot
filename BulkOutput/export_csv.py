"""
export_csv.py

Exports patient records from BulkOutput/output JSON files into a structured CSV file
suitable for clinical analysis, Excel, or tabular data science workflows.

Usage:
    python BulkOutput/export_csv.py
    python BulkOutput/export_csv.py --input BulkOutput/output/261026_vital_15_25.json
    python BulkOutput/export_csv.py --output custom_patients.csv
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RECORDS_DIR = BASE_DIR / "output"
DEFAULT_EXPORT_PATH = BASE_DIR / "bulk_patients_export.csv"

# Canonical list of vitals and their standard units for column headers
VITAL_COLUMNS = [
    ("Heart Rate", "bpm"),
    ("Systolic Blood Pressure", "mmHg"),
    ("Diastolic Blood Pressure", "mmHg"),
    ("Pulse Pressure", "mmHg"),
    ("Respiratory Rate", "breaths/min"),
    ("Body Temperature", "°C"),
    ("SpO2", "%"),
    ("Fasting Blood Glucose", "mg/dL"),
    ("HbA1c", "%"),
    ("Total Cholesterol", "mg/dL"),
    ("LDL Cholesterol", "mg/dL"),
    ("HDL Cholesterol", "mg/dL"),
    ("Platelet Count", "cells/µL"),
    ("WBC Count", "cells/µL"),
    ("Hemoglobin", "g/dL"),
    ("Hematocrit", "%"),
    ("BMI", "kg/m²"),
    ("Waist Circumference", "cm"),
    ("Sleep Hours", "hours/night"),
    ("Physical Activity Minutes per Week", "min/week"),
]

SYMPTOM_NAMES = [
    "Headache",
    "Lethargy or Restlessness",
    "Mucosal Bleeding",
    "Muscle and Joint Pain",
    "Nausea or Vomiting",
    "Persistent Vomiting",
    "Retro-orbital Pain",
    "Severe Abdominal Pain",
    "Skin Rash",
]

WARNING_SIGN_SYMPTOMS = {
    "Severe Abdominal Pain",
    "Persistent Vomiting",
    "Mucosal Bleeding",
    "Lethargy or Restlessness",
}
LOW_PLATELETS_THRESHOLD = 100000
NARROW_PULSE_PRESSURE_THRESHOLD = 20


def get_all_records() -> list[dict]:
    """Load all records from BulkOutput/output/*.json files."""
    files = sorted(RECORDS_DIR.glob("*.json"))
    records = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! Skipping {path.name}: {exc}")
            continue
        for r in data.get("records", []):
            records.append({**r, "source_file": path.name})
    return records


def calculate_pulse_pressure(record: dict) -> float | None:
    vitals = record.get("vitals", {})
    sys_val = (vitals.get("Systolic Blood Pressure") or {}).get("value")
    dia_val = (vitals.get("Diastolic Blood Pressure") or {}).get("value")
    if sys_val is not None and dia_val is not None:
        return round(float(sys_val) - float(dia_val), 1)
    return None


def get_dengue_warning_signs(record: dict) -> list[str]:
    signs = []
    v = record.get("vitals", {})
    pp = calculate_pulse_pressure(record)
    if pp is not None and pp <= NARROW_PULSE_PRESSURE_THRESHOLD:
        signs.append(f"narrow pulse pressure ({pp} mmHg)")
    platelets = v.get("Platelet Count")
    if platelets and platelets.get("value") is not None and platelets["value"] < LOW_PLATELETS_THRESHOLD:
        signs.append(f"low platelet count ({platelets['value']} cells/µL)")
    for s, on in record.get("symptoms", {}).items():
        if on and s in WARNING_SIGN_SYMPTOMS:
            signs.append(s)
    return signs


def flatten_record(r: dict) -> dict:
    row = {
        "ID": r.get("id"),
        "Name": r.get("name", "Unknown"),
        "Sex": r.get("sex", ""),
        "Age": r.get("age", ""),
        "Age Group": r.get("age_group", ""),
        "Condition": r.get("condition", ""),
    }

    vitals = r.get("vitals", {})
    pp = calculate_pulse_pressure(r)

    for vital_name, unit in VITAL_COLUMNS:
        col_header = f"{vital_name} ({unit})"
        if vital_name == "Pulse Pressure":
            row[col_header] = pp if pp is not None else ""
        else:
            vital_obj = vitals.get(vital_name)
            val = vital_obj.get("value") if isinstance(vital_obj, dict) else None
            row[col_header] = val if val is not None else ""

    symptoms = r.get("symptoms", {})
    for sym in SYMPTOM_NAMES:
        row[f"Symptom: {sym}"] = "Yes" if symptoms.get(sym) else "No"

    signs = get_dengue_warning_signs(r)
    row["Has Dengue Warning Signs"] = "Yes" if signs else "No"
    row["Dengue Warning Signs Found"] = "; ".join(signs) if signs else "None"
    row["Source File"] = r.get("source_file", "")

    return row


def export_records_to_csv(records: list[dict] | None = None,
                          output_path: Path | str | None = None) -> Path:
    """
    Exports the given list of records (or all records from output/) to a CSV file.
    Uses 'utf-8-sig' for native UTF-8 opening in Microsoft Excel on Windows.
    """
    if records is None:
        records = get_all_records()

    if output_path is None:
        output_path = DEFAULT_EXPORT_PATH
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [flatten_record(r) for r in records]
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["ID", "Name", "Sex", "Age", "Age Group", "Condition", "Source File"]

    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_file_to_csv(json_path: Path | str,
                       output_path: Path | str | None = None) -> Path:
    """
    Exports a single JSON file of patient records to a corresponding CSV file.
    If output_path is not specified, uses the same path with .csv extension.
    """
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.with_suffix(".csv")
    else:
        output_path = Path(output_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    records = [{**r, "source_file": json_path.name} for r in data.get("records", [])]
    return export_records_to_csv(records, output_path=output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export patient records to CSV")
    parser.add_argument("-i", "--input", help="Specific JSON file to export (default: all files in output/)", default=None)
    parser.add_argument("-o", "--output", help="Output CSV path", default=None)
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")
        export_file_to_csv(in_path, output_path=out_path)
        print(f"Successfully exported {in_path.name} to CSV:\n  {out_path.resolve()}")
    else:
        out_path = Path(args.output) if args.output else DEFAULT_EXPORT_PATH
        out_file = export_records_to_csv(output_path=out_path)
        count = len(get_all_records())
        print(f"Successfully exported {count} patient records to CSV:\n  {out_file.resolve()}")


if __name__ == "__main__":
    main()
