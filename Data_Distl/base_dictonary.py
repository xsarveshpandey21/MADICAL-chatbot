"""
base_dictonary.py

Reference values used by dataCleaner.py to check and fix the scraped
dictionaries:
    PARAMETERS        - normal human levels   (Category_Dict/output)
    DENGUE_PARAMETERS - values seen in dengue (Category_Dict/output_dengue),
                        see the "Dengue reference" section further down.

Every parameter has
    display_name : canonical key written to the cleaned JSON
    category     : vitals / body / blood / lifestyle
    unit         : canonical unit (all scraped values are converted to it)
    aliases      : names the scraper has used for this parameter
    hard_limits  : (low, high) - a healthy reference value outside this is not
                   humanly possible (e.g. 32 C body temperature) and is rejected
    tolerance    : how far (in `unit`) a scraped bound may differ from the
                   reference bound and still be accepted
    ranges       : age/sex specific healthy ranges. "ages" is (from, to) with
                   from <= age < to. "sex" is "male", "female" or None (both).
                   min / max of None means the guideline sets no limit on that
                   side (e.g. "total cholesterol below 200").
    source       : guideline the ranges come from

Values are resting / fasting reference ranges for healthy people from widely
used clinical guidelines (AHA, WHO, CDC, ADA, NHLBI, IDF, AASM, PALS, Mayo
Clinic). Child ranges are typical normal ranges for the age band.
"""

import re

PARAMETERS = {
    # ------------------------------------------------------------------ vitals
    "heart_rate": {
        "display_name": "Heart Rate",
        "category": "vitals",
        "unit": "bpm",
        "aliases": ["heart rate", "resting heart rate", "pulse", "pulse rate"],
        "hard_limits": (40, 200),
        "tolerance": 10,
        "ranges": [
            {"ages": (0, 3), "sex": None, "min": 98, "max": 140},
            {"ages": (3, 6), "sex": None, "min": 80, "max": 120},
            {"ages": (6, 13), "sex": None, "min": 70, "max": 110},
            {"ages": (13, 200), "sex": None, "min": 60, "max": 100},
        ],
        "source": "AHA / PALS resting heart rate by age",
    },
    "systolic_bp": {
        "display_name": "Systolic Blood Pressure",
        "category": "vitals",
        "unit": "mmHg",
        "aliases": ["systolic blood pressure", "systolic bp", "systolic"],
        "hard_limits": (70, 140),
        "tolerance": 8,
        "ranges": [
            {"ages": (0, 3), "sex": None, "min": 86, "max": 106},
            {"ages": (3, 6), "sex": None, "min": 89, "max": 112},
            {"ages": (6, 13), "sex": None, "min": 97, "max": 115},
            {"ages": (13, 18), "sex": None, "min": 102, "max": 120},
            {"ages": (18, 200), "sex": None, "min": 90, "max": 119},
        ],
        "source": "AAP 2017 pediatric BP guideline; AHA/ACC 2017 (normal < 120/80)",
    },
    "diastolic_bp": {
        "display_name": "Diastolic Blood Pressure",
        "category": "vitals",
        "unit": "mmHg",
        "aliases": ["diastolic blood pressure", "diastolic bp", "diastolic"],
        "hard_limits": (35, 90),
        "tolerance": 6,
        "ranges": [
            {"ages": (0, 3), "sex": None, "min": 42, "max": 63},
            {"ages": (3, 6), "sex": None, "min": 46, "max": 72},
            {"ages": (6, 13), "sex": None, "min": 57, "max": 76},
            {"ages": (13, 18), "sex": None, "min": 61, "max": 80},
            {"ages": (18, 200), "sex": None, "min": 60, "max": 79},
        ],
        "source": "AAP 2017 pediatric BP guideline; AHA/ACC 2017 (normal < 120/80)",
    },
    "respiratory_rate": {
        "display_name": "Respiratory Rate",
        "category": "vitals",
        "unit": "breaths/min",
        "aliases": ["respiratory rate", "breathing rate", "respiration rate"],
        "hard_limits": (8, 60),
        "tolerance": 4,
        "ranges": [
            {"ages": (0, 3), "sex": None, "min": 24, "max": 40},
            {"ages": (3, 6), "sex": None, "min": 22, "max": 34},
            {"ages": (6, 13), "sex": None, "min": 18, "max": 30},
            {"ages": (13, 200), "sex": None, "min": 12, "max": 20},
        ],
        "source": "PALS / clinical vital sign references",
    },
    "body_temperature": {
        "display_name": "Body Temperature",
        "category": "vitals",
        "unit": "°C",
        "aliases": ["body temperature", "temperature", "core temperature", "oral temperature"],
        "hard_limits": (35.5, 38.0),
        "tolerance": 0.4,
        "ranges": [
            {"ages": (0, 13), "sex": None, "min": 36.5, "max": 37.5},
            {"ages": (13, 65), "sex": None, "min": 36.1, "max": 37.2},
            {"ages": (65, 200), "sex": None, "min": 36.0, "max": 37.2},
        ],
        "source": "Oral temperature references; older adults run slightly lower",
    },
    "spo2": {
        "display_name": "SpO2",
        "category": "vitals",
        "unit": "%",
        "aliases": ["spo2", "oxygen saturation", "blood oxygen", "blood oxygen saturation",
                    "peripheral oxygen saturation", "o2 saturation"],
        "hard_limits": (90, 100),
        "tolerance": 2,
        "ranges": [
            {"ages": (0, 200), "sex": None, "min": 95, "max": 100},
        ],
        "source": "WHO pulse oximetry manual",
    },
    # -------------------------------------------------------------------- body
    "bmi": {
        "display_name": "BMI",
        "category": "body",
        "unit": "kg/m^2",
        "aliases": ["bmi", "body mass index", "body mass index bmi"],
        "hard_limits": (12, 32),
        "tolerance": 1.5,
        "ranges": [
            # Children: approx. CDC 5th-85th percentile BMI-for-age at the band's age.
            {"ages": (0, 5), "sex": None, "min": 14.0, "max": 17.5},
            {"ages": (5, 10), "sex": None, "min": 13.8, "max": 19.0},
            {"ages": (10, 15), "sex": None, "min": 14.8, "max": 22.0},
            {"ages": (15, 20), "sex": None, "min": 17.0, "max": 25.0},
            {"ages": (20, 65), "sex": None, "min": 18.5, "max": 24.9},
            {"ages": (65, 200), "sex": None, "min": 22.0, "max": 27.0},
        ],
        "source": "WHO adult BMI; CDC BMI-for-age (5th-85th pct); geriatric optimum 22-27",
    },
    "waist_circumference": {
        "display_name": "Waist Circumference",
        "category": "body",
        "unit": "cm",
        "aliases": ["waist circumference", "waist size", "waist", "waist measurement"],
        "hard_limits": (35, 130),
        "tolerance": 6,
        "ranges": [
            # Children: approx. 90th percentile; waist-to-height < 0.5 is the better check.
            {"ages": (0, 5), "sex": None, "min": None, "max": 55},
            {"ages": (5, 10), "sex": None, "min": None, "max": 65},
            {"ages": (10, 15), "sex": "male", "min": None, "max": 78},
            {"ages": (10, 15), "sex": "female", "min": None, "max": 75},
            {"ages": (15, 18), "sex": "male", "min": None, "max": 88},
            {"ages": (15, 18), "sex": "female", "min": None, "max": 80},
            {"ages": (18, 200), "sex": "male", "min": None, "max": 94},
            {"ages": (18, 200), "sex": "female", "min": None, "max": 80},
        ],
        "source": "IDF (Europid) < 94 cm men / < 80 cm women; pediatric 90th percentile",
    },
    # ------------------------------------------------------------------- blood
    "fasting_glucose": {
        "display_name": "Fasting Blood Glucose",
        "category": "blood",
        "unit": "mg/dL",
        "aliases": ["glucose", "fasting glucose", "blood glucose", "fasting blood glucose",
                    "fasting blood sugar", "blood sugar", "fasting plasma glucose"],
        "hard_limits": (60, 126),
        "tolerance": 10,
        "ranges": [
            {"ages": (0, 200), "sex": None, "min": 70, "max": 99},
        ],
        "source": "ADA Standards of Care (normal fasting < 100 mg/dL)",
    },
    "hba1c": {
        "display_name": "HbA1c",
        "category": "blood",
        "unit": "%",
        "aliases": ["hba1c", "a1c", "hemoglobin a1c", "hemoglobin a1c hba1c", "hemoglobin hba1c",
                    "glycated hemoglobin", "glycosylated hemoglobin"],
        "hard_limits": (3.5, 6.0),
        "tolerance": 0.3,
        "ranges": [
            {"ages": (0, 200), "sex": None, "min": 4.0, "max": 5.6},
        ],
        "source": "ADA (normal < 5.7 %, 5.7-6.4 % is prediabetes)",
    },
    "total_cholesterol": {
        "display_name": "Total Cholesterol",
        "category": "blood",
        "unit": "mg/dL",
        "aliases": ["total cholesterol", "cholesterol", "serum cholesterol"],
        "hard_limits": (80, 240),
        "tolerance": 10,
        "ranges": [
            {"ages": (0, 20), "sex": None, "min": None, "max": 170},
            {"ages": (20, 200), "sex": None, "min": None, "max": 200},
        ],
        "source": "NHLBI pediatric lipid guideline; NCEP ATP III (desirable < 200 mg/dL)",
    },
    "ldl_cholesterol": {
        "display_name": "LDL Cholesterol",
        "category": "blood",
        "unit": "mg/dL",
        "aliases": ["ldl", "ldl cholesterol", "low density lipoprotein",
                    "low density lipoprotein ldl cholesterol"],
        "hard_limits": (30, 160),
        "tolerance": 15,
        "ranges": [
            {"ages": (0, 20), "sex": None, "min": None, "max": 110},
            {"ages": (20, 200), "sex": None, "min": None, "max": 100},
        ],
        "source": "NHLBI pediatric lipid guideline; NCEP ATP III (optimal < 100 mg/dL)",
    },
    "hdl_cholesterol": {
        "display_name": "HDL Cholesterol",
        "category": "blood",
        "unit": "mg/dL",
        "aliases": ["hdl", "hdl cholesterol", "high density lipoprotein",
                    "high density lipoprotein hdl cholesterol"],
        "hard_limits": (30, 100),
        "tolerance": 8,
        "ranges": [
            {"ages": (0, 20), "sex": None, "min": 45, "max": None},
            {"ages": (20, 200), "sex": "male", "min": 40, "max": None},
            {"ages": (20, 200), "sex": "female", "min": 50, "max": None},
        ],
        "source": "NHLBI pediatric lipid guideline; NCEP ATP III",
    },
    "hemoglobin": {
        "display_name": "Hemoglobin",
        "category": "blood",
        "unit": "g/dL",
        "aliases": ["hemoglobin", "haemoglobin", "hemoglobin hb", "hb", "hgb"],
        "hard_limits": (9, 20),
        "tolerance": 1.0,
        "ranges": [
            {"ages": (0, 5), "sex": None, "min": 11.0, "max": 14.0},
            {"ages": (5, 12), "sex": None, "min": 11.5, "max": 15.5},
            {"ages": (12, 18), "sex": "male", "min": 13.0, "max": 16.0},
            {"ages": (12, 18), "sex": "female", "min": 12.0, "max": 16.0},
            {"ages": (18, 200), "sex": "male", "min": 13.2, "max": 16.6},
            {"ages": (18, 200), "sex": "female", "min": 11.6, "max": 15.0},
        ],
        "source": "WHO anaemia thresholds; Mayo Clinic adult reference ranges",
    },
    # --------------------------------------------------------------- lifestyle
    "sleep_hours": {
        "display_name": "Sleep Hours",
        "category": "lifestyle",
        "unit": "hours/night",
        "aliases": ["sleep hours", "sleep duration", "sleep", "hours of sleep"],
        "hard_limits": (5, 16),
        "tolerance": 1,
        "ranges": [
            {"ages": (0, 3), "sex": None, "min": 11, "max": 14},
            {"ages": (3, 6), "sex": None, "min": 10, "max": 13},
            {"ages": (6, 13), "sex": None, "min": 9, "max": 12},
            {"ages": (13, 18), "sex": None, "min": 8, "max": 10},
            {"ages": (18, 65), "sex": None, "min": 7, "max": 9},
            {"ages": (65, 200), "sex": None, "min": 7, "max": 8},
        ],
        "source": "AASM / National Sleep Foundation recommendations",
    },
    "physical_activity": {
        "display_name": "Physical Activity Minutes per Week",
        "category": "lifestyle",
        "unit": "min/week",
        "aliases": ["physical activity minutes per week", "physical activity minutes",
                    "physical activity", "exercise minutes per week", "exercise"],
        "hard_limits": (0, 2000),
        "tolerance": 30,
        "ranges": [
            # 1-4 yrs: 180 min/day of any activity; 5-17: 60 min/day moderate-vigorous;
            # adults: at least 150 min/week moderate (more is fine, so no max).
            {"ages": (0, 5), "sex": None, "min": 1260, "max": None},
            {"ages": (5, 18), "sex": None, "min": 420, "max": None},
            {"ages": (18, 200), "sex": None, "min": 150, "max": None},
        ],
        "source": "WHO 2020 guidelines on physical activity",
    },
}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


_ALIASES = {_normalise(a): key for key, p in PARAMETERS.items() for a in p["aliases"]}

# Fallback keyword rules, checked in order (HbA1c before hemoglobin, LDL/HDL
# before generic cholesterol).
_KEYWORDS = [
    ("a1c", "hba1c"), ("glycated", "hba1c"),
    ("ldl", "ldl_cholesterol"), ("low density", "ldl_cholesterol"),
    ("hdl", "hdl_cholesterol"), ("high density", "hdl_cholesterol"),
    ("cholesterol", "total_cholesterol"),
    ("systolic", "systolic_bp"), ("diastolic", "diastolic_bp"),
    ("blood pressure", "blood_pressure"),
    ("glucose", "fasting_glucose"), ("sugar", "fasting_glucose"),
    ("hemoglobin", "hemoglobin"), ("haemoglobin", "hemoglobin"),
    ("heart rate", "heart_rate"), ("pulse", "heart_rate"),
    ("respirat", "respiratory_rate"), ("breathing", "respiratory_rate"),
    ("temperature", "body_temperature"),
    ("oxygen", "spo2"), ("spo2", "spo2"),
    ("body mass", "bmi"), ("bmi", "bmi"),
    ("waist", "waist_circumference"),
    ("sleep", "sleep_hours"),
    ("physical activity", "physical_activity"), ("exercise", "physical_activity"),
]

# Scraped "Blood Pressure" is one combined parameter; the cleaner splits it.
BLOOD_PRESSURE = "blood_pressure"


def resolve_parameter(name: str) -> str | None:
    """Map a scraped parameter name to a PARAMETERS key (or BLOOD_PRESSURE)."""
    norm = _normalise(name)
    if norm in ("blood pressure", "bp", "blood pressure bp"):
        return BLOOD_PRESSURE
    if norm in _ALIASES:
        return _ALIASES[norm]
    for word, key in _KEYWORDS:
        if word in norm:
            return key
    return None


def reference_age(age_low: int, age_high: int) -> float:
    """Age used to pick the reference band: the middle of the age group."""
    return (age_low + age_high) / 2


def _find_range(param: dict, age_low: int, age_high: int, sex: str) -> dict:
    age = reference_age(age_low, age_high)
    for r in param["ranges"]:
        lo, hi = r["ages"]
        if lo <= age < hi and r["sex"] in (None, sex):
            return r
    raise KeyError(f"No reference range for {param['display_name']}, age {age}, sex {sex}")


def get_reference(key: str, age_low: int, age_high: int, sex: str) -> dict:
    """Healthy range of `key` for the age group and sex."""
    r = _find_range(PARAMETERS[key], age_low, age_high, sex)
    return {"min": r["min"], "max": r["max"]}


def format_range(lo, hi, unit: str) -> str:
    """Human readable range; min and max are inclusive."""
    if lo is not None and hi is not None:
        return f"{lo}-{hi} {unit}"
    if hi is not None:
        return f"<= {hi} {unit}"
    if lo is not None:
        return f">= {lo} {unit}"
    return "no limit"


def base_dictionary(age_low: int, age_high: int, sex: str) -> dict:
    """The full reference dictionary for one age group and sex."""
    out = {}
    for key, p in PARAMETERS.items():
        ref = get_reference(key, age_low, age_high, sex)
        out[p["display_name"]] = {
            "category": p["category"],
            "unit": p["unit"],
            **ref,
            "healthy_value": format_range(ref["min"], ref["max"], p["unit"]),
            "source": p["source"],
        }
    return out


# ===========================================================================
# Dengue reference (used for Category_Dict/output_dengue)
# ===========================================================================
# What a person WITH DENGUE looks like, per age group and sex. Two kinds:
#   numeric : typical range in dengue patients (febrile / critical phase), plus
#             hard_limits, tolerance and the severe-dengue level, as above.
#   symptom : share of dengue patients who have it. Each range gives
#             "prevalence" (low, high) - a scraped value inside it is kept, one
#             outside is moved to the nearest edge - and "default", used when
#             the scraped value is missing.
# core = True parameters come from the Dengue app (BulkData/app.py) and are
# added when the scraper missed them; core = False ones are only checked.
# Symptom prevalences are approximate ranges from published case series.
DENGUE_PARAMETERS = {
    # ------------------------------------------------------- numeric (core)
    "body_temperature": {
        "display_name": "Body Temperature",
        "kind": "numeric", "category": "vitals", "unit": "°C", "core": True,
        "aliases": ["body temperature", "temperature", "fever"],
        "hard_limits": (36.0, 42.0),
        "tolerance": 0.5,
        "ranges": [{"ages": (0, 200), "sex": None, "min": 38.0, "max": 40.5}],
        "severe_value": "Defervescence (drop to <= 38 °C) on days 3-7 marks the critical phase",
        "source": "WHO Dengue guidelines 2009 / CDC clinical guidance",
    },
    "platelet_count": {
        "display_name": "Platelet Count",
        "kind": "numeric", "category": "blood", "unit": "cells/µL", "core": True,
        "aliases": ["platelet count", "platelets", "platelet", "thrombocyte count"],
        "hard_limits": (1000, 450000),
        "tolerance": 15000,
        "ranges": [{"ages": (0, 200), "sex": None, "min": 20000, "max": 100000}],
        "severe_value": "< 20,000 cells/µL, or a rapid drop together with rising hematocrit",
        "source": "WHO Dengue guidelines 2009 (thrombocytopenia <= 100,000/µL)",
    },
    "wbc_count": {
        "display_name": "WBC Count",
        "kind": "numeric", "category": "blood", "unit": "cells/µL", "core": True,
        "aliases": ["wbc count", "wbc", "white blood cell count", "white blood cells",
                    "leukocyte count", "total leukocyte count"],
        "hard_limits": (500, 15000),
        "tolerance": 800,
        "ranges": [{"ages": (0, 200), "sex": None, "min": 2000, "max": 5000}],
        "severe_value": "Progressive leukopenia (< 2,000 cells/µL) often precedes the critical phase",
        "source": "WHO Dengue guidelines 2009 (leukopenia <= 5,000/µL)",
    },
    "hematocrit": {
        "display_name": "Hematocrit",
        "kind": "numeric", "category": "blood", "unit": "%", "core": True,
        "aliases": ["hematocrit", "haematocrit", "hct", "packed cell volume", "pcv"],
        "hard_limits": (25, 65),
        "tolerance": 3,
        # Normal baseline + the >= 20 % rise (haemoconcentration) seen in dengue.
        "ranges": [
            {"ages": (0, 12), "sex": None, "min": 36, "max": 48},
            {"ages": (12, 18), "sex": "male", "min": 40, "max": 52},
            {"ages": (12, 18), "sex": "female", "min": 37, "max": 48},
            {"ages": (18, 200), "sex": "male", "min": 42, "max": 55},
            {"ages": (18, 200), "sex": "female", "min": 38, "max": 50},
        ],
        "severe_value": ">= 20 % rise from the patient's baseline (plasma leakage)",
        "source": "WHO Dengue guidelines 2009 (haemoconcentration >= 20 %)",
    },
    # --------------------------------------------------- numeric (optional)
    "heart_rate": {
        "display_name": "Heart Rate",
        "kind": "numeric", "category": "vitals", "unit": "bpm", "core": False,
        "aliases": ["heart rate", "pulse", "pulse rate"],
        "hard_limits": (50, 200),
        "tolerance": 10,
        "ranges": [
            {"ages": (0, 6), "sex": None, "min": 100, "max": 150},
            {"ages": (6, 13), "sex": None, "min": 90, "max": 140},
            {"ages": (13, 200), "sex": None, "min": 80, "max": 120},
        ],
        "severe_value": "Tachycardia with weak pulse and cold extremities = compensated shock",
        "source": "WHO Dengue guidelines 2009 (warning signs of shock)",
    },
    "systolic_bp": {
        "display_name": "Systolic Blood Pressure",
        "kind": "numeric", "category": "vitals", "unit": "mmHg", "core": False,
        "aliases": ["blood pressure", "systolic blood pressure", "systolic bp", "bp"],
        "hard_limits": (60, 150),
        "tolerance": 8,
        "ranges": [
            {"ages": (0, 13), "sex": None, "min": 85, "max": 110},
            {"ages": (13, 200), "sex": None, "min": 90, "max": 120},
        ],
        "severe_value": "Pulse pressure <= 20 mmHg or systolic < 90 mmHg (adults) = dengue shock",
        "source": "WHO Dengue guidelines 2009",
    },
    "ast": {
        "display_name": "AST (Liver Enzyme)",
        "kind": "numeric", "category": "blood", "unit": "U/L", "core": False,
        "aliases": ["ast", "liver enzymes", "liver enzyme", "aspartate aminotransferase",
                    "sgot", "transaminases"],
        "hard_limits": (10, 3000),
        "tolerance": 30,
        "ranges": [{"ages": (0, 200), "sex": None, "min": 40, "max": 200}],
        "severe_value": "AST or ALT >= 1,000 U/L = severe organ involvement",
        "source": "WHO Dengue guidelines 2009",
    },
    # ------------------------------------------------------ symptoms (core)
    "headache": {
        "display_name": "Headache",
        "kind": "symptom", "category": "symptom", "core": True,
        "aliases": ["headache", "severe headache"],
        "ranges": [{"ages": (0, 200), "sex": None, "prevalence": (60, 95), "default": 80}],
        "source": "WHO 2009; dengue case series",
    },
    "retro_orbital_pain": {
        "display_name": "Retro-orbital Pain",
        "kind": "symptom", "category": "symptom", "core": True,
        "aliases": ["retro orbital pain", "retroorbital pain", "eye pain", "pain behind the eyes"],
        "ranges": [
            {"ages": (0, 13), "sex": None, "prevalence": (10, 45), "default": 25},
            {"ages": (13, 200), "sex": None, "prevalence": (30, 70), "default": 50},
        ],
        "source": "WHO 2009; dengue case series (less often reported by young children)",
    },
    "muscle_joint_pain": {
        "display_name": "Muscle and Joint Pain",
        "kind": "symptom", "category": "symptom", "core": True,
        "aliases": ["muscle and joint pain", "joint muscle pain", "myalgia", "arthralgia",
                    "body ache", "muscle pain", "joint pain"],
        "ranges": [
            {"ages": (0, 13), "sex": None, "prevalence": (30, 70), "default": 50},
            {"ages": (13, 200), "sex": None, "prevalence": (60, 95), "default": 75},
        ],
        "source": "WHO 2009; dengue case series",
    },
    "nausea_vomiting": {
        "display_name": "Nausea or Vomiting",
        "kind": "symptom", "category": "symptom", "core": True,
        "aliases": ["nausea or vomiting", "nausea vomiting", "nausea", "vomiting"],
        "ranges": [
            {"ages": (0, 13), "sex": None, "prevalence": (50, 85), "default": 65},
            {"ages": (13, 200), "sex": None, "prevalence": (40, 80), "default": 55},
        ],
        "source": "WHO 2009; dengue case series",
    },
    "skin_rash": {
        "display_name": "Skin Rash",
        "kind": "symptom", "category": "symptom", "core": True,
        "aliases": ["skin rash", "rash", "maculopapular rash"],
        "ranges": [{"ages": (0, 200), "sex": None, "prevalence": (20, 60), "default": 40}],
        "source": "WHO 2009; dengue case series",
    },
    # ----------------------------------------- WHO warning signs (core)
    "abdominal_pain": {
        "display_name": "Severe Abdominal Pain",
        "kind": "symptom", "category": "warning sign", "core": True,
        "aliases": ["severe abdominal pain", "abdominal pain", "abdominal pain or tenderness"],
        "ranges": [
            {"ages": (0, 13), "sex": None, "prevalence": (20, 50), "default": 30},
            {"ages": (13, 200), "sex": None, "prevalence": (10, 40), "default": 20},
        ],
        "source": "WHO 2009 warning signs; dengue case series",
    },
    "persistent_vomiting": {
        "display_name": "Persistent Vomiting",
        "kind": "symptom", "category": "warning sign", "core": True,
        "aliases": ["persistent vomiting"],
        "ranges": [
            {"ages": (0, 13), "sex": None, "prevalence": (10, 40), "default": 20},
            {"ages": (13, 200), "sex": None, "prevalence": (5, 30), "default": 15},
        ],
        "source": "WHO 2009 warning signs; dengue case series",
    },
    "mucosal_bleeding": {
        "display_name": "Mucosal Bleeding",
        "kind": "symptom", "category": "warning sign", "core": True,
        "aliases": ["mucosal bleeding", "bleeding", "gum bleeding", "nose bleeding", "epistaxis"],
        "ranges": [{"ages": (0, 200), "sex": None, "prevalence": (5, 30), "default": 15}],
        "source": "WHO 2009 warning signs; dengue case series",
    },
    "lethargy": {
        "display_name": "Lethargy or Restlessness",
        "kind": "symptom", "category": "warning sign", "core": True,
        "aliases": ["lethargy or restlessness", "lethargy", "restlessness"],
        "ranges": [
            {"ages": (0, 13), "sex": None, "prevalence": (10, 40), "default": 20},
            {"ages": (13, 200), "sex": None, "prevalence": (5, 30), "default": 15},
        ],
        "source": "WHO 2009 warning signs; dengue case series",
    },
}

_DENGUE_ALIASES = {_normalise(a): key for key, p in DENGUE_PARAMETERS.items() for a in p["aliases"]}

_DENGUE_KEYWORDS = [
    ("persistent vomit", "persistent_vomiting"),
    ("platelet", "platelet_count"), ("thrombocyt", "platelet_count"),
    ("wbc", "wbc_count"), ("white blood", "wbc_count"), ("leukocyte", "wbc_count"),
    ("hematocrit", "hematocrit"), ("haematocrit", "hematocrit"),
    ("temperature", "body_temperature"), ("fever", "body_temperature"),
    ("heart rate", "heart_rate"), ("pulse", "heart_rate"),
    ("blood pressure", "systolic_bp"), ("systolic", "systolic_bp"),
    ("liver", "ast"), ("transaminase", "ast"),
    ("headache", "headache"),
    ("retro", "retro_orbital_pain"), ("eye pain", "retro_orbital_pain"),
    ("joint", "muscle_joint_pain"), ("muscle", "muscle_joint_pain"), ("myalgia", "muscle_joint_pain"),
    ("nausea", "nausea_vomiting"), ("vomit", "nausea_vomiting"),
    ("rash", "skin_rash"),
    ("abdominal", "abdominal_pain"),
    ("bleed", "mucosal_bleeding"),
    ("lethargy", "lethargy"), ("restless", "lethargy"),
]

# Parameters only a dengue dictionary has (temperature, heart rate etc. are
# also in healthy dictionaries, so they do not prove a file is dengue data).
DENGUE_SPECIFIC = {k for k, p in DENGUE_PARAMETERS.items()
                   if p["core"] and k != "body_temperature"}


def resolve_dengue_parameter(name: str) -> str | None:
    """Map a scraped dengue parameter name to a DENGUE_PARAMETERS key."""
    norm = _normalise(name)
    if norm in _DENGUE_ALIASES:
        return _DENGUE_ALIASES[norm]
    for word, key in _DENGUE_KEYWORDS:
        if word in norm:
            return key
    return None


def get_dengue_reference(key: str, age_low: int, age_high: int, sex: str) -> dict:
    """Dengue reference for `key`: {min, max} or {prevalence, default}."""
    r = _find_range(DENGUE_PARAMETERS[key], age_low, age_high, sex)
    if "prevalence" in r:
        return {"prevalence": r["prevalence"], "default": r["default"]}
    return {"min": r["min"], "max": r["max"]}


def base_dengue_dictionary(age_low: int, age_high: int, sex: str) -> dict:
    """The full dengue reference dictionary for one age group and sex."""
    out = {}
    for key, p in DENGUE_PARAMETERS.items():
        ref = get_dengue_reference(key, age_low, age_high, sex)
        entry = {"kind": p["kind"], "category": p["category"], "core": p["core"]}
        if p["kind"] == "symptom":
            entry.update(prevalence_percent=ref["default"], prevalence_range=list(ref["prevalence"]))
        else:
            entry.update(unit=p["unit"], **ref, dengue_value=format_range(ref["min"], ref["max"], p["unit"]),
                         severe_value=p["severe_value"])
        entry["source"] = p["source"]
        out[p["display_name"]] = entry
    return out


if __name__ == "__main__":
    import json
    import sys

    # python base_dictonary.py 20 25 male [dengue]
    args = sys.argv[1:]
    dengue = bool(args) and args[-1] == "dengue"
    if dengue:
        args = args[:-1]
    lo, hi, sex = (int(args[0]), int(args[1]), args[2]) if len(args) == 3 else (20, 25, "male")
    build = base_dengue_dictionary if dengue else base_dictionary
    print(json.dumps(build(lo, hi, sex), indent=2, ensure_ascii=False))
