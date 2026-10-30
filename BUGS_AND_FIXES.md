# Bugs and Fixes Report: Patient Health RAG Chatbot

This document details all bugs discovered in the codebase, explains their root causes, demonstrates how to reproduce them, and provides the exact fixes applied across the project.

---

## Quick Reference Table

| # | Bug Title | Affected File | Severity | Status |
|---|---|---|---|---|
| 1 | `TypeError` when `history` is `None` | `RAGChatBot/app.py` | **High** | **Fixed** |
| 2 | `AttributeError` on Uncached Name Search | `RAGChatBot/user_input.py` | **High** | **Fixed** |
| 3 | Side Panel Table Count/Limit Mismatch | `RAGChatBot/app.py` | **Medium** | **Fixed** |
| 4 | Gradio Streaming Chunk Lag / Missed Diffing | `RAGChatBot/app.py` | **Medium** | **Fixed** |
| 5 | Natural Age Query Parsing Failures | `RAGChatBot/ollama_engine.py` | **Medium** | **Fixed** |
| 6 | False-Positive Pronoun Pollution in Comparison | `RAGChatBot/user_input.py` | **Medium** | **Fixed** |
| 7 | Ordinal Regex Too Restrictive & Limited to 5 | `RAGChatBot/user_input.py` | **Low** | **Fixed** |
| 8 | Floating-Point Precision in Pulse Pressure | `RAGChatBot/ollama_engine.py` | **Low** | **Fixed** |
| 9 | Potential `TypeError` on Missing Vital Value in `flag()` | `RAGChatBot/ollama_engine.py` | **Low** | **Fixed** |
| 10 | Crash on Empty Records in `dataset_overview()` | `RAGChatBot/ollama_engine.py` | **Low** | **Fixed** |
| 11 | `run.bat` Environment PATH Mismatch | `run.bat` | **Low** | **Fixed** |
| 12 | Unsafe Dict Access `flags[k]` in `record_row()` | `RAGChatBot/ollama_engine.py` | **Medium** | **Fixed** |
| 13 | Negative / Sub-30 Diastolic Pressure in `fix_blood_pressure()` | `BulkOutput/bulkData.py` | **Medium** | **Fixed** |
| 14 | Capitalized Sex Discarding Generated People | `BulkOutput/bulkData.py` | **Medium** | **Fixed** |
| 15 | Unchecked Record `name` Key in `load_records()` | `RAGChatBot/user_input.py` | **Low** | **Fixed** |
| 16 | Missing Scraper Dependencies for `Category_Dict` | `Category_Dict/requirements.txt` | **Low** | **Documented** |
| 17 | Unrealistic General Human Age Range Upper Bound (200 -> 100) | `RAGChatBot/ollama_engine.py` | **Low** | **Fixed** |

---

## Detailed Bug Breakdown & Fixes

---

### Bug 1: `TypeError` when `history` is `None` on Empty Chatbot State
* **File**: `RAGChatBot/app.py` (Lines 76–87)
* **Severity**: High (Runtime Crash)

#### Why it happened
When a user launches the app or resets the chat, Gradio can send `history = None` into event callbacks. In `app.py`, the code guarded `session = session or new_session()`, but did not guard `history`. When `prepare_history(history, ...)` was called, iterating over `None` raised `TypeError: 'NoneType' object is not iterable`. Furthermore, `history + [...]` raised `TypeError: can only concatenate list (not "NoneType") to list`.

#### How to see / reproduce it
```bash
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'RAGChatBot'); from app import respond; gen = respond('How is Maria Lopez doing?', None, None, 'llama3:8b'); next(gen)"
```
*Output before fix:*
```text
TypeError: 'NoneType' object is not iterable
```

#### How to fix it
Add `history = list(history or [])` at the entry point of `respond()`:

```python
# --- BEFORE ---
def respond(message: str, history: list, session: dict | None, model: str):
    message = (message or "").strip()
    session = session or new_session()
    if not message:
        yield "", history, session, gr.update(), gr.update()
        return

# --- AFTER (FIXED) ---
def respond(message: str, history: list, session: dict | None, model: str):
    message = (message or "").strip()
    session = session or new_session()
    history = list(history or [])  # <-- Guard against None
    if not message:
        yield "", history, session, gr.update(), gr.update()
        return
```

Also protect `chat_turns()` in `RAGChatBot/ollama_engine.py`:
```python
# --- BEFORE ---
def chat_turns(history: list) -> list[dict]:
    return [{"role": m["role"], "content": message_text(m["content"])}
            for m in history if m.get("role") in ("user", "assistant")]

# --- AFTER (FIXED) ---
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
```

---

### Bug 2: `AttributeError` on Unset / Empty Full Name Cache
* **File**: `RAGChatBot/user_input.py` (Line 143)
* **Severity**: High (Potential Uncaught Exception)

#### Why it happened
In `find_names()`, partial name matching had a guard `if _cache["parts"]:`, but full name matching directly called `_cache["full"].finditer(text)`. If records were empty or `find_names()` was called before records were indexed, `_cache["full"]` was `None`, resulting in:
`AttributeError: 'NoneType' object has no attribute 'finditer'`.

#### How to see / reproduce it
```bash
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'RAGChatBot'); import user_input; user_input._cache['full'] = None; print(user_input.find_names('Maria', [{'name': 'Maria'}]))"
```

#### How to fix it
Guard `if _cache["full"]:` before calling `.finditer()`:

```python
# --- BEFORE ---
found = {_normalise(m.group(0)) for m in _cache["full"].finditer(text)}
if found:
    result.update(match_type="full", names_found=sorted(found),
                  matches=[r for r in records if _normalise(r["name"]) in found])
    return result

# --- AFTER (FIXED) ---
if _cache["full"]:
    found = {_normalise(m.group(0)) for m in _cache["full"].finditer(text)}
    if found:
        result.update(match_type="full", names_found=sorted(found),
                      matches=[r for r in records if _normalise(r["name"]) in found])
        return result
```

---

### Bug 3: Side Panel Table Displaying 15+ Records on Limited Queries (e.g., "Top 3")
* **File**: `RAGChatBot/app.py` (Line 73)
* **Severity**: Medium (Data & UI Inconsistency)

#### Why it happened
When a user asked a query with a limit such as *"Which dengue patients have the lowest platelets? Top 3"*, `retrieval["plan"]["limit"]` was set to 3. The LLM prompt correctly received only the 3 patients (`retrieval["shown"]`), but `dataset_summary()` passed `retrieval["matches"]` (all 26 matching dengue records) directly to `patient_table()`, displaying 15 patients in the side panel with text saying "26 matching record(s)".

#### How to see / reproduce it
1. Ask the chatbot: *"Which dengue patients have the lowest platelet count? Top 3"*
2. Check the right-hand side panel. Before the fix, it showed 15 rows instead of 3.

#### How to fix it
```python
# --- BEFORE ---
def dataset_summary(retrieval: dict, refine: bool) -> str:
    title = "Narrowed the last search" if refine else "Dataset search (no name in question)"
    text = f"**{title}**\n\nSearch: {plan_text(retrieval['plan'])}\n\n"
    if not retrieval["matches"]:
        return text + "No matching records."
    return text + f"**{len(retrieval['matches'])} matching record(s)**\n\n" + patient_table(retrieval["matches"])

# --- AFTER (FIXED) ---
def dataset_summary(retrieval: dict, refine: bool) -> str:
    title = "Narrowed the last search" if refine else "Dataset search (no name in question)"
    text = f"**{title}**\n\nSearch: {plan_text(retrieval['plan'])}\n\n"
    if not retrieval["matches"]:
        return text + "No matching records."
    limit = retrieval.get("plan", {}).get("limit") or 0
    if limit > 0 and len(retrieval["matches"]) > limit:
        displayed = retrieval.get("shown", retrieval["matches"][:limit])
        header = f"**Showing top {len(displayed)} of {len(retrieval['matches'])} matching record(s)**\n\n"
        return text + header + patient_table(displayed)
    return text + f"**{len(retrieval['matches'])} matching record(s)**\n\n" + patient_table(retrieval["matches"])
```

---

### Bug 4: Gradio In-Place Mutation Breaking Token Streaming
* **File**: `RAGChatBot/app.py` (Lines 118–123)
* **Severity**: Medium (UI Glitch / Streaming Drop)

#### Why it happened
In `respond()`, `history[-1]["content"] = text` mutated the list in place and yielded `history`. Gradio's internal state tracker relies on object diffing. When the list reference does not change, intermediate token streaming updates can be skipped or batched awkwardly.

#### How to fix it
Yield a shallow copy `list(history)` on each token chunk:

```python
# --- BEFORE ---
panel += memory_note(session, len(earlier) // 2)
yield "", history, session, panel, context
for text in answer:
    history[-1]["content"] = text
    yield "", history, session, panel, context

# --- AFTER (FIXED) ---
panel += memory_note(session, len(earlier) // 2)
yield "", list(history), session, panel, context
for text in answer:
    history[-1]["content"] = text
    yield "", list(history), session, panel, context
```

---

### Bug 5: Age Range Regex Failure for Natural Queries ("are over 60")
* **File**: `RAGChatBot/ollama_engine.py` (Lines 361–365)
* **Severity**: Medium (Faulty Query Planning)

#### Why it happened
The regular expression for detecting age thresholds was:
```python
rf"\b{AGE_NOUN}\s+(?:who are\s+)?(?:over|above|older than)\s+(\d{{1,3}})\b"
```
The group `(?:who are\s+)?` matched `"patients over 60"` or `"patients who are over 60"`, but **failed completely** on natural questions such as:
- *"How many dengue patients **are** over 60?"*
- *"Which patients **are** under 18?"*
Because the word `"are "` was not matched, `age_range()` returned `None`, and the age filter defaulted back to `0–200`.

#### How to see / reproduce it
```bash
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'RAGChatBot'); from ollama_engine import age_range; print('are over 60:', age_range('how many dengue patients are over 60?'))"
```
*Output before fix:* `None` (failed to extract age)

#### How to fix it
Update regexes to make `"who"` optional when followed by `"are"` / `"is"`:

```python
# --- BEFORE ---
over = (re.search(rf"\b{AGE_NOUN}\s+(?:who are\s+)?(?:over|above|older than)\s+(\d{{1,3}})\b", q)
        or re.search(r"\b(?:over|above|older than)\s+(\d{1,3})\s*(?:years|yrs|y/?o)\b", q)
        or re.search(r"\b(\d{1,3})\s*(?:\+|years? (?:and|or) (?:older|over|above))", q))
under = (re.search(rf"\b{AGE_NOUN}\s+(?:who are\s+)?(?:under|below|younger than)\s+(\d{{1,3}})\b", q)
         or re.search(r"\b(?:under|below|younger than)\s+(\d{1,3})\s*(?:years|yrs|y/?o)\b", q))

# --- AFTER (FIXED) ---
over = (re.search(rf"\b{AGE_NOUN}\s+(?:(?:who\s+)?(?:are|is)\s+)?(?:over|above|older than)\s+(\d{{1,3}})\b", q)
        or re.search(r"\b(?:over|above|older than)\s+(\d{1,3})\s*(?:years|yrs|y/?o)?\b", q)
        or re.search(r"\b(\d{1,3})\s*(?:\+|years? (?:and|or) (?:older|over|above))", q))
under = (re.search(rf"\b{AGE_NOUN}\s+(?:(?:who\s+)?(?:are|is)\s+)?(?:under|below|younger than)\s+(\d{{1,3}})\b", q)
         or re.search(r"\b(?:under|below|younger than)\s+(\d{1,3})\s*(?:years|yrs|y/?o)?\b", q))
```
*Output after fix:* `(61, 100)`

---

### Bug 6: False-Positive Pronoun Pollution in Single-Patient Questions
* **File**: `RAGChatBot/user_input.py` (Lines 202–206)
* **Severity**: Medium (Incorrect Context Assembly)

#### Why it happened
If a patient was already in focus (e.g. `Julian Patel`), and the user asked about a new single patient:
*"Is Maria Lopez in her 40s?"*
The pronoun detector matched `"her"`. Line 204 checked:
`if singular and focus: matches = focus + new`
The system mistakenly assumed the user wanted to compare Maria with Julian, adding Julian Patel into the context and telling the model: `(named + previous patient)`.

#### How to fix it
Require explicit comparison words (such as `compare`, `with`, `versus`, `vs`, `both`) before merging with the previous patient in focus:

```python
# --- BEFORE ---
if result["matches"]:
    result["route"] = "patient"
    if singular and focus:          # "compare her with Aidan Kim"
        new = [r for r in result["matches"] if r not in focus]
        result.update(matches=focus + new, reference="named + previous patient")

# --- AFTER (FIXED) ---
if result["matches"]:
    result["route"] = "patient"
    is_comparison = bool(re.search(r"\b(compare|versus|vs|both|along with|with)\b", text, re.IGNORECASE))
    if singular and focus and is_comparison:
        new = [r for r in result["matches"] if r not in focus]
        result.update(matches=focus + new, reference="named + previous patient")
```

---

### Bug 7: Ordinal Regex Too Restrictive & Limited to 5 Patients
* **File**: `RAGChatBot/user_input.py` (Lines 84–88, 173–180)
* **Severity**: Low (Limitation / Edge Case)

#### Why it happened
1. `ORDINALS` dictionary only went from `first` to `fifth`. Questions asking for `6th` to `10th` failed.
2. The regex required `(one|patient|person|record|name)` immediately after the ordinal. Questions like *"Is the second okay?"* or *"What about the 3rd?"* did not match and fell through to full dataset search.

#### How to fix it
```python
# --- BEFORE ---
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4,
            "4th": 4, "fifth": 5, "5th": 5, "last": -1}
ORDINAL_WORDS = re.compile(
    rf"\b({'|'.join(ORDINALS)})\s+(one|patient|person|record|name)\b|#\s?(\d+)\b|\bnumber\s+(\d+)\b",
    re.IGNORECASE)

def ordinal_pick(text: str, shown: list[dict]) -> list[dict]:
    m = ORDINAL_WORDS.search(text)
    if not m or not shown:
        return []
    n = ORDINALS[m[1].lower()] if m[1] else int(m[3] or m[4])
    if n == -1:
        return [shown[-1]]
    return [shown[n - 1]] if 1 <= n <= len(shown) else []

# --- AFTER (FIXED) ---
ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9, "tenth": 10, "10th": 10, "last": -1,
}
ORDINAL_WORDS = re.compile(
    rf"\b({'|'.join(ORDINALS)})(?:\s+(?:one|patient|person|record|name))?\b|#\s?(\d+)\b|\bnumber\s+(\d+)\b",
    re.IGNORECASE)

def ordinal_pick(text: str, shown: list[dict]) -> list[dict]:
    m = ORDINAL_WORDS.search(text)
    if not m or not shown:
        return []
    n = ORDINALS[m[1].lower()] if m[1] else int(m[2] or m[3])
    if n == -1:
        return [shown[-1]]
    return [shown[n - 1]] if 1 <= n <= len(shown) else []
```

---

### Bug 8: Floating-Point Precision Error in Pulse Pressure
* **File**: `RAGChatBot/ollama_engine.py` (Line 238)
* **Severity**: Low (Clinical Value Formatting)

#### Why it happened
Floating-point arithmetic on blood pressure values (e.g. `110.1 - 90.0`) yielded `20.099999999999994`, which was then embedded directly into LLM prompts as `Pulse Pressure: 20.099999999999994 mmHg`.

#### How to fix it
```python
# --- BEFORE ---
def pulse_pressure(record: dict) -> float | None:
    vitals = record.get("vitals", {})
    if "Systolic Blood Pressure" in vitals and "Diastolic Blood Pressure" in vitals:
        return vitals["Systolic Blood Pressure"]["value"] - vitals["Diastolic Blood Pressure"]["value"]
    return None

# --- AFTER (FIXED) ---
def pulse_pressure(record: dict) -> float | None:
    vitals = record.get("vitals", {})
    if "Systolic Blood Pressure" in vitals and "Diastolic Blood Pressure" in vitals:
        sys_val = vitals["Systolic Blood Pressure"].get("value")
        dia_val = vitals["Diastolic Blood Pressure"].get("value")
        if sys_val is not None and dia_val is not None:
            return round(sys_val - dia_val, 1)
    return None
```

---

### Bug 9: Safe Flagging for Missing Lab Values in `flag()`
* **File**: `RAGChatBot/ollama_engine.py` (Lines 216–223)
* **Severity**: Low (Defensive Coding)

#### Why it happened
If any synthetic record had a vital value that was `None`, comparing `value < rng["min"]` raised:
`TypeError: '<' not supported between instances of 'NoneType' and 'float'`.

#### How to fix it
```python
# --- BEFORE ---
def flag(value: float, rng: dict | None) -> str:
    if not rng:
        return "no reference"
    if rng["min"] is not None and value < rng["min"]:
        return "LOW"
    if rng["max"] is not None and value > rng["max"]:
        return "HIGH"
    return "normal"

# --- AFTER (FIXED) ---
def flag(value: float | None, rng: dict | None) -> str:
    if value is None or not rng:
        return "no reference"
    if rng.get("min") is not None and value < rng["min"]:
        return "LOW"
    if rng.get("max") is not None and value > rng["max"]:
        return "HIGH"
    return "normal"
```

---

### Bug 10: Empty Records Handling in `dataset_overview()`
* **File**: `RAGChatBot/ollama_engine.py` (Lines 556–565)
* **Severity**: Low (Crash on Empty Files)

#### Why it happened
If `dataset_overview()` was called when no records existed or `records` was empty, calling `min(ages)` raised:
`ValueError: min() arg is an empty sequence`.

#### How to fix it
```python
# --- BEFORE ---
def dataset_overview(records: list[dict], flags: dict) -> str:
    files = sorted({r["source_file"] for r in records})
    conditions = Counter(r.get("condition", "unknown") for r in records)
    sexes = Counter(r["sex"] for r in records)
    ages = [r["age"] for r in records]
    lines = [
        ...
        + f" | Age: {min(ages)}-{max(ages)} (median {median(ages)})",

# --- AFTER (FIXED) ---
def dataset_overview(records: list[dict], flags: dict) -> str:
    if not records:
        return "Dataset: 0 records."
    files = sorted({r["source_file"] for r in records if "source_file" in r})
    conditions = Counter(r.get("condition", "unknown") for r in records)
    sexes = Counter(r.get("sex", "unknown") for r in records)
    ages = [r["age"] for r in records if "age" in r]
    age_str = f"{min(ages)}-{max(ages)} (median {median(ages)})" if ages else "none"
    lines = [
        ...
        + f" | Age: {age_str}",
```

---

### Bug 11: `run.bat` Environment Isolation
* **File**: `run.bat`
* **Severity**: Low (Setup / Portability Issue)

#### Why it happened
`run.bat` executed `python app.py` against system PATH Python. If the project dependencies (`gradio`, `ollama`) were installed in `.venv` (as in this workspace), double-clicking `run.bat` crashed with:
`ModuleNotFoundError: No module named 'gradio'`.

#### How to fix it
Direct the script to the local virtual environment Python if available:

```bat
:: --- BEFORE ---
@echo off
cd /d "%~dp0RAGChatBot"
python app.py
pause

:: --- AFTER (FIXED) ---
@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    cd RAGChatBot
    "..\.venv\Scripts\python.exe" app.py
) else (
    cd RAGChatBot
    python app.py
)
pause
```

---

### Bug 12: Unsafe Dict Access `flags[k]` in `record_row()`
* **File**: `RAGChatBot/ollama_engine.py` (Line 601)
* **Severity**: Medium (KeyError on Custom Vitals)

#### Why it happened
In `record_row()`, when displaying patient rows in dataset retrieval context, `flags[k]` was accessed directly for every sorted vital:
```python
key_values = [f"{k} {vitals[k]['value']} {vitals[k]['unit']} ({flags[k]})" for k in shown if k in vitals]
```
If a record possessed a vital that had no baseline reference in `reference_ranges()` (and therefore no entry in `flags`), `flags[k]` raised an unhandled `KeyError`.

#### How to fix it
Use `flags.get(k, "normal")` and safe value/unit getters:
```python
# --- AFTER (FIXED) ---
key_values = [f"{k} {vitals[k].get('value')} {vitals[k].get('unit', '')} ({flags.get(k, 'normal')})"
              for k in shown if k in vitals]
```

---

### Bug 13: Sub-30 or Negative Diastolic Pressure in `fix_blood_pressure()`
* **File**: `BulkOutput/bulkData.py` (Line 293)
* **Severity**: Medium (Physically Impossible Clinical Value)

#### Why it happened
In `fix_blood_pressure()`:
```python
if sys_bp and dia_bp and dia_bp["value"] > sys_bp["value"] - 20:
    dia_bp["value"] = sys_bp["value"] - random.randint(20, 45)
```
If a patient had an extremely low systolic pressure (e.g. 45 mmHg), subtracting `random.randint(20, 45)` could set diastolic pressure to 0 mmHg or a negative value.

#### How to fix it
Clamp diastolic pressure to a physiological minimum of 30 mmHg:
```python
# --- AFTER (FIXED) ---
dia_bp["value"] = max(30, sys_bp["value"] - random.randint(20, 45))
```

---

### Bug 14: Capitalized Sex Discarding Generated People
* **File**: `BulkOutput/bulkData.py` (Line 215)
* **Severity**: Medium (Ollama Schema Sensitivity)

#### Why it happened
When Ollama returned people with `"sex": "Male"` or `"sex": "Female"`, line 216 evaluated:
`if sex not in ages_by_sex:`
Since `ages_by_sex` only contains lowercase keys `{"male", "female"}`, capitalized strings caused valid generated individuals to be silently discarded.

#### How to fix it
Normalize sex with `.lower()`:
```python
# --- AFTER (FIXED) ---
sex = str(p.get("sex", "")).lower()
```

---

### Bug 15: Unchecked Record `name` Key in `load_records()` and `patient_names()`
* **File**: `RAGChatBot/user_input.py` (Lines 129, 227)
* **Severity**: Low (Malformed JSON Vulnerability)

#### Why it happened
`full_names = {_normalise(r["name"]) for r in records}` directly accessed `r["name"]`. If any record was missing a name, it triggered a `KeyError`.

#### How to fix it
Add `if r.get("name")` guards:
```python
# --- AFTER (FIXED) ---
full_names = {_normalise(r["name"]) for r in records if r.get("name")}
```

---

### Bug 16: Missing Scraper Dependencies for `Category_Dict`
* **File**: `Category_Dict/category_good.py`, `Category_Dict/category_dengue.py`
* **Severity**: Low (Environment Setup)

#### Why it happened
`category_good.py` requires `requests`, `beautifulsoup4`, and `ddgs`. These dependencies were listed in `Category_Dict/requirements.txt` but not installed in the workspace `.venv`.

#### How to fix it
If scraping new categories is required, install:
```bash
pip install requests beautifulsoup4 ddgs
```

### Bug 17: Unrealistic General Human Age Range Upper Bound (200 -> 100)
* **File**: `RAGChatBot/ollama_engine.py` (Lines 115, 175, 342, 381)
* **Severity**: Low (Display / Query Aesthetics)

#### Why it happened
`DEFAULT_PLAN["age_max"]` and `PLAN_PROMPT` used `200` as the default unbounded upper age limit. When users asked questions with lower bounds such as *"over 60"* or *"dengue patients over 60"*, the query planner resolved the age filter to `age 61-200`. In human clinical datasets, age range naturally caps around 100, making `61-200` appear unnatural and inaccurate in UI search summaries.

#### How to see / reproduce it
```bash
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'RAGChatBot'); from ollama_engine import age_range, plan_text, DEFAULT_PLAN; p = dict(DEFAULT_PLAN); p['condition'] = 'dengue'; p['age_min'] = 61; print(plan_text(p))"
```
*Output before fix:* `condition = dengue; age 61-200`

#### How to fix it
1. Update `PLAN_PROMPT` to ask the LLM for `100 for age_max` instead of `200`.
2. Update `DEFAULT_PLAN["age_max"]` from `200` to `100`.
3. In `clean_plan()`, clamp any raw `age_max > 100` back to `DEFAULT_PLAN["age_max"]`.
4. In `age_range()`, cap `hi = min(hi, 100)` and update the threshold check to `lo > 100`.

*Output after fix:* `condition = dengue; age 61-100`

---

## How to Verify All Fixes

You can verify all fixed components with these quick commands:

```powershell
# 1. Verify compilation
.\.venv\Scripts\python.exe -m py_compile app.py run.py RAGChatBot/app.py RAGChatBot/user_input.py RAGChatBot/ollama_engine.py

# 2. Test history None safety
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'RAGChatBot'); from app import respond; gen = respond('How is Maria Lopez doing?', None, None, 'llama3:8b'); print('Passed:', next(gen) is not None)"

# 3. Test age parsing
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'RAGChatBot'); from ollama_engine import age_range; print('are over 60:', age_range('how many dengue patients are over 60?'))"

# 4. Test ordinals
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'RAGChatBot'); from user_input import ordinal_pick; print('second:', ordinal_pick('is the second okay?', [{'name': 'A'}, {'name': 'B'}]))"

# 5. Test Bulk Output CSV Export
.\.venv\Scripts\python.exe BulkOutput/export_csv.py
```
