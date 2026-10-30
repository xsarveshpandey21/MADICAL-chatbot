# 🏥 MADICAL Chatbot - Patient Health RAG Chatbot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-success?style=for-the-badge&logo=github)](https://xsarveshpandey21.github.io/MADICAL-chatbot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An intelligent, Retrieval-Augmented Generation (RAG) medical assistant that lets healthcare workers, researchers, and clinicians query patient health records, vital signs, and dengue diagnostics through an interactive Gradio web interface powered by local LLMs via Ollama, with an instant live web demo available on GitHub Pages.

👉 **[Launch the Live Web Demo on GitHub](https://xsarveshpandey21.github.io/MADICAL-chatbot/)**

---

## ✨ Features

- **🔍 Patient-Centric Lookup**: Ask natural language questions about individual patients by name (e.g., *"How is Maria Lopez doing?"*, *"Does Carlos Mendoza show dengue warning signs?"*).
- **📊 Dataset-Level Analytics**: Run structured queries across patient populations (e.g., *"Which dengue patients have the lowest platelet count?"*, *"How many female patients over 60 have high blood pressure?"*).
- **🩺 Clinical Range Grounding**: Automatically evaluates vital signs (Blood Pressure, Heart Rate, Platelets, WBC, Temperature, Hematocrit) against age- and sex-stratified healthy baseline dictionaries.
- **🧠 Multi-Turn Memory**: Maintains conversation context and automatically condenses older dialogue turns into concise summaries without losing focus on the current patient.
- **📥 CSV Exporting**: Export matched patient records or entire patient datasets directly to CSV from the UI.
- **🌐 Instant Live Access**:
  - **Live Web Demo**: Hosted directly via GitHub Pages at [xsarveshpandey21.github.io/MADICAL-chatbot](https://xsarveshpandey21.github.io/MADICAL-chatbot/)
  - **Gradio Tunnel**: Supports instant public live URL tunneling via Gradio `--share`

---

## 🌐 Live Web Demo

You can try the interactive demo immediately in your browser without installing anything locally:
- **Demo URL**: [https://xsarveshpandey21.github.io/MADICAL-chatbot/](https://xsarveshpandey21.github.io/MADICAL-chatbot/)
- **Features in the Demo**:
  - Interactive clinical AI chat assistant
  - 231 filterable, searchable synthetic patient records
  - Real-time vital signs comparison (thrombocytopenia, blood pressure, narrow pulse pressure &le; 20 mmHg)
  - Diagnostic criteria for WHO Dengue warning signs
  - Direct CSV export

---

## 🏗️ Project Architecture

```
MADICAL-chatbot/
├── app.py                     # Root entry point to launch the Gradio web UI
├── run.py                     # Convenience runner script
├── run.bat                    # Windows batch launcher (uses virtualenv if present)
├── requirements.txt           # Project dependencies
├── BUGS_AND_FIXES.md          # Comprehensive report of bug analyses and applied fixes
├── docs/                      # GitHub Pages live interactive demo
│   ├── index.html             # Single-page interactive clinical dashboard & chat engine
│   └── data/                  # Patient JSON and CSV records for client exploration
├── RAGChatBot/                # Main RAG application
│   ├── app.py                 # Gradio chat interface and state management
│   ├── ollama_engine.py       # Query planning, dataset retrieval & LLM generation
│   ├── user_input.py          # Entity extraction, regex name matching & session memory
│   └── requirements.txt
├── BulkOutput/                # Patient dataset generation and CSV outputs
│   ├── bulkData.py            # Generates synthetic patient records with realistic vitals
│   ├── export_csv.py          # Data export utilities
│   └── output/                # Synthesized patient records (CSV & JSON)
├── Category_Dict/             # Web scrapers and medical baseline ranges
│   ├── category_good.py       # Scrapes & structures normal reference ranges
│   ├── category_dengue.py     # Structures dengue warning signs and diagnostic criteria
│   └── output/                # Baseline dictionaries stratified by age & sex
└── Data_Distl/                # Data cleaning and dictionary curation pipeline
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- **Python 3.10+**
- **Ollama**: Download and install Ollama from [ollama.ai](https://ollama.ai).
  Pull your desired model (e.g., `llama3` or `qwen2.5`):
  ```bash
  ollama pull llama3
  ```

### 2. Clone the Repository

```bash
git clone https://github.com/xsarveshpandey21/MADICAL-chatbot.git
cd MADICAL-chatbot
```

### 3. Create a Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Running the Application Locally

### Option A: Local Run
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:7860
```

### Option B: Make It Live (Public Gradio Link)
To share the chatbot live over the internet with a temporary public URL (`https://xxxx.gradio.live`):
```bash
python app.py --share
```
or set the environment variable:
```bash
# Windows PowerShell:
$env:SHARE_LIVE="true"; python app.py

# Windows Command Prompt:
set SHARE_LIVE=true && python app.py

# Linux/macOS:
SHARE_LIVE=true python app.py
```

### Option C: Windows Batch Runner
Double-click `run.bat` or execute:
```cmd
run.bat
```

---

## 💬 Sample Queries

| Query Type | Example Prompt |
|---|---|
| **Patient Status** | *"How is Maria Lopez doing?"* |
| **Dengue Symptoms** | *"Does Carlos Mendoza show severe dengue warning signs?"* |
| **Vital Statistics** | *"What is Sarah Jenkins' pulse pressure and platelet level?"* |
| **Dataset Aggregation** | *"How many dengue patients are there?"* |
| **Extreme Values** | *"Which dengue patients have the lowest platelet count?"* |
| **Filtered Search** | *"List female patients over 60 with high blood pressure."* |

---

## 🐛 Bug Fixes & Improvements

For an in-depth breakdown of bugs detected, root causes, reproduction steps, and resolved issues, consult [BUGS_AND_FIXES.md](BUGS_AND_FIXES.md).

---

## 📄 License

This project is licensed under the MIT License.
