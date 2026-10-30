"""
app.py

Step 3 of the RAG pipeline: Gradio front end.

    question -> user_input.analyze()      (regex name match + route)
             -> patient / follow_up : ollama_engine.generate_insight()
                dataset (no name)   : ollama_engine.retrieve_dataset() + answer_dataset()
             -> chat window

The chat keeps its context until "Clear chat" is pressed: the session
remembers the patient in focus and the last search ("what about her
platelets?", "is the second one okay?", "how many of them are female?"), and
Ollama gets the earlier turns with every question (older turns are folded
into a summary instead of being dropped).

Usage
    python app.py        then open http://127.0.0.1:7860
"""

import random
import sys
from pathlib import Path

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "BulkOutput"))
from export_csv import export_records_to_csv

from ollama_engine import (LLM_MODEL, MAX_PATIENTS, answer_conversation, answer_dataset,
                           build_context, conversation_context, generate_insight, list_models,
                           plan_text, prepare_history, retrieve_dataset)
from user_input import analyze, load_records, new_session, patient_names

MAX_TABLE_ROWS = 15         # patients listed in the match panel


def example_questions() -> list[str]:
    names = patient_names()
    picks = random.sample(names, min(2, len(names)))
    templates = ["How is {} doing?", "Does {} show dengue warning signs?"]
    return [t.format(n) for t, n in zip(templates, picks)] + [
        "How many dengue patients are there?",
        "Which dengue patients have the lowest platelet count?",
        "List female patients over 60 with high blood pressure.",
    ]


def patient_table(matches: list[dict]) -> str:
    rows = [f"| {r['name']} | {r['sex']} | {r['age']} | {r.get('condition', '?')} | {r['source_file']} #{r['id']} |"
            for r in matches[:MAX_TABLE_ROWS]]
    more = f"\n\n…and {len(matches) - MAX_TABLE_ROWS} more" if len(matches) > MAX_TABLE_ROWS else ""
    return "| Name | Sex | Age | Condition | Record |\n|---|---|---|---|---|\n" + "\n".join(rows) + more


def memory_note(session: dict, turns: int) -> str:
    focus = ", ".join(r["name"] for r in session["focus"]) or "none"
    folded = f" ({session['summarized'] // 2} summarised)" if session["summarized"] else ""
    return f"\n\n---\n*Memory: {turns} earlier turn(s){folded} · patient in focus: {focus}*"


def remember_patients(session: dict, records: list[dict]):
    for r in records:
        if r["name"] not in session["discussed"]:
            session["discussed"].append(r["name"])


def match_summary(found: dict) -> str:
    how = found["reference"] or f"{found['match_type']} name match"
    return f"**{len(found['matches'])} patient(s)** ({how})\n\n" + patient_table(found["matches"])


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


def respond(message: str, history: list, session: dict | None, model: str):
    message = (message or "").strip()
    session = session or new_session()
    history = list(history or [])
    if not message:
        yield "", history, session, gr.update(), gr.update()
        return

    model = model or LLM_MODEL
    found = analyze(message, session)
    earlier = prepare_history(history, session, model)      # before this question is added
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": ""}]

    if found["route"] == "dataset":
        history[-1]["content"] = "*Searching all records…*"
        yield "", list(history), session, "**Dataset search**…", ""
        base = within = None
        if found["refine"]:
            base = session["dataset"]["plan"]
            if base["limit"]:                   # "of them" = the patients just listed
                within = session["dataset"]["shown"]
        retrieval = retrieve_dataset(message, model, base, within)
        panel, context = dataset_summary(retrieval, found["refine"]), retrieval["context"]
        answer = answer_dataset(message, context, model, earlier, session["summary"])
        session["dataset"] = {"plan": retrieval["plan"], "shown": retrieval["shown"]}
        session["searches"].append(f"{plan_text(retrieval['plan'])} -> {len(retrieval['matches'])} match(es)")
        if len(retrieval["shown"]) == 1:        # "who is the oldest?" -> "is she okay?"
            session["focus"] = retrieval["shown"]
            remember_patients(session, retrieval["shown"])
    elif found["route"] == "conversation":
        panel, context = "**Conversation** (no search needed)", conversation_context(session)
        answer = answer_conversation(message, session, model, earlier)
    else:
        panel, context = match_summary(found), build_context(found["matches"])
        question = message
        if found["route"] == "follow_up":       # tell the model who "she" / "the second one" is
            names = ", ".join(r["name"] for r in found["matches"][:MAX_PATIENTS])
            question = f"{message}\n(This question is about: {names} - {found['reference']}.)"
        answer = generate_insight(question, found["matches"], model, earlier, session["summary"])
        session["focus"] = found["matches"]
        remember_patients(session, found["matches"][:MAX_PATIENTS])

    panel += memory_note(session, len(earlier) // 2)
    yield "", list(history), session, panel, context
    for text in answer:
        history[-1]["content"] = text
        yield "", list(history), session, panel, context


def clear():
    return [], new_session(), "Chat cleared. Memory reset.", ""


def refresh_models():
    models = list_models() or [LLM_MODEL]
    return gr.update(choices=models, value=LLM_MODEL if LLM_MODEL in models else models[0])


def download_all_csv():
    records = load_records()
    local_csv = BASE_DIR / "bulk_patients_export.csv"
    export_records_to_csv(records, output_path=local_csv)
    export_records_to_csv(records)
    return str(local_csv)


with gr.Blocks(title="Patient Health RAG Chatbot") as demo:
    gr.Markdown("# Patient Health RAG Chatbot\n"
                "Ask about a patient by name, or ask about all patients without a name "
                "(e.g. *How many dengue patients are over 60?*). Records come from "
                "`BulkOutput/output`; answers are generated by Ollama.")
    session = gr.State(new_session())

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=520)
            with gr.Row():
                question = gr.Textbox(placeholder="e.g. How is Maria Lopez doing?",
                                      show_label=False, scale=5)
                send = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear chat")
            gr.Examples(example_questions(), inputs=question)

        with gr.Column(scale=2):
            models = list_models() or [LLM_MODEL]
            model = gr.Dropdown(models, value=LLM_MODEL if LLM_MODEL in models else models[0],
                                label="Ollama model")
            refresh = gr.Button("Refresh models", size="sm")
            with gr.Row():
                gr.Markdown(f"**{len(load_records())} records** loaded.")
                download_btn = gr.DownloadButton("⬇️ Download All (CSV)", value=download_all_csv, size="sm")
            file_download = gr.File(value=download_all_csv, label="Download Patient Records (.csv)",
                                    file_count="single", interactive=False)
            matched = gr.Markdown("Matched patients will appear here.")
            with gr.Accordion("Retrieved context sent to Ollama", open=False):
                context = gr.Textbox(lines=20, show_label=False)

    outputs = [question, chatbot, session, matched, context]
    send.click(respond, [question, chatbot, session, model], outputs)
    question.submit(respond, [question, chatbot, session, model], outputs)
    clear_btn.click(clear, None, [chatbot, session, matched, context])
    refresh.click(refresh_models, None, model)
    download_btn.click(download_all_csv, outputs=[download_btn, file_download])


if __name__ == "__main__":
    allowed = [str(BASE_DIR), str(BASE_DIR.parent / "BulkOutput"), str(BASE_DIR.parent)]
    share = '--share' in sys.argv or os.environ.get('SHARE_LIVE', 'false').lower() in ('true', '1')
    demo.launch(share=share, allowed_paths=allowed)
