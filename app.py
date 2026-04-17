import json
import gradio as gr

import stt
import intent as intent_module
from memory import clear_history, load_history, save_history
from tools import create_file, generate_and_save_code, summarize_text, general_chat


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _execute_intent(item: dict, full_text: str, history: list[dict]) -> dict:
    """Run a single classified intent and return a result dict."""
    kind = item.get("intent", "general_chat")
    desc = item.get("description", full_text)
    fname = item.get("filename")
    lang = item.get("language")

    if kind == "create_file":
        filename = fname or "new_file.txt"
        msg = create_file(filename)
        return {"intent": kind, "action": f"Create file → {filename}", "result": msg}

    if kind == "write_code":
        result = generate_and_save_code(desc, filename=fname, language=lang)
        code_block = f"```{lang or 'python'}\n{result['code']}\n```" if result["code"] else ""
        return {
            "intent": kind,
            "action": f"Generate code → {result['filename']}",
            "result": f"{result['message']}\n\n{code_block}",
        }

    if kind == "summarize":
        summary = summarize_text(desc)
        if fname:
            save_msg = create_file(fname, summary)
            return {
                "intent": kind,
                "action": f"Summarize and save → {fname}",
                "result": f"{summary}\n\n---\n{save_msg}",
            }
        return {"intent": kind, "action": "Summarize text", "result": summary}

    # general_chat (default)
    reply = general_chat(desc, chat_history=history)
    return {"intent": "general_chat", "action": "Chat response", "result": reply}


def process_audio(audio_path, confirm_file_ops, history):
    """End-to-end pipeline: transcribe → classify → execute → display."""
    history = history or []

    # --- Step 1: Transcribe ---
    if audio_path is None:
        return "No audio provided.", "{}", "—", "Please record or upload audio.", history

    transcription = stt.transcribe(audio_path)
    if transcription.startswith("ERROR:"):
        return transcription, "{}", "—", transcription, history

    # --- Step 2: Classify intent ---
    intents = intent_module.classify(transcription, chat_history=history)
    intents_json = json.dumps({"intents": intents}, indent=2)

    # --- Step 3: Human-in-the-loop check ---
    file_intents = {"create_file", "write_code"}
    needs_confirm = any(i["intent"] in file_intents for i in intents)
    if needs_confirm and not confirm_file_ops:
        return (
            transcription,
            intents_json,
            "Blocked — file operation requires confirmation",
            "Toggle **'Confirm file operations'** and try again to allow file creation.",
            history,
        )

    # --- Step 4: Execute each intent ---
    actions = []
    results = []
    for item in intents:
        out = _execute_intent(item, transcription, history)
        actions.append(out["action"])
        results.append(out["result"])

    actions_str = "\n".join(f"• {a}" for a in actions)
    results_str = "\n\n---\n\n".join(results)

    # --- Step 5: Update session history ---
    history.append({"role": "user", "content": transcription})
    history.append({"role": "assistant", "content": results_str})
    save_history(history)

    return transcription, intents_json, actions_str, results_str, history


# ---------------------------------------------------------------------------
# Text-input pipeline (for testing without audio)
# ---------------------------------------------------------------------------

def process_text(text, confirm_file_ops, history):
    """Same pipeline but starting from typed text instead of audio."""
    history = history or []

    if not text or not text.strip():
        return "No text provided.", "{}", "—", "Please type a command.", history

    transcription = text.strip()
    intents = intent_module.classify(transcription, chat_history=history)
    intents_json = json.dumps({"intents": intents}, indent=2)

    file_intents = {"create_file", "write_code"}
    needs_confirm = any(i["intent"] in file_intents for i in intents)
    if needs_confirm and not confirm_file_ops:
        return (
            transcription,
            intents_json,
            "Blocked — file operation requires confirmation",
            "Toggle **'Confirm file operations'** and try again to allow file creation.",
            history,
        )

    actions = []
    results = []
    for item in intents:
        out = _execute_intent(item, transcription, history)
        actions.append(out["action"])
        results.append(out["result"])

    actions_str = "\n".join(f"• {a}" for a in actions)
    results_str = "\n\n---\n\n".join(results)

    history.append({"role": "user", "content": transcription})
    history.append({"role": "assistant", "content": results_str})
    save_history(history)

    return transcription, intents_json, actions_str, results_str, history


def reset_memory():
    """Clear in-memory and persisted session history."""
    clear_history()
    return [], "Session memory cleared from disk."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Voice-Controlled AI Agent

Speak or type a command. The agent transcribes your audio, detects intent, and executes the action.

| Intent | What it does | Example prompt |
|---|---|---|
| **Create File** | Creates an empty file/folder in `output/` | *"Create a file called notes.txt"* |
| **Write Code** | Generates code and saves it to `output/` | *"Write a Python function for binary search"* |
| **Summarize** | Summarizes the given text (optionally saves to file) | *"Summarize: The quick brown fox..."* |
| **General Chat** | Answers questions conversationally | *"What is recursion?"* |

**Tips:** You can combine multiple commands in one prompt.
Example: *"Create a folder called project, write a script to calculate the fibonacci sequence, and summarize what recursion is."*
Enable the **Confirm file operations** checkbox before any file/code commands.
"""

with gr.Blocks(title="Voice AI Agent") as demo:
    gr.Markdown(DESCRIPTION)
    session_history = gr.State(load_history())

    with gr.Row():
        # ---- Left column: inputs ----
        with gr.Column(scale=1):
            with gr.Tab("Record"):
                audio_mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record audio",
                )
            with gr.Tab("Upload"):
                audio_upload = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Upload audio file",
                )
            with gr.Tab("Text"):
                text_input = gr.Textbox(
                    lines=3,
                    placeholder="Type a command instead of speaking…",
                    label="Text input",
                )

            confirm_toggle = gr.Checkbox(
                label="Confirm file operations",
                value=False,
                info="Enable to allow the agent to create/write files in output/",
            )

            with gr.Row():
                btn_mic = gr.Button("Process Recording", variant="primary")
                btn_upload = gr.Button("Process Upload", variant="primary")
                btn_text = gr.Button("Process Text", variant="primary")
            btn_clear_memory = gr.Button("Clear Persistent Memory", variant="secondary")

        # ---- Right column: outputs ----
        with gr.Column(scale=2):
            out_transcription = gr.Textbox(label="Transcribed Text", lines=2, interactive=False)
            out_intent = gr.Code(label="Detected Intent(s)", language="json", interactive=False)
            out_action = gr.Textbox(label="Action Taken", lines=2, interactive=False)
            out_result = gr.Markdown(label="Result")
            memory_status = gr.Textbox(label="Memory Status", lines=1, interactive=False)

    # ---- Wiring ----
    shared_outputs = [out_transcription, out_intent, out_action, out_result, session_history]

    btn_mic.click(
        fn=process_audio,
        inputs=[audio_mic, confirm_toggle, session_history],
        outputs=shared_outputs,
    )
    btn_upload.click(
        fn=process_audio,
        inputs=[audio_upload, confirm_toggle, session_history],
        outputs=shared_outputs,
    )
    btn_text.click(
        fn=process_text,
        inputs=[text_input, confirm_toggle, session_history],
        outputs=shared_outputs,
    )
    btn_clear_memory.click(
        fn=reset_memory,
        inputs=[],
        outputs=[session_history, memory_status],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
