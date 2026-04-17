# Voice-Controlled Local AI Agent

A voice-controlled AI agent that accepts audio input, classifies user intent, executes local tools, and displays the entire pipeline in a clean Gradio UI.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Audio Input  │────▶│ AssemblyAI   │────▶│  Groq LLM        │────▶│ Tool Router  │
│  Mic / Upload │     │ (STT)        │     │  (Intent Classify)│     │              │
└──────────────┘     └──────────────┘     └──────────────────┘     └──────┬───────┘
                                                                          │
                     ┌────────────────────────────────────────────────────┤
                     │                  │                  │              │
               ┌─────▼─────┐    ┌──────▼──────┐   ┌──────▼─────┐  ┌────▼────────┐
               │ File Ops   │    │ Code Gen     │   │ Summarizer │  │ General Chat│
               │ create_file│    │ write & save │   │            │  │             │
               └─────┬─────┘    └──────┬──────┘   └──────┬─────┘  └────┬────────┘
                     │                  │                  │              │
                     └──────────────────┴──────────────────┴──────────────┘
                                            │
                                    ┌───────▼───────┐
                                    │   Gradio UI    │
                                    │  Results Panel │
                                    └───────────────┘
```

## Features

- **Audio Input** — Record via microphone or upload `.wav`/`.mp3` files
- **Text Input** — Type commands directly for quick testing
- **Speech-to-Text** — Powered by AssemblyAI
- **Intent Classification** — Groq LLM (Llama 3.3 70B) with structured JSON output
- **Compound Commands** — A single utterance can trigger multiple actions (e.g., "Summarize this and save it to summary.txt")
- **Human-in-the-Loop** — Confirmation toggle required before any file operations execute
- **Sandboxed File Ops** — All files are created inside the `output/` directory only
- **Persistent Memory** — Conversation history is stored on disk and restored on restart
- **Graceful Degradation** — Clear error messages for failed transcription, unknown intents, etc.
- **Model Benchmarking** — Built-in benchmark script for intent accuracy and latency metrics

## Supported Intents

| Intent | Description | Example |
|--------|-------------|---------|
| `create_file` | Create an empty file or folder | "Create a file called notes.txt" |
| `write_code` | Generate code and save to a file | "Write a Python retry function" |
| `summarize` | Summarize provided text | "Summarize the following paragraph…" |
| `general_chat` | General conversation | "What is the capital of France?" |

## Setup

### 1. Clone and install dependencies

```bash
cd VoiceAIAgent
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
ASSEMBLYAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

- **AssemblyAI**: Sign up at [assemblyai.com](https://www.assemblyai.com/) — free tier provides 330 hours of transcription.
- **Groq**: Sign up at [console.groq.com](https://console.groq.com/) — free tier available.

### 3. Run the app

```bash
python app.py
```

The Gradio UI will open in your browser (typically at `http://localhost:7860`).

## Usage

1. **Record or upload** audio, or type a command in the Text tab.
2. **Toggle "Confirm file operations"** if your command involves creating files.
3. Click the corresponding **Process** button.
4. View the results: transcription, detected intent, action taken, and final output.

## Persistent Memory

- Chat memory is automatically persisted to `output/session_memory.json`.
- On app startup, history is restored and used for context-aware responses.
- Use the **Clear Persistent Memory** button in the UI to reset stored memory.

## Model Benchmarking

Run the benchmark suite:

```bash
python benchmarks/run_benchmarks.py
```

Optional flags:

```bash
python benchmarks/run_benchmarks.py --rounds 5 --save output/benchmark_report.json
```

What this measures:

- **Intent classification quality** using exact-match accuracy over `benchmarks/intent_samples.json`
- **Intent classification latency** (avg / p50 / max)
- **General chat latency** (avg / p50 / max)
- **Summarization latency** (avg / p50 / max)

## Model Choices — Rationale

### Speech-to-Text: AssemblyAI (API)

The assignment recommends a local HuggingFace model (Whisper, wav2vec). However, running Whisper locally requires a CUDA-capable GPU or takes very long on CPU. AssemblyAI was chosen because:

- **Free tier**: 330 hours of transcription at no cost.
- **No GPU required**: Runs via API, so it works on any machine.
- **High accuracy**: State-of-the-art speech recognition.
- **Simple SDK**: The `assemblyai` Python package makes integration trivial.

### LLM: Groq API (Llama 3.3 70B Versatile)

A local LLM via Ollama was the preferred option, but Ollama is not installed on this machine and the 70B-class models that are needed for reliable JSON-structured intent classification require significant RAM/VRAM. Groq was chosen because:

- **Free tier**: Generous rate limits for development.
- **Extremely fast**: Groq's LPU inference engine provides sub-second responses.
- **Llama 3.3 70B**: High-quality open model with excellent instruction following and JSON output support.
- **Already available**: The `groq` Python SDK was pre-installed.

## Project Structure

```
VoiceAIAgent/
├── app.py              # Gradio UI and orchestration
├── stt.py              # AssemblyAI speech-to-text
├── intent.py           # Groq-based intent classification
├── config.py           # API keys and constants
├── tools/
│   ├── __init__.py     # Package exports
│   ├── file_ops.py     # Sandboxed file/folder creation
│   ├── code_gen.py     # Code generation + save
│   ├── summarizer.py   # Text summarization
│   └── chat.py         # General conversation
├── output/             # All generated files go here
├── requirements.txt
├── .env.example
└── README.md
```

## Safety

All file operations are restricted to the `output/` directory. Path traversal attempts (e.g., `../../etc/passwd`) are rejected. The human-in-the-loop confirmation toggle adds an extra layer of protection — file operations are blocked unless the user explicitly enables them.
