# 🎬 Automated Video Translation Pipeline

This project is a modular and scalable pipeline designed to automate the process of transcribing and translating video content. It leverages **Whisper.cpp** for high-performance local transcription and integrates **LLM Providers** via UI automation to ensure high-quality, context-aware translations.

## 🌟 Key Features

* **Audio Extraction**: Automated conversion of video tracks to 16kHz WAV format.
* **Local Transcription**: Uses `whisper.cpp` for fast, offline, and private speech-to-text.
* **SRT Optimization**: Advanced logic to merge identical consecutive segments and eliminate "flickering" effects.
* **Translation Engines**:
  * **Legacy Engine**: Rapid translation using Google Translate APIs with line-level caching and custom technical dictionaries.
  * **LLM Engine (Local)**: OpenAI-compatible local endpoint support (e.g., llama.cpp).
  * **LLM Engine (UI)**: High-quality translation via Large Language Models (e.g., Copilot via UI Automation).
  * **Hybrid Engine** *(new)*: Triple-source arbitration — S1 (source SRT) + L1 (literal Google Translate) + Mt (LLM draft) → HybridRefiner for professional post-editing with incremental re-run support.
* **Resilient Workspace**: Intelligent directory mirroring that allows resuming the process at any stage.

---

## 📂 Project Structure

```text
project_root/
├── .vscode/
│   └── tasks.json            # VS Code tasks (ruff, pylint, coverage)
├── configs/
│   ├── settings.json          # Global configuration (paths, languages, etc.)
│   ├── system_prompt.txt      # System prompt template for LLM translation
│   └── refinement_protocol.txt # Hybrid refiner arbitration instructions
├── src/
│   ├── main.py                # Pipeline orchestrator (4-step pipeline)
│   ├── modules/
│   │   ├── translator.py      # BaseTranslator — abstract class (skip logic, standardize)
│   │   ├── legacy_translator.py # Google Translate + dictionary + line-cache
│   │   ├── llm_translator.py  # Chunked LLM translation
│   │   ├── extractor.py       # Audio extraction (FFmpeg segmentation)
│   │   ├── transcriber.py     # Whisper.cpp subprocess wrapper
│   │   ├── srt_optimizer.py   # SRT structure cleaning (Step 3)
│   │   ├── strategies/
│   │   │   └── hybrid_refiner.py # Triple-source arbitration (S1+L1+Mt)
│   │   └── providers/
│   │       ├── base_provider.py  # LLMProvider ABC + LLMProviderError
│   │       ├── llama_provider.py # Local llama.cpp HTTP provider
│   │       └── copilot_ui.py     # Browser-based UI automation provider
│   └── utils/
│       ├── srt_handler.py     # SRT parsing, rendering, alignment, hashing
│       └── file_handler.py    # DirectoryMirrorTask base class
├── tests/
│   ├── test_srt_handler.py    # 84 tests (Unicode, CJK, merge, standardize)
│   ├── test_file_handler.py
│   ├── test_llm_provider.py
│   └── test_llm_translator.py
├── pyproject.toml             # Build config + ruff + pylint + coverage settings
├── requirements.txt
└── readme.md
```

---

## ⚙️ Configuration (`settings.json`)

Configure your environment before launching the pipeline:

```json
{
  "base_working_dir": "./workspace",
  "whisper": {
    "bin_path": "C:/path/to/whisper/whisper-cli.exe",
    "model_path": "C:/path/to/whisper/models/ggml-large-v3.bin",
    "lang": "auto"
  },
  "llm_config": {
    "source_lang": "English",
    "target_lang": "French",
    "chunk_size": 20
  },
  "technical_dictionary": {
    "example term": "target term",
    "industry keyword": "mot-clé métier"
  }
}

```

---

## 🚀 Usage

### 1. Full Execution

Processes everything from the source video to the final translated subtitle:

```bash
python src/main.py --input "./source_videos" --output "./results" --mode full --engine llm-ui

```

Available `--engine` values:

* `legacy` (Google Translate + technical dictionary)
* `llm-local` (local OpenAI-compatible endpoint, e.g. llama.cpp)
* `llm-ui` (Copilot UI automation)
* `hybrid` (triple-source arbitration: Legacy L1 + LLM Mt → Refiner)

### 2. Step-by-Step Execution

You can isolate specific stages using the `--mode` flag:

* `extract`: Extract audio from video.
* `transcribe`: Generate raw subtitles from audio.
* `optimize`: Clean and merge subtitle blocks (produces S1).
* `translate`: Translate subtitles only (requires existing SRT files in `3_clean_srt/`).

### 3. Hybrid Engine Details

The `hybrid` engine executes three sub-steps automatically:

1. **L1 generation** — Legacy translator produces a literal reference.
2. **Mt generation** — LLM translator produces a stylistic draft.
3. **Arbitration** — `HybridRefiner` aligns L1 and Mt against S1 timestamps and sends windowed prompts to the LLM for final post-editing.

On **re-runs**, the refiner detects which blocks are already correctly translated and only re-processes problematic windows (missing, empty, or untranslated blocks), saving significant LLM cost.

---

## 🛠️ Requirements & Setup

Instead of installing everything from your global environment, it is highly recommended to use a clean virtual environment to avoid conflicts:

1. **Create and activate a virtual environment**:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

```


2. **Install core dependencies**:
```bash
pip install -r requirements.txt

```

3. **Install development tools** (linting, coverage):
```bash
pip install ruff pylint pytest-cov

```

4. **Run tests**:

```bash
python -m pytest tests/ -v

```

5. **Run tests with coverage**:

```bash
python -m pytest --cov=src --cov-config=pyproject.toml --cov-report=term-missing tests/ -v

```

6. **Linting**:

```bash
# Ruff (fast linter + formatter)
python -m ruff check src/ tests/
python -m ruff format src/ tests/

# Pylint (deep analysis)
python -m pylint src/modules/ src/utils/ src/main.py --rcfile=pyproject.toml

```

7. **VS Code Tasks** (Ctrl+Shift+P → "Tasks: Run Task"):
   * `Ruff: lint` / `Ruff: lint + fix` / `Ruff: format`
   * `Pylint: full analysis`
   * `Pytest: run all` (default test task)
   * `Coverage: run + report` / `Coverage: open HTML report`
   * `Quality: full check (ruff + pylint + coverage)` — runs all three sequentially

8. **External Requirements**:
* **FFmpeg**: Must be in your System PATH.
* **Whisper.cpp**: Compiled executable (e.g., `main.exe`).
* **Microsoft Edge**: Required for the default `CopilotUIProvider`.

---
### ⚙️ Full Configuration Reference (`settings.json`)

Here is the exhaustive list of parameters available in your configuration file:

| Category | Parameter | Description |
| --- | --- | --- |
| **Global** | `base_working_dir` | Root folder where the `internals/` workspace will be created. |
| **Whisper** | `bin_path` | Absolute path to the `whisper.cpp` executable. |
|  | `model_path` | Path to the `.bin` model file (e.g., `ggml-medium.bin`). |
|  | `lang` | Source language code (`ru`, `en`, `fr` or `auto`). |
| **LLM Engine** | `source_lang` | Full name of the source language for the prompt (e.g., "Russian"). |
|  | `target_lang` | Full name of the target language (e.g., "French"). |
|  | `chunk_size` | Number of SRT blocks sent in a single prompt (default: 30). |
|  | `chunk_delay` | Seconds between LLM API calls (rate-limit protection, default: 1.0). |
|  | `prompt_file` | Path to the system prompt template text file used by the LLM translator. |
| **Legacy** | `source_lang` / `target_lang` | ISO language codes for Google Translate (e.g., `en`, `fr`). |
|  | `max_chars_batch` | Character limit for Google Translate batches. |
|  | `retry_delay` | Seconds to wait between translation retries. |
|  | `max_retries` | Maximum number of retry attempts when rate-limited (HTTP 429). |
|  | `cache_file` | Path to the JSON cache used for line-level translation reuse. |
| **Hybrid** | `refinement_protocol_file` | Path to the arbitration system prompt (default: `configs/refinement_protocol.txt`). |
| **Context** | `technical_dictionary` | Key-value pairs of terms to ensure consistent translation. |

---

### 🤖 Expanding LLM Providers

The architecture is designed to be **Provider-Agnostic**. While `CopilotUIProvider` is the default, you can easily switch to other models.

#### 1. Switching to other Web LLMs

The current UI Automation logic targets window titles and UI elements. You can create new providers for **ChatGPT**, **Claude**, or **Gemini** by inheriting from the `LLMProvider` base class and adapting the `pywinauto` selectors to target those browsers.

#### 2. Using a Local LLM (LM Studio / Ollama)

For total privacy and no UI interaction, you can implement a local provider. Local LLMs usually provide an OpenAI-compatible API.

**Example of a Local Provider (`src/modules/providers/local_llm.py`)**:

```python
import requests
from .base_provider import LLMProvider

class LocalLLMProvider(LLMProvider):
    def __init__(self, api_url="http://localhost:1234/v1/chat/completions"):
        self.api_url = api_url

  def ask(self, content: str, prompt: str) -> str:
        response = requests.post(self.api_url, json={
      "messages": [
        {"role": "system", "content": content},
        {"role": "user", "content": prompt}
      ],
            "temperature": 0.1
        })
        return response.json()['choices'][0]['message']['content']

```

#### 3. Integration

To use your new provider, simply update the factory in `main.py`:

```python
# Instead of CopilotUIProvider
provider = LocalLLMProvider(api_url="http://localhost:11434/v1/chat/completions")

```

---

## ⚠️ Important Notes

* **UI Focus**: When using the LLM engine, the script interacts with your browser. Ensure the target window is visible and avoid manual input during the automated paste/send cycles.
* **UTF-8 Standard**: All files are processed using UTF-8 encoding. If you encounter character issues on Windows, the script includes a "replace" safety mechanism to prevent crashes.
* **Data Integrity**: The `internals/` folder preserves intermediate files. If a translation is interrupted, you can resume without losing transcription progress.

---
## 🚀 Future Improvements

The project is designed with a modular architecture, allowing for several high-impact evolutions:

#### 1. Local Voice Synthesis (TTS)

* **Feature**: Generate a localized audio track directly from the translated SRT files.
* **Tech**: Integration of local TTS engines like **Coqui TTS** or **Piper** to produce high-quality, natural-sounding voiceovers without cloud costs.
* **Goal**: Create fully dubbed videos automatically.

#### 2. Advanced Validation & Feedback Loop

* **Structural Integrity**: Add a post-translation validation layer to ensure the LLM output strictly matches the source timestamps and block counts.
* **Auto-Correction**: If a mismatch is detected (e.g., the LLM merged two blocks or skipped a timestamp), the system could automatically re-send only the failed chunk.
* **Consistency Check**: Verify that the number of lines in the translated block matches the original to avoid subtitle desynchronization.

#### 3. UI Automation Robustness

* **Headless Support**: Explore `playwright` or `selenium-wire` to interact with LLM web interfaces in the background, reducing the need for an active window.
* **Dynamic Selectors**: Implement smarter UI element detection in `pywinauto` to make the providers more resilient to browser updates or UI changes on Bing/Copilot.
* **Multi-Provider Failover**: Automatically switch from one LLM provider (e.g., Copilot) to another (e.g., a local Llama 3 instance) if a rate limit or error is detected.

#### 4. Advanced Prompt Engineering

* **Few-Shot Prompting**: Include example translation pairs (source -> target) directly in the system prompt to improve the "style" and "tone" of the translation.
* **Dynamic Context Injection**: Automatically inject relevant parts of the technical dictionary based on the words detected in the current chunk.
* **Chain-of-Thought**: Force the model to explain its choice for difficult technical terms before providing the final SRT block to increase accuracy.

#### 5. User Interface (GUI)

* **Dashboard**: A web-based dashboard (using **Streamlit** or **Flask**) to monitor transcription progress in real-time, edit the technical dictionary on the fly, and manually correct specific translation blocks.

---
