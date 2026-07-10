# Kira — local voice assistant for macOS

A Jarvis-style voice assistant that lives in your Mac's menu bar. Say **"Кира"**
and a popover slides out from the menu bar icon: it listens, thinks (visibly),
acts on your Mac, searches the web and answers out loud — powered by a local
LLM small enough to run on any Apple Silicon Mac, including 8 GB M1 models.

*Интерфейс и голос — русскоязычные; документация на русском: [README.ru.md](README.ru.md).*

## Features

- **Wake word "Кира"** — fully offline: a lightweight VAD gates the mic,
  Whisper transcribes locally, a regex catches the name in any grammatical case
- **Local LLM with tool calling** — Qwen3-1.7B via Ollama (~44 tok/s on M4),
  streaming generation with visible reasoning steps
- **26 tools**: open/quit apps, volume, media control, weather, timers with
  voice announcements, Apple Reminders / Calendar / Notes, iMessage, clipboard,
  Spotlight file search, exchange rates, precise calculator, system status,
  lock screen, Shortcuts, and more
- **Smart web search** — DuckDuckGo (or Ollama Web Search API if
  `OLLAMA_API_KEY` is set) with snippets fed straight into the model's context;
  fresh news digests come from RSS feeds; visited sites are shown in the UI
- **Deterministic routing** — frequent intents (time, arithmetic, clipboard,
  prices, news, creative requests) bypass the LLM or force the right tool,
  eliminating small-model hallucinations where it matters most
- **Neural voice** — Microsoft edge-tts (Seraphina by default) with pipelined
  sentence-by-sentence synthesis; automatic offline fallback to Silero TTS v4,
  then to macOS `say`. Speech starts before generation finishes
- **Menu bar UI** — native-looking popover with an animated orb that breathes
  when idle, pulses with your voice, spins while thinking and changes color
  per state; live token/s counter; step timeline (thinking → searching →
  reading → answer) like the Claude app

## Architecture

```
mic 48 kHz ──► decimate to 16 kHz ──► Silero VAD ──► Whisper (MLX)
                                                        │ text
                    ┌───────────────────────────────────┘
                    ▼
      deterministic router (time / math / clipboard / prices / news / creative)
                    │ everything else
                    ▼
      Qwen3-1.7B via Ollama ──► tool calls (AppleScript, shell, web)
                    │ streaming tokens
                    ▼
      sentence splitter ──► edge-tts / Silero ──► speakers
                    ▼
      PySide6 menu bar popover (orb, steps, sources, tok/s)
```

Key design decision: **don't trust a 1.7B model with decisions it reliably
gets wrong.** Time, arithmetic and clipboard are answered by code directly;
price/news questions force a real web search before generation; creative
requests disable tools entirely so the model composes instead of retelling
search results; a "phantom action" gate catches the model *claiming* it did
something without calling a tool.

## Requirements

- Apple Silicon Mac (M1 or newer), macOS 14+
- [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/)
- ~4 GB of disk for models (LLM 1.4 GB + Whisper 0.5 GB + TTS 38 MB)

## Install & run

```bash
brew install ollama
ollama pull qwen3:1.7b

git clone https://github.com/maricusik/kira.git
cd kira
uv sync

uv run python -m kira --ui         # menu bar app with wake word
```

On first launch macOS will ask for microphone permission, and Whisper
(~500 MB) downloads from Hugging Face.

Other modes:

```bash
uv run python -m kira              # text REPL, spoken answers
uv run python -m kira --voice      # voice dialog without wake word
uv run python -m kira --wake       # wake word, terminal only
uv run python -m kira --no-speak   # silent text mode
uv run python -m kira --no-think   # disable model reasoning (faster)
uv run python -m kira --model qwen3:4b   # any Ollama model with tools
```

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `KIRA_VOICE` | `seraphina` | `seraphina` / `svetlana` / `ava` (edge-tts), `xenia` / `baya` (Silero, offline) |
| `KIRA_RATE` | `+15%` | speech tempo for edge-tts voices |
| `OLLAMA_API_KEY` | — | enables Ollama Web Search API instead of DuckDuckGo |
| `KIRA_DEBUG` | — | verbose audio pipeline logging |

## Project layout

```
kira/
  app.py     PySide6 menu bar UI: popover, orb, step timeline, tray icon
  agent.py   LLM loop: streaming, tool calling, deterministic routing, guards
  tools.py   26 tools (AppleScript, shell, web search, timers, ...)
  listen.py  mic capture, Silero VAD, Whisper transcription, wake word
  speech.py  TTS queue: edge-tts → Silero → say, sentence streaming
  main.py    CLI entry point and REPL
```

## Roadmap

- Instant wake word via Vosk keyword spotting (currently ~1.5 s, Whisper-based)
- Native SwiftUI app with Metal orb shader and MLX Swift inference
- Accessibility API actions (clicking, reading the screen)
- Login item / autostart

## License

MIT
