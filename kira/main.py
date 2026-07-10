"""Кира — локальный голосовой ассистент для macOS.

Запуск:  uv run python -m kira            (текстовый ввод, голосовой ответ)
         uv run python -m kira --voice    (голосовой диалог без имени)
         uv run python -m kira --wake     (вызов по имени «Кира», в терминале)
         uv run python -m kira --ui       (интерфейс со сферой + вызов по имени)
         uv run python -m kira --no-speak (тихий текстовый режим)
"""

import argparse
import sys

import ollama

from . import agent, speech

DIM = "\033[2m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_tool(name: str, args: dict) -> None:
    pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"\n  {DIM}⚙ {name}({pretty}){RESET}")


def print_tool_result(name: str, args: dict, result: str) -> None:
    """Показать, какие сайты были использованы при поиске."""
    import re
    if name == "web_search":
        found = re.findall(r"https?://([^/\s]+)", result)
    elif name == "read_webpage":
        found = re.findall(r"https?://([^/\s]+)", str(args.get("url", "")))
    else:
        return
    domains = list(dict.fromkeys(d.removeprefix("www.") for d in found))
    if domains:
        print(f"  {DIM}🌐 {', '.join(domains[:4])}{RESET}")


class StreamPrinter:
    """Печатает поток генерации: мысли тускло, ответ обычно, озвучка по предложениям."""

    def __init__(self, speak: bool):
        self.tts = speech.SentenceStreamer() if speak else None
        self.thinking_started = False
        self.content_started = False
        self.tps = 0.0

    def on_thinking(self, delta: str) -> None:
        if not self.thinking_started:
            self.thinking_started = True
            print(f"{DIM}💭 ", end="", flush=True)
        print(delta, end="", flush=True)

    def on_content(self, delta: str) -> None:
        if not self.content_started:
            self.content_started = True
            if self.thinking_started:
                print(RESET)
                self.thinking_started = False
            print(f"{BOLD}Кира ›{RESET} ", end="", flush=True)
        print(delta, end="", flush=True)
        if self.tts:
            self.tts.feed(delta)

    def on_stats(self, tps: float) -> None:
        self.tps = tps

    def finish(self) -> None:
        if self.thinking_started:  # ответ так и не начался (только мысли)
            print(RESET)
        if self.tts:
            self.tts.flush()
        print(f"  {DIM}{self.tps:.0f} ток/с{RESET}\n" if self.tps else "")


def read_voice_input() -> str | None:
    """Записать фразу с микрофона и распознать её. None, если речи не было."""
    from . import listen

    print(f"{MAGENTA}🎤 Слушаю…{RESET}", flush=True)
    audio = listen.record_phrase()
    if audio is None:
        return None
    print(f"{DIM}…распознаю…{RESET}", flush=True)
    return listen.transcribe(audio) or None


def read_wake_input() -> str | None:
    """Ждать обращения по имени. Возвращает команду или None (имя не звучало)."""
    from . import listen

    audio = listen.record_phrase()
    if audio is None:
        return None
    command = listen.extract_command(listen.transcribe(audio))
    if command is None:
        return None
    if not command:  # сказали только «Кира» — переспросить
        speech.speak("Да?")
        speech.wait()
        print(f"{MAGENTA}🎤 Да?{RESET}", flush=True)
        audio = listen.record_phrase()
        if audio is None:
            return None
        command = listen.transcribe(audio).strip()
    return command or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Кира — локальный ассистент для macOS")
    parser.add_argument("--model", default=agent.DEFAULT_MODEL, help="модель Ollama")
    parser.add_argument("--voice", action="store_true", help="голосовой ввод (микрофон + Whisper)")
    parser.add_argument("--wake", action="store_true", help="активация по имени «Кира» (в терминале)")
    parser.add_argument("--ui", action="store_true", help="интерфейс со сферой + активация по имени")
    parser.add_argument("--no-speak", action="store_true", help="не озвучивать ответы")
    parser.add_argument("--no-think", action="store_true",
                        help="отключить размышления модели (быстрее, но проще ответы)")
    args = parser.parse_args()
    think = not args.no_think  # думает по умолчанию, но коротко (см. agent.MAX_THINK_CHARS)

    if args.ui:
        from .app import run_app
        run_app(args.model, think)
        return

    print(f"{DIM}Загружаю модель {args.model}…{RESET}", flush=True)
    try:
        agent.warm_up(args.model)
    except ollama.ResponseError as e:
        if "not found" in str(e).lower():
            sys.exit(f"Модель не найдена. Скачайте её: ollama pull {args.model}")
        sys.exit(f"Ошибка Ollama: {e}")
    except ConnectionError:
        sys.exit("Ollama не запущена. Запустите приложение Ollama или `ollama serve`.")

    listening = args.voice or args.wake
    if listening:
        from . import listen
        print(f"{DIM}Загружаю Whisper (при первом запуске скачается ~500 МБ)…{RESET}", flush=True)
        listen.warm_up()
        hint = "Скажите «Кира» и команду." if args.wake else "Говорите."
        print(f"{BOLD}Кира на связи.{RESET} {hint} Выход — Ctrl+C.\n")
    else:
        print(f"{BOLD}Кира на связи.{RESET} Пишите запрос (выход — Ctrl+C или /exit).\n")

    messages = agent.new_conversation()
    while True:
        try:
            if args.wake:
                user_input = read_wake_input()
                if not user_input:
                    continue
                print(f"{CYAN}Вы ›{RESET} {user_input}")
            elif args.voice:
                user_input = read_voice_input()
                if not user_input:
                    continue
                print(f"{CYAN}Вы ›{RESET} {user_input}")
            else:
                user_input = input(f"{CYAN}Вы ›{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nДо связи.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "выход"):
            print("До связи.")
            break

        speech.stop()  # новый вопрос обрывает прошлую озвучку
        messages.append({"role": "user", "content": user_input})
        printer = StreamPrinter(speak=not args.no_speak)
        try:
            agent.respond(args.model, messages, on_tool=print_tool,
                          on_thinking=printer.on_thinking,
                          on_content=printer.on_content,
                          on_stats=printer.on_stats,
                          on_tool_result=print_tool_result,
                          think=think)
        except Exception as e:
            print(f"Ошибка: {e}")
            continue
        printer.finish()
        if not args.no_speak and listening:
            speech.wait()  # иначе микрофон услышит саму Киру


if __name__ == "__main__":
    main()
