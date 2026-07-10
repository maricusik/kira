"""Инструменты Киры: действия на маке через osascript, open, pmset и т.д.

Каждая функция — это tool для LLM. Ollama строит JSON-схему из сигнатуры
и докстринга, поэтому докстринги написаны в формате Google style.
Все функции возвращают строку — результат, который увидит модель.
"""

import datetime
import subprocess
import urllib.parse


def _run(cmd: list[str], timeout: int = 15) -> str:
    """Запустить команду и вернуть её вывод или текст ошибки."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return f"Ошибка: команда '{cmd[0]}' не найдена"
    except subprocess.TimeoutExpired:
        return "Ошибка: команда не ответила вовремя"
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    if p.returncode != 0:
        return f"Ошибка: {err or out or 'команда завершилась с ошибкой'}"
    return out or "Готово"


def _osascript(script: str, *args: str) -> str:
    """Выполнить AppleScript. Пользовательские строки передаются через argv,
    чтобы не собирать скрипт конкатенацией (иначе кавычки в тексте всё ломают)."""
    return _run(["osascript", "-e", script, *args])


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

# города → таймзоны для мирового времени (частые запросы)
_CITY_TZ = {
    "нью-йорк": "America/New_York", "new york": "America/New_York",
    "лос-анджелес": "America/Los_Angeles", "сан-франциско": "America/Los_Angeles",
    "лондон": "Europe/London", "париж": "Europe/Paris",
    "берлин": "Europe/Berlin", "москва": "Europe/Moscow",
    "киев": "Europe/Kyiv", "тель-авив": "Asia/Jerusalem",
    "хайфа": "Asia/Jerusalem", "иерусалим": "Asia/Jerusalem",
    "дубай": "Asia/Dubai", "токио": "Asia/Tokyo",
    "пекин": "Asia/Shanghai", "сеул": "Asia/Seoul",
    "сидней": "Australia/Sydney", "минск": "Europe/Minsk",
    "алматы": "Asia/Almaty", "ташкент": "Asia/Tashkent",
}


def get_current_time(city: str = "") -> str:
    """Get the current date, time and weekday.

    Args:
        city: Optional city name for world time ("Нью-Йорк", "Tokyo").
            Empty string = local time on this Mac.
    """
    if city:
        import difflib
        from zoneinfo import ZoneInfo
        # difflib прощает падежи и опечатки: «нью-йорке» → «нью-йорк»
        match = difflib.get_close_matches(city.lower().replace("ё", "е"),
                                          _CITY_TZ, n=1, cutoff=0.7)
        if match:
            now = datetime.datetime.now(ZoneInfo(_CITY_TZ[match[0]]))
            return f"{city}: " + now.strftime("%A, %Y-%m-%d, %H:%M")
        return web_search(f"сколько сейчас времени {city}")
    return datetime.datetime.now().strftime("%A, %Y-%m-%d, %H:%M")


# модель часто передаёт русские названия — переводим в реальные имена приложений
_APP_NAMES = {
    "сафари": "Safari", "калькулятор": "Calculator", "заметки": "Notes",
    "музыка": "Music", "почта": "Mail", "календарь": "Calendar",
    "фото": "Photos", "фотографии": "Photos", "терминал": "Terminal",
    "хром": "Google Chrome", "телеграм": "Telegram", "телеграмм": "Telegram",
    "настройки": "System Settings", "системные настройки": "System Settings",
    "карты": "Maps", "сообщения": "Messages", "напоминания": "Reminders",
    "файндер": "Finder", "финдер": "Finder",
}


def _app_name(name: str) -> str:
    return _APP_NAMES.get(name.strip().lower(), name)


def open_app(app_name: str) -> str:
    """Open (launch) a macOS application by its name.

    Args:
        app_name: Application name, e.g. "Safari", "Calculator", "Telegram", "Музыка"/"Music".
    """
    return _run(["open", "-a", _app_name(app_name)])


def quit_app(app_name: str) -> str:
    """Quit (close) a running macOS application by its name.

    Args:
        app_name: Application name, e.g. "Calculator", "Safari".
    """
    return _osascript('on run argv\ntell application (item 1 of argv) to quit\nend run',
                      _app_name(app_name))


def open_url(url: str) -> str:
    """Open a URL in the default web browser.

    Args:
        url: The full URL to open, e.g. "https://example.com".
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return _run(["open", url])


# DDG-новости для ru-региона несвежие и полупустые; общие «что нового?»
# надёжнее закрывать RSS-лентами крупных изданий — они всегда актуальны
_NEWS_FEEDS = ["https://lenta.ru/rss/last24", "https://tass.ru/rss/v2.xml"]


def _rss_news(max_items: int = 8) -> list[tuple[str, str]]:
    """Свежие заголовки из RSS: [(заголовок, ссылка), …]."""
    import urllib.request
    import xml.etree.ElementTree as ET
    items: list[tuple[str, str]] = []
    per_feed = max_items // len(_NEWS_FEEDS) + 1
    for feed in _NEWS_FEEDS:
        try:
            req = urllib.request.Request(feed, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as f:
                root = ET.fromstring(f.read())
            for item in list(root.iter("item"))[:per_feed]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if title:
                    items.append((title, link))
        except Exception:
            continue
    return items[:max_items]


def web_search(query: str) -> str:
    """Search the web and return top results (titles, URLs, snippets) as text.
    Use for news, prices, facts you are unsure about, or anything current.
    Answer the user based on these results; use read_webpage for details.

    Args:
        query: What to search for, in any language.
    """
    import os

    # фирменный поиск Ollama (как «у больших») — если есть ключ с ollama.com
    if os.environ.get("OLLAMA_API_KEY"):
        try:
            import ollama
            res = ollama.web_search(query, max_results=5)
            items = [f"[{i}] {r.title}\n{r.url}\n{r.content}"
                     for i, r in enumerate(res.results, 1)]
            if items:
                return "\n\n".join(items)
        except Exception:
            pass  # падаем на DuckDuckGo

    import re
    region = "ru-ru" if re.search(r"[а-яёА-ЯЁ]", query) else "us-en"
    is_news = bool(re.search(r"новост|что нового|что происходит|news", query, re.I))

    if is_news:
        headlines = _rss_news()
        if len(headlines) >= 4:
            found = "\n\n".join(f"[{i}] {title}\n{link}"
                                for i, (title, link) in enumerate(headlines, 1))
            return found + ("\n\n(Перескажи 4 главные новости, по одному короткому "
                            "предложению каждая, без ссылок и markdown.)")

    from ddgs import DDGS
    results = []
    if is_news:  # RSS не ответил — пробуем новостной индекс DDG
        for timelimit in ("d", "w"):
            try:
                results = DDGS().news(query, region=region,
                                      timelimit=timelimit, max_results=8)
            except Exception:
                results = []
            if results:
                for r in results:
                    r.setdefault("href", r.get("url", ""))
                break
    if not results:
        try:
            results = DDGS().text(query, region=region, max_results=5)
        except Exception as e:
            return f"Ошибка поиска: {e}"
    if not results:
        return "Поиск ничего не нашёл"
    found = "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )
    # маленькие модели, получив простыню сниппетов, норовят написать доклад —
    # напоминание в конце результата держит ответ разговорным
    return found + "\n\n(Ответь пользователю кратко, 1-2 разговорных предложения, без markdown и списков.)"


def read_webpage(url: str) -> str:
    """Fetch a web page and return its readable text content.
    Use after web_search when the snippets are not enough to answer.

    Args:
        url: Full URL of the page to read.
    """
    try:
        import trafilatura
        html = trafilatura.fetch_url(url)
        if not html:
            return "Ошибка: страница не загрузилась"
        text = trafilatura.extract(html, include_comments=False) or ""
    except Exception as e:
        return f"Ошибка: {e}"
    if not text.strip():
        return "Ошибка: не удалось извлечь текст со страницы"
    return text[:3000]


def get_battery() -> str:
    """Get the Mac's current battery charge level and charging status."""
    return _run(["pmset", "-g", "batt"])


def set_volume(level: int) -> str:
    """Set the system output volume.

    Args:
        level: Volume from 0 (mute) to 100 (max).
    """
    level = max(0, min(100, int(float(level))))
    result = _osascript(f"set volume output volume {level}")
    return f"Громкость установлена на {level}%" if result == "Готово" else result


def media_control(action: str) -> str:
    """Control music playback in Spotify or Apple Music (whichever is running).

    Args:
        action: One of "play_pause", "next", "previous".
    """
    if _run(["pgrep", "-x", "Spotify"]).isdigit():
        player = "Spotify"
    elif _run(["pgrep", "-x", "Music"]).isdigit():
        player = "Music"
    else:
        return "Ни Spotify, ни Music сейчас не запущены"
    commands = {"play_pause": "playpause", "next": "next track", "previous": "previous track"}
    cmd = commands.get(action)
    if not cmd:
        return f"Неизвестное действие '{action}', доступны: {', '.join(commands)}"
    result = _osascript(f'tell application "{player}" to {cmd}')
    return f"{player}: {action}" if result == "Готово" else result


def create_note(title: str, body: str) -> str:
    """Create a new note in the Apple Notes app.

    Args:
        title: Note title.
        body: Note text content.
    """
    script = (
        "on run argv\n"
        'tell application "Notes" to tell account 1 to make new note at folder 1 '
        "with properties {name:item 1 of argv, body:item 2 of argv}\n"
        "end run"
    )
    result = _osascript(script, title, body)
    return f"Заметка «{title}» создана" if not result.startswith("Ошибка") else result


def get_weather(city: str = "") -> str:
    """Get the current weather. Uses wttr.in (needs internet).

    Args:
        city: City name in NOMINATIVE case ("Хайфа", not "Хайфе") or in English.
            Empty string = detect by IP (current location).
    """
    import re

    def fetch(name: str) -> str:
        url = f"https://wttr.in/{urllib.parse.quote(name)}?format=3&m"
        return _run(["curl", "-s", "--max-time", "10", url])

    def ok(result: str) -> bool:
        low = result.lower()
        return not (result.startswith("Ошибка") or "not found" in low
                    or "error" in low or "unknown" in low)

    result = fetch(city)
    if ok(result):
        return result
    # модель часто передаёт город в падеже («в Хайфе» → «хайфе») —
    # пробуем вернуть именительный заменой окончания
    if re.search(r"[еиую]$", city, re.IGNORECASE):
        alt = re.sub(r"[еиую]$", "а", city)
        result = fetch(alt)
        if ok(result):
            return result
    # совсем не нашли — отдаём модели результаты веб-поиска
    return web_search(f"погода {city} сейчас температура")


def run_shortcut(name: str) -> str:
    """Run a macOS Shortcuts automation by its exact name.

    Args:
        name: The exact name of the shortcut as it appears in the Shortcuts app.
    """
    result = _run(["shortcuts", "run", name])
    return f"Команда «{name}» выполнена" if result == "Готово" else result


# --------------------------------------------------------------------------
# Вычисления и конвертация
# --------------------------------------------------------------------------

def calculate(expression: str) -> str:
    """Calculate a math expression EXACTLY. Use for ANY arithmetic: percentages,
    multiplication, division etc. Never do math in your head.

    Args:
        expression: Math expression, e.g. "234*17", "2400*0.15", "(5+3)/2".
    """
    import ast
    import operator as op

    import re
    expression = (expression.replace("×", "*").replace("÷", "/")
                  .replace("^", "**").replace(",", "."))
    # проценты: «15% of 2400» / «15% от 2400» → (15/100)*2400, «15%» → (15/100)
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(?:of|от)\s*", r"(\1/100)*", expression)
    expression = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expression)
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
           ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
           ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("недопустимое выражение")

    try:
        result = ev(ast.parse(expression, mode="eval").body)
    except Exception:
        return f"Ошибка: не смогла вычислить «{expression}»"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {round(result, 6) if isinstance(result, float) else result}"


_RATES_CACHE: dict = {}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies at the current exchange rate.

    Args:
        amount: Amount to convert, e.g. 100.
        from_currency: ISO code: USD, EUR, RUB, ILS, GBP...
        to_currency: ISO code to convert into.
    """
    import json
    import time
    import urllib.request

    src, dst = from_currency.upper().strip(), to_currency.upper().strip()
    try:
        cached = _RATES_CACHE.get(src)
        if cached is None or time.time() - cached[0] > 3600:
            with urllib.request.urlopen(
                    f"https://open.er-api.com/v6/latest/{src}", timeout=10) as f:
                rates = json.load(f)["rates"]
            _RATES_CACHE[src] = (time.time(), rates)
        else:
            rates = cached[1]
        rate = rates.get(dst)
        if rate is None:
            return f"Ошибка: не знаю валюту {dst}"
        value = float(amount) * rate
        return f"{amount} {src} = {value:,.2f} {dst} (курс {rate:.4f})".replace(",", " ")
    except Exception as e:
        return f"Ошибка курса валют: {e}"


# --------------------------------------------------------------------------
# Таймеры (живут, пока запущена Кира)
# --------------------------------------------------------------------------

_TIMERS: list[dict] = []


def set_timer(minutes: float, label: str = "") -> str:
    """Set a countdown timer. When it fires, Kira announces it aloud
    and shows a notification.

    Args:
        minutes: Duration in minutes (can be fractional, e.g. 0.5 = 30 sec).
        label: What the timer is for, e.g. "пицца" (optional).
    """
    import threading
    import time

    minutes = float(minutes)
    if not 0 < minutes <= 24 * 60:
        return "Ошибка: длительность должна быть от секунд до суток"
    name = label or "таймер"

    def fire():
        _TIMERS[:] = [t for t in _TIMERS if t["name"] != name or t["end"] > time.time()]
        _osascript(f'display notification "{name}" with title "Кира: время вышло!"')
        _run(["afplay", "/System/Library/Sounds/Glass.aiff"])
        try:
            from . import speech
            speech.enqueue(f"Время вышло! {label}" if label else "Таймер сработал!")
        except Exception:
            pass

    timer = threading.Timer(minutes * 60, fire)
    timer.daemon = True
    timer.start()
    _TIMERS.append({"name": name, "end": time.time() + minutes * 60})
    pretty = f"{minutes:g} мин" if minutes >= 1 else f"{int(minutes * 60)} сек"
    return f"Таймер «{name}» поставлен на {pretty}"


def list_timers() -> str:
    """List currently running timers and how much time is left on each."""
    import time

    active = [t for t in _TIMERS if t["end"] > time.time()]
    if not active:
        return "Активных таймеров нет"
    return "; ".join(f"«{t['name']}»: осталось {int((t['end'] - time.time()) / 60)} мин "
                     f"{int((t['end'] - time.time()) % 60)} сек" for t in active)


# --------------------------------------------------------------------------
# Напоминания, календарь, сообщения
# --------------------------------------------------------------------------

def create_reminder(title: str, due: str = "") -> str:
    """Create a reminder in the Apple Reminders app.

    Args:
        title: Reminder text, e.g. "купить хлеб".
        due: Optional due date-time as "YYYY-MM-DD HH:MM" (24h). Empty = no date.
    """
    when = None
    if due:
        try:
            when = datetime.datetime.strptime(due.strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            return "Ошибка: дата должна быть в формате YYYY-MM-DD HH:MM"
    if when is None:
        result = _osascript(
            'on run argv\ntell application "Reminders" to make new reminder '
            "with properties {name:item 1 of argv}\nend run", title)
        return f"Напоминание «{title}» создано" if not result.startswith("Ошибка") else result
    script = """on run argv
set d to current date
set year of d to (item 2 of argv as integer)
set month of d to (item 3 of argv as integer)
set day of d to (item 4 of argv as integer)
set hours of d to (item 5 of argv as integer)
set minutes of d to (item 6 of argv as integer)
set seconds of d to 0
tell application "Reminders" to make new reminder with properties {name:(item 1 of argv), due date:d, remind me date:d}
end run"""
    result = _osascript(script, title, str(when.year), str(when.month),
                        str(when.day), str(when.hour), str(when.minute))
    if result.startswith("Ошибка"):
        return result
    return f"Напоминание «{title}» на {when:%d.%m в %H:%M} создано"


def create_calendar_event(title: str, start: str, duration_minutes: int = 60) -> str:
    """Create an event in the Apple Calendar app.

    Args:
        title: Event name, e.g. "встреча с Андреем".
        start: Start date-time as "YYYY-MM-DD HH:MM" (24h).
        duration_minutes: Event length in minutes, default 60.
    """
    try:
        when = datetime.datetime.strptime(start.strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        return "Ошибка: дата должна быть в формате YYYY-MM-DD HH:MM"
    script = """on run argv
set d to current date
set year of d to (item 2 of argv as integer)
set month of d to (item 3 of argv as integer)
set day of d to (item 4 of argv as integer)
set hours of d to (item 5 of argv as integer)
set minutes of d to (item 6 of argv as integer)
set seconds of d to 0
set e to d + (item 7 of argv as integer) * minutes
tell application "Calendar" to tell (first calendar whose writable is true) to make new event with properties {summary:(item 1 of argv), start date:d, end date:e}
end run"""
    result = _osascript(script, title, str(when.year), str(when.month), str(when.day),
                        str(when.hour), str(when.minute), str(int(duration_minutes)))
    if result.startswith("Ошибка"):
        return result
    return f"Событие «{title}» {when:%d.%m в %H:%M} добавлено в календарь"


def get_today_events() -> str:
    """List today's events from the Apple Calendar app."""
    script = """set d1 to current date
set time of d1 to 0
set d2 to d1 + 1 * days
set out to ""
tell application "Calendar"
  repeat with c in calendars
    repeat with e in (every event of c whose start date is greater than or equal to d1 and start date is less than d2)
      set out to out & (time string of (start date of e)) & " — " & (summary of e) & linefeed
    end repeat
  end repeat
end tell
return out"""
    result = _run(["osascript", "-e", script], timeout=30)
    if result == "Готово" or not result.strip():
        return "На сегодня событий в календаре нет"
    return result


def send_imessage(recipient: str, text: str) -> str:
    """Send an iMessage via the Messages app. Use ONLY when the user explicitly
    asks to send a message and both recipient and text are clear.

    Args:
        recipient: Phone number or Apple ID email of the recipient.
        text: Message text to send.
    """
    script = """on run argv
tell application "Messages"
  set svc to 1st account whose service type = iMessage
  send (item 2 of argv) to participant (item 1 of argv) of svc
end tell
end run"""
    result = _osascript(script, recipient, text)
    return f"Сообщение отправлено: {recipient}" if not result.startswith("Ошибка") else result


# --------------------------------------------------------------------------
# Буфер обмена, файлы, система
# --------------------------------------------------------------------------

def get_clipboard() -> str:
    """Read the current text content of the clipboard."""
    result = _run(["pbpaste"])
    if result == "Готово":
        return "Буфер обмена пуст"
    # ярлык обязателен: без него модель принимает содержимое буфера
    # за реплику пользователя и отвечает на неё
    return f"Содержимое буфера обмена: «{result[:1500]}»"


def set_clipboard(text: str) -> str:
    """Put text into the clipboard.

    Args:
        text: Text to copy.
    """
    try:
        subprocess.run(["pbcopy"], input=text, text=True, timeout=10)
        return "Скопировано в буфер обмена"
    except Exception as e:
        return f"Ошибка: {e}"


def find_files(query: str) -> str:
    """Find files on this Mac by name using Spotlight.

    Args:
        query: Part of the file name, e.g. "отчёт" or "presentation.pdf".
    """
    result = _run(["mdfind", "-name", query], timeout=20)
    if result.startswith("Ошибка") or not result.strip() or result == "Готово":
        return f"Файлы с именем «{query}» не найдены"
    lines = result.splitlines()[:5]
    return "\n".join(lines) + (f"\n…и ещё {len(result.splitlines()) - 5}"
                               if len(result.splitlines()) > 5 else "")


def open_path(path: str) -> str:
    """Open a file or folder in Finder / default app.

    Args:
        path: Absolute path to the file or folder.
    """
    import os
    path = os.path.expanduser(path.strip())
    if not os.path.exists(path):
        return f"Ошибка: путь не существует: {path}"
    return _run(["open", path])


def system_status() -> str:
    """Get Mac system status: free disk space, memory, uptime and battery."""
    import os
    disk = _run(["df", "-h", "/"]).splitlines()
    disk_line = disk[1].split() if len(disk) > 1 else []
    free = disk_line[3] if len(disk_line) > 3 else "?"
    mem_total = int(_run(["sysctl", "-n", "hw.memsize"]) or 0) // (1024 ** 3)
    uptime = _run(["uptime"])
    battery = _run(["pmset", "-g", "batt"]).splitlines()
    batt = battery[1].split("\t")[1].split(";")[0] if len(battery) > 1 else "?"
    return (f"Свободно на диске: {free}; память: {mem_total} ГБ всего; "
            f"батарея: {batt}; {uptime}")


def lock_screen() -> str:
    """Lock the screen (turn off the display, password required to resume)."""
    return _run(["pmset", "displaysleepnow"])


TOOLS = [
    get_current_time,
    open_app,
    quit_app,
    open_url,
    web_search,
    read_webpage,
    get_battery,
    set_volume,
    media_control,
    create_note,
    get_weather,
    run_shortcut,
    calculate,
    convert_currency,
    set_timer,
    list_timers,
    create_reminder,
    create_calendar_event,
    get_today_events,
    send_imessage,
    get_clipboard,
    set_clipboard,
    find_files,
    open_path,
    system_status,
    lock_screen,
]

REGISTRY = {fn.__name__: fn for fn in TOOLS}
