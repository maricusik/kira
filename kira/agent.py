"""Мозг Киры: диалог с LLM через Ollama и выполнение инструментов."""

import datetime
import re
import time
from typing import Callable

import ollama

from . import tools

DEFAULT_MODEL = "qwen3:1.7b"  # ~44 ток/с на M4; qwen3.5:2b умнее, но вдвое медленнее
MAX_TOOL_ROUNDS = 6
MAX_THINK_CHARS = 500  # «думать немного»: длиннее — обрываем и отвечаем без раздумий

SYSTEM_PROMPT = """\
You are Kira (Кира) — a friendly female local assistant running on the user's Mac.

Rules:
- Always reply in the same language the user writes in. In Russian, speak about \
yourself in the feminine gender (e.g. "я открыла", "я нашла").
- Your reply is spoken aloud via TTS, so keep it SHORT and conversational \
(1-2 sentences; news digests may be 4-5 short sentences). No markdown, \
no lists, no emojis, no code blocks. EXCEPTION: if the user explicitly asks \
for a story, tale, essay or a text of a given length — you CAN and MUST \
write the full text exactly as requested; never refuse.
- Write only in the user's language. Never mix in Chinese characters or \
other foreign scripts.
- Use tools to act on the Mac and to get real facts (time, battery, weather). \
Never guess facts a tool can provide.
- For ANY arithmetic (percentages, multiplication...) call calculate — never \
do math in your head. For currency questions use convert_currency.
- For "напомни..." use create_reminder; compute the date from Today's date \
("завтра в 10" → next day 10:00). For "поставь таймер" use set_timer.
- Your training data is outdated, so NEVER answer from memory if the question \
mentions: price/cost ("сколько стоит"), exchange rates ("курс"), news \
("новости", "что нового"), sports results, releases, or any "сейчас/сегодня" \
fact you cannot get from another tool. For these you MUST call web_search \
first and answer strictly from its results. Name the real source site from \
the results; if you did not search, do not claim any source. Never read \
URLs aloud. Use read_webpage only if snippets are not enough.
- After a tool succeeds, confirm briefly what you did. If it failed, say so honestly.
- Be warm and a little witty, but never at the cost of clarity.
- When reasoning (thinking), be extremely brief: 2-3 short sentences maximum, \
then answer.
- NEVER say you did or opened something without actually calling the tool. \
Either call the tool for real, or answer directly yourself.
- Creative or chat requests (расскажи историю, анекдот, стихи, поболтать) \
need NO tools — just answer. If a request is too vague, ask one short \
clarifying question instead of guessing.\
"""


def new_conversation() -> list:
    # без даты модель считает, что на дворе год из её тренировочных данных,
    # и подставляет его в поисковые запросы
    today = datetime.date.today().strftime("%Y-%m-%d")
    return [{"role": "system", "content": SYSTEM_PROMPT + f"\nToday's date: {today}."}]


# маленькая модель ненадёжно решает, когда нужен поиск: то отвечает из памяти,
# то отказывается. На типичные «свежие» вопросы ищем сами, до генерации.
_FORCE_SEARCH = re.compile(
    r"сколько стоит|стоимость|цена|цены|курс|биткоин|биткойн|крипто|котировк"
    r"|новост|что нового|что происходит в|кто выиграл|счёт матча|результат матча"
    r"|how much|price of|news",
    re.IGNORECASE)


_FORCE_TIME = re.compile(
    r"который час|сколько (?:сейчас )?времени|время (?:сейчас )?в\s", re.IGNORECASE)
_WEEKDAYS_RU = ["понедельник", "вторник", "среда", "четверг",
                "пятница", "суббота", "воскресенье"]


def _maybe_pretime(query: str,
                   on_tool: Callable | None,
                   on_tool_result: Callable | None) -> str | None:
    """Вопросы о времени: отвечаем сами, без модели.

    Точное время нам известно и так; модель же то выдумывает его, то
    копирует из предыдущих реплик диалога.
    """
    if not _FORCE_TIME.search(query):
        return None
    import difflib
    from zoneinfo import ZoneInfo
    m = re.search(r"\bв\s+([а-яёa-z][а-яёa-z\- ]{2,30}?)\s*\??$", query, re.IGNORECASE)
    city = m.group(1).strip() if m else ""
    if city:
        match = difflib.get_close_matches(city.lower().replace("ё", "е"),
                                          tools._CITY_TZ, n=1, cutoff=0.7)
        if not match:
            return None  # незнакомый город — пусть разбирается модель
        now = datetime.datetime.now(ZoneInfo(tools._CITY_TZ[match[0]]))
        answer = f"В {city.title()} сейчас {now:%H:%M}, {_WEEKDAYS_RU[now.weekday()]}."
    else:
        now = datetime.datetime.now()
        answer = f"Сейчас {now:%H:%M}, {_WEEKDAYS_RU[now.weekday()]}."
    if on_tool:
        on_tool("get_current_time", {"city": city})
    if on_tool_result:
        on_tool_result("get_current_time", {"city": city}, answer)
    return answer


_MATH_TRIGGER = re.compile(r"сколько будет|посчитай|вычисли|процент\w* от", re.IGNORECASE)
_WORD_OPS = [("умножить на", "*"), ("умножь на", "*"), ("умножить", "*"),
             ("разделить на", "/"), ("раздели на", "/"), ("делить на", "/"),
             ("плюс", "+"), ("минус", "-"), ("в степени", "**"), ("икс", "*")]


def _maybe_precalc(query: str,
                   on_tool: Callable | None,
                   on_tool_result: Callable | None) -> str | None:
    """Арифметика без модели: она то теряет проценты, то считает в уме."""
    if not _MATH_TRIGGER.search(query):
        return None
    expr = query.lower()
    expr = re.sub(r"(\d+[.,]?\d*)\s*процент\w*\s+от\s+(\d+[.,]?\d*)",
                  r"(\1/100)*\2", expr)
    for word, op in _WORD_OPS:
        expr = expr.replace(word, op)
    expr = re.sub(r"[^0-9.,+\-*/()% ]", " ", expr).strip()
    if not (re.search(r"\d", expr) and re.search(r"[+\-*/%]", expr)):
        return None
    result = tools.calculate(expr)
    if result.startswith("Ошибка"):
        return None
    if on_tool:
        on_tool("calculate", {"expression": expr})
    if on_tool_result:
        on_tool_result("calculate", {"expression": expr}, result)
    value = result.split("= ")[-1]
    return f"Получается {value}."


_CLIP_RE = re.compile(r"буфер\w* обмена|в буфере", re.IGNORECASE)


def _maybe_preclip(query: str,
                   on_tool: Callable | None,
                   on_tool_result: Callable | None) -> str | None:
    """«Что в буфере обмена?» — читаем сами: модель отвечает на содержимое
    буфера как на реплику пользователя."""
    if not (_CLIP_RE.search(query)
            and re.search(r"что|покажи|прочитай|скажи|узна", query, re.IGNORECASE)):
        return None
    result = tools.get_clipboard()
    if on_tool:
        on_tool("get_clipboard", {})
    if on_tool_result:
        on_tool_result("get_clipboard", {}, result)
    return result if len(result) < 400 else result[:400] + "…"


def _maybe_presearch(messages: list,
                     on_tool: Callable | None,
                     on_tool_result: Callable | None) -> bool:
    """Принудительный поиск для «свежих» вопросов. True — поиск был выполнен."""
    last = messages[-1] if messages else {}
    if not isinstance(last, dict) or last.get("role") != "user":
        return False
    query = last.get("content", "")
    if not _FORCE_SEARCH.search(query):
        return False
    if on_tool:
        on_tool("web_search", {"query": query})
    result = tools.web_search(query)
    if on_tool_result:
        on_tool_result("web_search", {"query": query}, result)
    # ссылки модели не даём — иначе она зачитывает их вслух; источники
    # пользователь и так видит отдельной строкой в интерфейсе
    clean = re.sub(r"https?://\S+", "", result)
    if re.search(r"новост|что нового|что происходит|news", query, re.IGNORECASE):
        instruction = ("Твой ответ — РОВНО 4 коротких предложения, каждое пересказывает "
                       "одну отдельную новость из результатов. Не сокращай до одного "
                       "предложения. Без ссылок, без markdown, без нумерации.")
    else:
        instruction = "Ответь кратко на их основе, назови сайт-источник по названию."
    messages.append({
        "role": "system",
        "content": "Свежие результаты веб-поиска по запросу пользователя:\n"
                   + clean + "\n" + instruction,
    })
    return True


def _fit_args(fn, args: dict) -> dict:
    """Подогнать аргументы модели под сигнатуру инструмента.

    Маленькие модели путают имена параметров (location вместо city) —
    единственный чужой аргумент отдаём первому параметру, лишние отбрасываем,
    вместо того чтобы падать с TypeError на глазах у пользователя.
    """
    import inspect
    params = list(inspect.signature(fn).parameters)
    unknown = [k for k in args if k not in params]
    if len(args) == 1 and unknown and params:
        return {params[0]: next(iter(args.values()))}
    return {k: v for k, v in args.items() if k in params}


# «я открыла поиск…» без реального вызова инструмента — частый глюк маленьких
# моделей: рассказывают о действии вместо его выполнения
_PHANTOM_ACTION = re.compile(
    r"^\s*(я\s+(открыла?|запустила?|нашла|поискала?|ищу)|открываю|запускаю"
    r"|сейчас\s+(по)?ищу|давай(те)?\s+поищ)",
    re.IGNORECASE)


class _PhantomGate:
    """Придерживает начало ответа: фантомное «я сделала X» не показываем."""

    def __init__(self, on_content, armed: bool):
        self.on_content = on_content
        self.buf = ""
        self.released = not armed
        self.suppressed = False

    def feed(self, delta: str) -> None:
        if self.suppressed:
            return
        if self.released:
            if self.on_content:
                self.on_content(delta)
            return
        self.buf += delta
        if _PHANTOM_ACTION.search(self.buf):
            self.suppressed = True
            return
        if len(self.buf) >= 50:
            self.released = True
            if self.on_content:
                self.on_content(self.buf)
            self.buf = ""

    def finish(self) -> None:
        if not self.released and not self.suppressed:
            if self.buf and self.on_content:
                self.on_content(self.buf)
            self.released = True
            self.buf = ""


# творческие запросы: инструменты отключаем совсем, иначе маленькая модель
# лезет в поиск и пересказывает найденное вместо сочинительства
_CREATIVE_RE = re.compile(
    r"сочини|придумай|расскажи(?:те)?(?:\s+(?:мне|нам))?\s+(историю|сказку|стих|рассказ|анекдот|шутку|байку)"
    r"|напиши\s+(историю|сказку|стих|рассказ|песню|поэму)"
    r"|расскажем\s+.{0,20}(историю|сказку)",
    re.IGNORECASE)


def _chat_stream(model: str, messages: list, think: bool, use_tools: bool = True):
    return ollama.chat(
        model=model,
        messages=messages,
        tools=tools.TOOLS if use_tools else None,
        think=think,
        stream=True,
        keep_alive="30m",
        # у qwen3.5 в Ollama зашит presence_penalty=1.5 — на коротких ответах
        # это даёт бессвязицу; прижимаем генерацию к более предсказуемой
        options={"temperature": 0.6, "presence_penalty": 1.0},
    )


def respond(model: str, messages: list,
            on_tool: Callable[[str, dict], None] | None = None,
            on_thinking: Callable[[str], None] | None = None,
            on_content: Callable[[str], None] | None = None,
            on_stats: Callable[[float], None] | None = None,
            on_tool_result: Callable[[str, dict, str], None] | None = None,
            think: bool = False) -> str:
    """Прогнать диалог через модель потоково, выполняя инструменты, вернуть ответ.

    Колбэки вызываются по мере генерации: on_thinking — токены размышлений,
    on_content — токены ответа (их можно сразу озвучивать), on_tool — перед
    каждым вызовом инструмента, on_tool_result — после (с результатом),
    on_stats — скорость генерации в токенах/сек. think=True включает режим
    размышлений (качественнее, но ответ начинается заметно позже).
    """
    last_user = next((m.get("content", "") for m in reversed(messages)
                      if isinstance(m, dict) and m.get("role") == "user"), "")

    direct = (_maybe_pretime(last_user, on_tool, on_tool_result)
              or _maybe_precalc(last_user, on_tool, on_tool_result)
              or _maybe_preclip(last_user, on_tool, on_tool_result))
    if direct is not None:  # частые запросы отвечаем сами — мгновенно и без ошибок
        if on_content:
            on_content(direct)
        messages.append({"role": "assistant", "content": direct})
        return direct

    creative = bool(_CREATIVE_RE.search(last_user))
    searched = False
    if creative:
        messages.append({
            "role": "system",
            "content": "Это творческий запрос. Сочини текст полностью сама — "
                       "не ищи в интернете и не пересказывай чужое. "
                       "НЕ начинай ответ со слова «Я»: начни сразу с повествования, "
                       "например «Жил-был…» или «Однажды…».",
        })
    else:
        searched = _maybe_presearch(messages, on_tool, on_tool_result)
    # после автопоиска «я нашла…» — правда, а не фантомное действие
    any_tool = searched
    nudged = False     # одёргивали ли модель за фантомное действие
    for _ in range(MAX_TOOL_ROUNDS):
        n_chunks = 0                 # один чанк стрима ≈ один токен
        started = time.monotonic()
        last_stats = started

        def track(chunk) -> None:
            nonlocal n_chunks, last_stats
            n_chunks += 1
            now = time.monotonic()
            if on_stats and now - last_stats >= 0.5:
                last_stats = now
                on_stats(n_chunks / (now - started))
            if on_stats and chunk.done and chunk.eval_count and chunk.eval_duration:
                on_stats(chunk.eval_count / (chunk.eval_duration / 1e9))

        def consume(think_mode: bool):
            """Прочитать один поток генерации. aborted=True — мысли превысили лимит."""
            content, thinking = "", ""
            tool_calls: list = []
            gate = _PhantomGate(on_content, armed=not any_tool and not nudged)
            for chunk in _chat_stream(model, messages, think=think_mode,
                                      use_tools=not creative):
                track(chunk)
                msg = chunk.message
                if msg.thinking:
                    thinking += msg.thinking
                    if on_thinking:
                        on_thinking(msg.thinking)
                    if len(thinking) > MAX_THINK_CHARS:
                        return content, thinking, tool_calls, True, gate.suppressed
                if msg.content:
                    content += msg.content
                    gate.feed(msg.content)
                if msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)
            gate.finish()
            return content, thinking, tool_calls, False, gate.suppressed

        try:
            content, thinking, tool_calls, aborted, phantom = consume(think)
        except ollama.ResponseError as e:
            if "think" not in str(e).lower():
                raise
            # модель без режима размышлений — повторяем без него
            content, thinking, tool_calls, aborted, phantom = consume(False)
        if aborted:  # задумалась слишком глубоко — обрываем и отвечаем сразу
            if on_thinking:
                on_thinking(" …ладно, к делу.")
            content, thinking, tool_calls, _, phantom = consume(False)
            thinking = ""

        assistant: dict = {"role": "assistant", "content": content}
        if thinking:
            assistant["thinking"] = thinking
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)

        if not tool_calls:
            if phantom and not nudged:
                # заявила действие, но ничего не вызвала
                nudged = True
                if creative:  # сочинять, а не искать
                    messages.append({
                        "role": "system",
                        "content": "Ничего не ищи и не говори о поиске. Сочини текст "
                                   "прямо сейчас, начни с первого предложения.",
                    })
                else:  # выполняем заявленный поиск за неё
                    if on_tool:
                        on_tool("web_search", {"query": last_user})
                    result = tools.web_search(last_user)
                    if on_tool_result:
                        on_tool_result("web_search", {"query": last_user}, result)
                    messages.append({
                        "role": "system",
                        "content": "Ты заявила поиск, но не выполнила его. Вот реальные "
                                   "результаты поиска:\n" + re.sub(r"https?://\S+", "", result)
                                   + "\nОтветь по ним кратко. Если они не по теме — ответь "
                                   "сама по существу или задай один уточняющий вопрос.",
                    })
                continue
            return content
        any_tool = True
        for call in tool_calls:
            name = call.function.name
            args = dict(call.function.arguments or {})
            if on_tool:
                on_tool(name, args)
            fn = tools.REGISTRY.get(name)
            try:
                result = fn(**_fit_args(fn, args)) if fn else \
                    f"Ошибка: неизвестный инструмент '{name}'"
            except Exception as e:  # модель могла передать кривые аргументы
                result = f"Ошибка: {e}"
            if on_tool_result:
                on_tool_result(name, args, str(result))
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})
    return "Я запуталась в инструментах, попробуй переформулировать."


def warm_up(model: str) -> None:
    """Прогреть модель, чтобы первый настоящий ответ не тормозил."""
    ollama.chat(model=model, messages=[{"role": "user", "content": "hi"}],
                think=False, keep_alive="30m", options={"num_predict": 1})
