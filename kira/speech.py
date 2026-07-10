"""Озвучка Киры: очередь предложений, три уровня качества голоса.

1. Нейроголос Microsoft через edge-tts (svetlana/ava/seraphina) — самый
   красивый, нужен интернет;
2. Silero TTS v4 офлайн (xenia/baya/kseniya) — фолбэк без сети;
3. системный `say` — крайний случай и не-русский текст.

Голос задаётся переменной KIRA_VOICE. Очередь позволяет начинать говорить,
пока модель ещё генерирует хвост ответа: SentenceStreamer режет поток
на предложения.
"""

import concurrent.futures
import os
import queue
import re
import subprocess
import threading

VOICE = os.environ.get("KIRA_VOICE", "seraphina")
RATE = os.environ.get("KIRA_RATE", "+15%")  # темп речи edge-голосов
_EDGE_VOICES = {
    "svetlana": "ru-RU-SvetlanaNeural",
    "ava": "en-US-AvaMultilingualNeural",
    "seraphina": "de-DE-SeraphinaMultilingualNeural",
}
SAMPLE_RATE = 48_000
_MODEL_PATH = os.path.expanduser("~/.cache/kira/v4_ru.pt")
_MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
_MAX_TTS_CHARS = 800  # предел silero на один вызов

_milena_available: bool | None = None
_silero = None
_silero_failed = False

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
# всё, что плохо звучит в TTS: markdown-символы, эмодзи и прочие пиктограммы
_NOISE = re.compile(r"[*_#`>|~←-⇿☀-➿\U0001F000-\U0001FAFF]")
# конец предложения: знак + возможная закрывающая кавычка/скобка + пробел
_SENTENCE_END = re.compile(r"[.!?…]+[\"»')\]]*\s")

_q: queue.Queue[str] = queue.Queue()          # текст предложений на синтез
_audio_q: queue.Queue[tuple] = queue.Queue()  # готовый звук: синтез обгоняет плеер
_current: subprocess.Popen | None = None
_lock = threading.Lock()
_stop_playback = threading.Event()


def _get_silero():
    """Лениво загрузить Silero TTS (модель ~38 МБ, docker не нужен, CPU)."""
    global _silero, _silero_failed
    if _silero is None and not _silero_failed:
        try:
            import torch
            if not os.path.exists(_MODEL_PATH):
                import urllib.request
                os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
                urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            torch.set_num_threads(4)
            _silero = torch.package.PackageImporter(_MODEL_PATH).load_pickle(
                "tts_models", "model")
            _silero.to(torch.device("cpu"))
        except Exception:
            _silero_failed = True  # дальше работаем через say
    return _silero


def warm_up() -> None:
    """Прогреть TTS, чтобы первая фраза не запаздывала.

    Для edge-голосов греть нечего (облако); Silero грузим заранее, только
    если он выбран основным — как фолбэк он подхватится лениво.
    """
    if VOICE in _EDGE_VOICES:
        return
    model = _get_silero()
    if model is not None:
        try:
            model.apply_tts(text="привет", speaker=VOICE,
                            sample_rate=SAMPLE_RATE, put_accent=True, put_yo=True)
        except Exception:
            pass


def _has_milena() -> bool:
    """Есть ли в системе русский голос Milena (кэшируем проверку)."""
    global _milena_available
    if _milena_available is None:
        try:
            out = subprocess.run(
                ["say", "-v", "?"], capture_output=True, text=True, timeout=10
            ).stdout
            _milena_available = "Milena" in out
        except Exception:
            _milena_available = False
    return _milena_available


def clean(text: str) -> str:
    """Убрать из текста то, что TTS читает вслух как мусор."""
    text = re.sub(r"https?://\S+", "", text)  # ссылки вслух не читаем
    # у Qwen иногда проскакивают иероглифы — русский голос их не прочтёт
    text = re.sub(r"[぀-ヿ一-鿿가-힯]+", "", text)
    return re.sub(r"\s+", " ", _NOISE.sub("", text)).strip()


def _speakable_ru(text: str) -> str:
    """Подготовить текст для Silero: числа и символы — словами."""
    text = text.replace("%", " процентов").replace("$", " долларов ")
    from num2words import num2words

    def number(m: re.Match) -> str:
        s = m.group(0).replace(",", ".")
        try:
            value = float(s) if "." in s else int(s)
            return num2words(value, lang="ru")
        except Exception:
            return m.group(0)

    return re.sub(r"\d+(?:[.,]\d+)?", number, text)


def _synth_edge(text: str) -> str | None:
    """Синтез нейроголосом Microsoft в mp3-файл. None — нет сети или сбой."""
    voice = _EDGE_VOICES.get(VOICE)
    if voice is None:
        return None
    try:
        import asyncio
        import tempfile

        import edge_tts

        path = tempfile.mktemp(suffix=".mp3", prefix="kira_tts_")
        asyncio.run(edge_tts.Communicate(text, voice, rate=RATE).save(path))
        return path
    except Exception:
        return None


def _synth_silero(text: str):
    """Синтез Silero в numpy-массив. None — не вышло."""
    model = _get_silero()
    if model is None:
        return None
    try:
        import numpy as np
        chunks = [model.apply_tts(text=_speakable_ru(text[i:i + _MAX_TTS_CHARS]),
                                  speaker="xenia", sample_rate=SAMPLE_RATE,
                                  put_accent=True, put_yo=True).numpy()
                  for i in range(0, len(text), _MAX_TTS_CHARS)]
        return np.concatenate(chunks)
    except Exception:
        return None


def _speak_say(text: str) -> None:
    """Фолбэк: системный say (Milena для русского)."""
    global _current
    cmd = ["say"]
    if _CYRILLIC.search(text) and _has_milena():
        cmd += ["-v", "Milena"]
    cmd.append(text)
    with _lock:
        _current = subprocess.Popen(cmd)
    _current.wait()
    with _lock:
        _current = None


def _trim_silence(audio, sample_rate: int):
    """Обрезать тишину по краям — главный источник пауз между фразами."""
    import numpy as np
    loud = np.where(np.abs(audio) > 0.012)[0]
    if len(loud) == 0:
        return audio
    start = max(0, int(loud[0]) - sample_rate // 100)       # оставить 10 мс форы
    end = min(len(audio), int(loud[-1]) + sample_rate // 20)  # и 50 мс хвоста
    return audio[start:end]


def _mp3_to_pcm(path: str):
    """Декодировать mp3 в PCM встроенным afconvert. None — не вышло."""
    import numpy as np
    wav_path = path + ".wav"
    try:
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", path, wav_path],
                       capture_output=True, timeout=15, check=True)
        import wave
        with wave.open(wav_path) as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return _trim_silence(audio, sr), sr
    except Exception:
        return None
    finally:
        for p in (path, wav_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _synth_any(text: str) -> tuple:
    """Синтезировать фразу любым доступным способом → элемент для плеера."""
    path = _synth_edge(text)
    if path is not None:
        pcm = _mp3_to_pcm(path)
        if pcm is not None:
            return ("pcm", pcm)
    if _CYRILLIC.search(text):
        audio = _synth_silero(text)
        if audio is not None:
            return ("pcm", (_trim_silence(audio, SAMPLE_RATE), SAMPLE_RATE))
    return ("say", text)


# до трёх фраз синтезируются одновременно; очередь сохраняет порядок
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


def _synth_worker() -> None:
    while True:
        text = _q.get()
        try:
            _audio_q.put(("fut", _executor.submit(_synth_any, text)))
        except Exception:
            pass
        finally:
            _q.task_done()


def _play_pcm(audio, sr: int) -> None:
    """Блокирующее воспроизведение кусками.

    В отличие от sd.play (колбэки из Python), блокирующая запись не страдает
    от занятости интерпретатора анимацией окна — иначе звук трещит,
    когда панель открыта.
    """
    import numpy as np
    import sounddevice as sd
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    _stop_playback.clear()
    block = sr // 5  # куски по 200 мс: между ними можно оборвать озвучку
    with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as out:
        for i in range(0, len(audio), block):
            if _stop_playback.is_set():
                break
            out.write(audio[i:i + block])


def _play_worker() -> None:
    while True:
        kind, payload = _audio_q.get()
        try:
            if kind == "fut":
                kind, payload = payload.result()
            if kind == "pcm":
                _play_pcm(*payload)
            elif kind == "say":
                _speak_say(payload)
        except Exception:
            pass
        finally:
            _audio_q.task_done()


threading.Thread(target=_synth_worker, daemon=True).start()
threading.Thread(target=_play_worker, daemon=True).start()


def enqueue(text: str) -> None:
    """Добавить фразу в очередь озвучки (говорится после предыдущих)."""
    text = clean(text)
    if text:
        _q.put(text)


def speak(text: str) -> None:
    """Озвучить текст, оборвав всё, что звучало или ждало в очереди."""
    stop()
    enqueue(text)


def stop() -> None:
    """Остановить текущую озвучку и очистить обе очереди."""
    while True:
        try:
            _q.get_nowait()
            _q.task_done()
        except queue.Empty:
            break
    while True:
        try:
            kind, payload = _audio_q.get_nowait()
            if kind == "fut":
                payload.cancel()
            _audio_q.task_done()
        except queue.Empty:
            break
    _stop_playback.set()  # оборвать текущее блокирующее воспроизведение
    with _lock:
        if _current is not None and _current.poll() is None:
            _current.terminate()


def wait() -> None:
    """Дождаться, пока договорится вся очередь (чтобы микрофон не слышал Киру)."""
    _q.join()
    _audio_q.join()


class SentenceStreamer:
    """Копит поток токенов и озвучивает предложения по мере их завершения."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> None:
        self._buf += delta
        while True:
            m = _SENTENCE_END.search(self._buf)
            if not m:
                break
            sentence = self._buf[:m.end()].strip()
            self._buf = self._buf[m.end():]
            if len(sentence) > 1:
                enqueue(sentence)

    def flush(self) -> None:
        """Озвучить остаток (конец ответа без завершающего пробела)."""
        if self._buf.strip():
            enqueue(self._buf)
        self._buf = ""
