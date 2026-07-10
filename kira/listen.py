"""Голосовой ввод: запись с микрофона + распознавание через Whisper (MLX).

Пайплайн по best practices для macOS:
- микрофон открывается на НАТИВНЫХ 48 кГц (если просить у CoreAudio сразу
  16 кГц, его внутренний ресемплинг даёт искажённый звук, на котором
  Whisper галлюцинирует);
- 48к → 16к переводим прореживанием каждого третьего сэмпла: блоки
  фиксированные и кратные трём, фаза не рвётся, а речь в микрофоне мака
  не содержит энергии выше 8 кГц, чтобы алиасинг был проблемой
  (проверено: soxr.ResampleStream на этом же звуке давал кашу, а
  децимация распознаётся отлично);
- начало/конец речи определяет нейросетевой Silero VAD (кадры ровно по
  512 сэмплов @16 кГц) — в отличие от порога по громкости, он не
  срабатывает на щелчки, стук клавиатуры и прочие транзиенты;
- буфер предзаписи (~0.6 с) приклеивается к началу фразы, чтобы не
  терять первое, тихо произнесённое слово.
"""

import collections
import os
import queue
import re
from typing import Callable

import numpy as np
import sounddevice as sd

WHISPER_MODEL = "mlx-community/whisper-small-mlx"

NATIVE_SR = 48_000            # родная частота микрофонов на маках
TARGET_SR = 16_000            # частота Whisper и Silero VAD
FRAME = 512                   # кадр Silero VAD: ровно 512 сэмплов @16к (32 мс)
FRAME_SEC = FRAME / TARGET_SR
VAD_THRESHOLD = 0.5           # вероятность речи, с которой кадр считается речью
SILENCE_AFTER = 1.0           # сек тишины = конец фразы
MIN_SPEECH = 0.25             # сек речи, меньше — считаем ложным срабатыванием
MAX_PHRASE = 30.0             # сек, предохранитель
PREROLL = 0.6                 # сек звука до срабатывания VAD (тихое начало слова)

_vad = None


def _get_vad():
    global _vad
    if _vad is None:
        from pysilero_vad import SileroVoiceActivityDetector
        _vad = SileroVoiceActivityDetector()
    return _vad


def _debug(msg: str) -> None:
    if os.environ.get("KIRA_DEBUG"):
        print(f"[listen] {msg}", flush=True)


def record_phrase(on_level: Callable[[float], None] | None = None) -> np.ndarray | None:
    """Записать одну фразу с микрофона. Возвращает float32 16kHz mono или None.

    on_level, если задан, получает громкость 0..1 каждые ~30 мс — для анимации UI.
    """
    vad = _get_vad()
    vad.reset()
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())  # только кладём в очередь, не блокируем

    ring = np.empty(0, dtype=np.float32)          # накопитель для нарезки кадров VAD
    preroll: collections.deque[np.ndarray] = collections.deque(maxlen=int(PREROLL / FRAME_SEC))
    chunks: list[np.ndarray] = []
    speech_started = False
    speech_frames = 0
    silent_frames = 0
    silence_limit = int(SILENCE_AFTER / FRAME_SEC)
    max_frames = int(MAX_PHRASE / FRAME_SEC)

    with sd.InputStream(samplerate=NATIVE_SR, channels=1, dtype="float32",
                        blocksize=int(NATIVE_SR * 0.03), callback=callback):
        while True:
            ring = np.concatenate([ring, audio_q.get()[::3]])  # 48к → 16к
            done = False
            while len(ring) >= FRAME:
                frame, ring = ring[:FRAME], ring[FRAME:]
                if on_level:
                    on_level(min(1.0, float(np.sqrt(np.mean(frame ** 2))) / 0.15))
                prob = vad.process_chunk((frame * 32767).astype(np.int16).tobytes())
                speech = prob >= VAD_THRESHOLD

                if not speech_started:
                    preroll.append(frame)
                    if speech:
                        speech_started = True
                        chunks.extend(preroll)  # тихое начало фразы из буфера
                        speech_frames = 1
                    continue

                chunks.append(frame)
                speech_frames += speech
                silent_frames = 0 if speech else silent_frames + 1
                if silent_frames >= silence_limit or len(chunks) >= max_frames:
                    done = True
                    break
            if done:
                break

    if speech_frames * FRAME_SEC < MIN_SPEECH:
        _debug(f"слишком мало речи ({speech_frames * FRAME_SEC:.2f}с) — игнорирую")
        return None
    audio = np.concatenate(chunks)
    _debug(f"записана фраза {len(audio) / TARGET_SR:.1f}с (речи {speech_frames * FRAME_SEC:.1f}с)")
    return audio


# «Кира» в разных падежах + латиницей: так Whisper может расслышать имя
_WAKE_RE = re.compile(r"(?:^|[^а-яёa-z])(кир[ауыео]?|кирой|kira|kiera)(?:[^а-яёa-z]|$)",
                      re.IGNORECASE)


def extract_command(text: str) -> str | None:
    """Найти обращение к Кире во фразе.

    Возвращает: None — имени нет (игнорируем фразу); "" — только имя
    («Кира!» → надо переспросить); иначе — сам текст команды.
    """
    m = _WAKE_RE.search(text)
    if not m:
        return None
    after = text[m.end():].strip(" \t,.!?—–-:;")
    if len(after) >= 3:
        return after
    before = text[:m.start()].strip(" \t,.!?—–-:;")  # «открой сафари, Кира»
    if len(before) >= 3:
        return before
    return ""


def warm_up() -> None:
    """Прогреть Whisper и VAD, чтобы первая настоящая фраза не тормозила."""
    _get_vad()
    transcribe(np.zeros(TARGET_SR, dtype=np.float32))


def transcribe(audio: np.ndarray) -> str:
    """Распознать речь. Язык определяется автоматически."""
    import mlx_whisper  # ленивый импорт: тянет MLX, нужен только в voice-режиме

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=WHISPER_MODEL)
    return str(result.get("text", "")).strip()
