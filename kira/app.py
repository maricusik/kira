"""Интерфейс Киры: иконка в менюбаре + панель, выезжающая сверху (PySide6).

Кира живёт в строке меню macOS: там висит её сфера-иконка, которая меняет
цвет по состоянию (серая — ждёт, голубая — слушает, фиолетовая — думает,
бирюзовая — говорит). Панель с диалогом плавно выезжает из-под менюбара,
когда Кира слышит своё имя или по клику на иконку, и уезжает обратно
через несколько секунд после ответа.

В панели: живая сфера, распознанная фраза, поток размышлений модели,
ответ по мере генерации и скорость в токенах/сек. Esc — спрятать панель,
меню иконки — выход.

Запуск: uv run python -m kira --ui
"""

import math
import os
import threading

from PySide6.QtCore import (QEasingCurve, QObject, QParallelAnimationGroup,
                            QPoint, QPointF, QPropertyAnimation, QRectF, Qt,
                            QTimer, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QCursor, QIcon, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMenu,
                               QScrollArea, QSystemTrayIcon, QVBoxLayout,
                               QWidget)

from . import agent, speech

STATE_COLORS = {
    "idle": QColor(110, 130, 160),
    "listening": QColor(60, 190, 255),
    "thinking": QColor(170, 120, 255),
    "speaking": QColor(60, 230, 180),
}

STATE_LABELS = {
    "idle": "Скажите «Кира»…",
    "listening": "Слушаю",
    "thinking": "Думаю…",
    "speaking": "",
}


class Worker(QObject):
    """Фоновый цикл: ждать имя → слушать команду → LLM → озвучить."""

    state = Signal(str)        # idle / listening / thinking / speaking
    level = Signal(float)      # громкость микрофона 0..1
    user_text = Signal(str)
    step = Signal(str)         # шаг работы («ищу…», «читаю…») — лента в панели
    thinking_delta = Signal(str)  # токены размышлений модели, по мере генерации
    reply_delta = Signal(str)     # токены ответа, по мере генерации
    stats = Signal(float)      # скорость генерации, токенов/сек
    awake = Signal()           # услышала имя — развернуть панель
    done = Signal()            # ответ закончен — можно сворачиваться

    sources = Signal(str)      # какие сайты посетила (web_search / read_webpage)

    def __init__(self, model: str, think: bool = False):
        super().__init__()
        self.model = model
        self.think = think
        self.messages = agent.new_conversation()
        self._domains: list[str] = []

    def _on_tool(self, name: str, args: dict) -> None:
        """Событие «начала делать»: шаг для ленты в панели."""
        import re
        if name == "web_search":
            self.step.emit(f"🔍 ищу: «{str(args.get('query', ''))[:48]}»")
        elif name == "read_webpage":
            m = re.search(r"https?://([^/\s]+)", str(args.get("url", "")))
            site = m.group(1).removeprefix("www.") if m else "страницу"
            self.step.emit(f"🌐 читаю {site}")
        else:
            self.step.emit(f"⚙️ {name}")

    def _on_tool_result(self, name: str, args: dict, result: str) -> None:
        import re
        if name == "web_search":
            found = re.findall(r"https?://([^/\s]+)", result)
        elif name == "read_webpage":
            found = re.findall(r"https?://([^/\s]+)", str(args.get("url", "")))
        else:
            return
        for d in found:
            d = d.removeprefix("www.")
            if d not in self._domains:
                self._domains.append(d)
        if self._domains:
            self.sources.emit("🌐 " + ", ".join(self._domains[:4]))

    def run(self) -> None:
        from . import listen  # ленивый импорт MLX/Whisper

        agent.warm_up(self.model)
        listen.warm_up()
        speech.warm_up()
        print("Кира слушает — скажите «Кира». Выход: Ctrl+C в терминале.")

        while True:
            self.state.emit("idle")
            # on_wake: Vosk услышал «Кира» прямо в потоке — свечение и панель
            # включаются мгновенно, ещё до конца фразы
            audio = listen.record_phrase(on_level=self.level.emit,
                                         on_wake=self.awake.emit)
            if audio is None:
                continue
            heard = listen.transcribe(audio)
            command = listen.extract_command(heard)
            print(f"услышала: {heard!r} → команда: {command!r}", flush=True)
            if command is None:
                continue  # говорили не с Кирой

            self.awake.emit()
            if not command:  # сказали только «Кира» — переспросить
                self.state.emit("speaking")
                speech.speak("Да?")
                speech.wait()
                self.state.emit("listening")
                audio = listen.record_phrase(on_level=self.level.emit)
                command = listen.transcribe(audio).strip() if audio is not None else ""
                if not command:
                    self.done.emit()
                    continue

            self.user_text.emit(command)
            self.state.emit("thinking")
            self.messages.append({"role": "user", "content": command})
            self._domains = []

            tts = speech.SentenceStreamer()
            speaking = False

            def on_content(delta: str) -> None:
                nonlocal speaking
                if not speaking:  # пошёл ответ — озвучиваем, не дожидаясь конца
                    speaking = True
                    self.state.emit("speaking")
                self.reply_delta.emit(delta)
                tts.feed(delta)

            try:
                agent.respond(self.model, self.messages,
                              on_tool=self._on_tool,
                              on_thinking=self.thinking_delta.emit,
                              on_content=on_content,
                              on_stats=self.stats.emit,
                              on_tool_result=self._on_tool_result,
                              think=self.think)
            except Exception as e:
                error = f"Что-то пошло не так: {e}"
                self.reply_delta.emit(error)
                tts.feed(error)
            tts.flush()
            speech.wait()
            self.done.emit()


class Orb(QWidget):
    """Светящаяся сфера: дышит в покое, пульсирует от голоса, крутится в раздумьях."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(90, 90)
        self._state = "idle"
        self._phase = 0.0
        self._level = 0.0       # сглаженная громкость
        self._target = 0.0
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(40)  # 25 fps: достаточно плавно и меньше конкуренции со звуком

    def set_state(self, state: str) -> None:
        self._state = state

    def set_level(self, level: float) -> None:
        self._target = level

    def _tick(self) -> None:
        self._phase += 0.06
        if self._state == "speaking":
            # у say нет уровня звука — синтезируем «речевую» пульсацию
            self._target = 0.35 + 0.3 * abs(math.sin(self._phase * 2.1) * math.sin(self._phase * 0.7))
        elif self._state == "thinking":
            self._target = 0.15
        self._level += (self._target - self._level) * 0.25
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        color = STATE_COLORS[self._state]

        breath = 0.05 * math.sin(self._phase)           # спокойное «дыхание»
        r = min(w, h) * 0.30 * (1 + breath + 0.45 * self._level)

        # ореол — несколько тающих колец
        for i in range(4, 0, -1):
            glow = QColor(color)
            glow.setAlpha(int(14 * i * (0.6 + self._level)))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.NoPen)
            gr = r * (1 + i * 0.22)
            p.drawEllipse(int(cx - gr), int(cy - gr), int(gr * 2), int(gr * 2))

        # ядро с градиентом
        grad = QRadialGradient(cx - r * 0.25, cy - r * 0.3, r * 1.9)
        grad.setColorAt(0.0, QColor(255, 255, 255, 235))
        grad.setColorAt(0.45, color)
        grad.setColorAt(1.0, QColor(color.red() // 3, color.green() // 3, color.blue() // 3))
        p.setBrush(QBrush(grad))
        p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        # в состоянии «думаю» вокруг сферы бегает дуга
        if self._state == "thinking":
            pen = QPen(QColor(255, 255, 255, 190), 3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            ring = r * 1.35
            start = int(-self._phase * 180 * 16) % (360 * 16)
            p.drawArc(int(cx - ring), int(cy - ring), int(ring * 2), int(ring * 2),
                      start, 100 * 16)


def _detect_notch() -> tuple[float, float, float] | None:
    """Положение чёлки (notch) экрана: (левый x, правый x, высота) или None.

    macOS отдаёт безопасные зоны через NSScreen: safeAreaInsets.top > 0
    значит чёлка есть, а auxiliaryTop*Area дают области по бокам от неё.
    """
    try:
        from AppKit import NSScreen
        screen = NSScreen.screens()[0]
        top = float(screen.safeAreaInsets().top)
        if top <= 0:
            return None
        width = float(screen.frame().size.width)
        left = float(screen.auxiliaryTopLeftArea().size.width)
        right = width - float(screen.auxiliaryTopRightArea().size.width)
        return (left, right, top)
    except Exception:
        return None


class NotchIsland(QWidget):
    """«Динамический остров» вокруг чёлки, как на iPhone.

    Услышав «Кира», из чёлки плавно вырастает чёрная капсула с живой
    зелёной звуковой волной: слушает — волна дышит от голоса, думает —
    мягко перекатывается, говорит — танцует. После ответа капсула
    втягивается обратно. На маках без чёлки — та же капсула, свисающая
    из-под верхнего края экрана. Окно прозрачно для кликов.
    """

    COLOR = QColor(52, 224, 130)   # зелёный
    BARS = 21                      # столбиков в звуковой волне
    GROW_W = 130                   # на сколько капсула шире чёлки (с каждой стороны)
    GROW_H = 38                    # высота полосы с волной под чёлкой

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.Tool | Qt.WindowTransparentForInput
                            | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._notch = _detect_notch()
        self._place()
        self._phase = 0.0
        self._level = 0.0        # сглаженная громкость голоса
        self._target_level = 0.0
        self._expand = 0.0       # 0..1, раскрытие капсулы
        self._target = 0.0
        self._mode = "listening"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _place(self) -> None:
        """Окно — полоса сверху по центру экрана, вокруг чёлки."""
        screen = QApplication.primaryScreen().geometry()
        w = int((self._notch[1] - self._notch[0] if self._notch else 190)
                + self.GROW_W * 2 + 60)
        cx = int((self._notch[0] + self._notch[1]) / 2) if self._notch \
            else screen.center().x()
        self.setGeometry(cx - w // 2, screen.top(), w,
                         int((self._notch[2] if self._notch else 0) + self.GROW_H + 26))

    # интерфейс совместим с прежним ScreenGlow — сигналы не переподключаем
    def show_glow(self) -> None:
        self._target = 1.0
        if not self.isVisible():
            self._notch = _detect_notch()
            self._place()
            self.show()
        self.raise_()
        if not self._timer.isActive():
            self._timer.start(33)

    def hide_glow(self) -> None:
        self._target = 0.0

    def set_state(self, state: str) -> None:
        if state == "idle":
            self.hide_glow()
        else:
            self._mode = state

    def set_level(self, level: float) -> None:
        self._target_level = level

    def _tick(self) -> None:
        self._phase += 0.14
        self._expand += (self._target - self._expand) * 0.16
        self._level += (self._target_level - self._level) * 0.3
        if self._target == 0.0 and self._expand < 0.02:
            self._timer.stop()
            self.hide()
            return
        self.update()

    def _bar_amp(self, i: int) -> float:
        """Амплитуда столбика волны 0..1 в зависимости от состояния."""
        center = 1.0 - abs(i - (self.BARS - 1) / 2) / ((self.BARS - 1) / 2)  # купол
        if self._mode == "thinking":
            return 0.25 + 0.3 * (0.5 + 0.5 * math.sin(self._phase * 0.7 + i * 0.55))
        if self._mode == "speaking":
            wave = abs(math.sin(self._phase * 1.1 + i * 0.9)) \
                * abs(math.sin(self._phase * 0.37 + i * 0.23))
            return 0.15 + 0.85 * wave * (0.45 + 0.55 * center)
        # listening: волна дышит от громкости голоса
        wave = abs(math.sin(self._phase + i * 0.8))
        return 0.12 + (0.25 + 1.6 * self._level) * wave * center

    def paintEvent(self, event) -> None:
        if self._expand <= 0.02:
            return
        k = self._expand * self._expand * (3 - 2 * self._expand)  # smoothstep
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if self._notch:
            notch_w = self._notch[1] - self._notch[0]
            notch_h = self._notch[2]
        else:
            notch_w, notch_h = 190, 0
        pill_w = notch_w + 12 + 2 * self.GROW_W * k
        pill_h = notch_h + 2 + self.GROW_H * k
        x = (w - pill_w) / 2
        r = min(20.0, pill_h * 0.45)

        # капсула: прямые верхние углы (прижата к краю экрана), круглые нижние
        path = QPainterPath()
        path.moveTo(x, 0)
        path.lineTo(x, pill_h - r)
        path.quadTo(x, pill_h, x + r, pill_h)
        path.lineTo(x + pill_w - r, pill_h)
        path.quadTo(x + pill_w, pill_h, x + pill_w, pill_h - r)
        path.lineTo(x + pill_w, 0)
        path.closeSubpath()
        p.setPen(QPen(QColor(255, 255, 255, int(28 * k)), 1))
        p.setBrush(QBrush(QColor(8, 8, 10, int(245 * k))))
        p.drawPath(path)

        if k < 0.45:  # волну показываем, когда капсула почти раскрылась
            return
        strip_top = notch_h + 4
        strip_h = pill_h - strip_top - 7
        if strip_h < 8:
            return
        bar_zone = pill_w - 56
        gap = bar_zone / self.BARS
        bw = max(3.0, gap * 0.45)
        color = QColor(self.COLOR)
        color.setAlpha(int(255 * k))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(color))
        for i in range(self.BARS):
            amp = max(0.08, min(1.0, self._bar_amp(i)))
            bh = strip_h * amp
            bx = x + 28 + i * gap + (gap - bw) / 2
            by = strip_top + (strip_h - bh) / 2
            p.drawRoundedRect(QRectF(bx, by, bw, bh), bw / 2, bw / 2)


class KiraWindow(QWidget):
    """Безрамочная панель, выезжающая из-под менюбара поверх всех окон."""

    W, H = 380, 552
    HIDE_AFTER_MS = 8000
    SLIDE_PX = 46        # на сколько панель «спрятана» вверх перед выездом
    THINKING_TAIL = 300  # показываем последние N символов размышлений
    ARROW_W, ARROW_H = 26, 12  # «клювик», указывающий на иконку в менюбаре
    RADIUS = 14

    def __init__(self, model: str, think: bool = False):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(self.W, self.H)
        self._drag_offset = None
        self._anchor_x: int | None = None  # x иконки в менюбаре (задаёт трей)
        self._arrow_x = self.W - 60        # позиция клювика внутри окна

        self.orb = Orb()
        self.orb.setFixedHeight(116)  # компактная сфера — место отдано тексту
        self.status = QLabel(STATE_LABELS["idle"])
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: rgba(255,255,255,140); font-size: 13px;")
        self.user_label = QLabel("")
        self.user_label.setAlignment(Qt.AlignCenter)
        self.user_label.setWordWrap(True)
        self.user_label.setStyleSheet("color: rgba(255,255,255,150); font-size: 14px;")
        # лента шагов, как в приложении Claude: думаю → ищу → читаю → …
        self.steps_widget = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_widget)
        self.steps_layout.setContentsMargins(4, 0, 4, 0)
        self.steps_layout.setSpacing(3)
        self._thinking_step: QLabel | None = None
        self._thinking_text = ""
        self.reply_label = QLabel("")
        self.reply_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.reply_label.setWordWrap(True)
        self.reply_label.setStyleSheet("color: white; font-size: 15px; font-weight: 600;")
        # длинные ответы (истории, дайджесты) прокручиваются, а не распирают панель
        self.reply_scroll = QScrollArea()
        self.reply_scroll.setWidget(self.reply_label)
        self.reply_scroll.setWidgetResizable(True)
        self.reply_scroll.setFrameShape(QFrame.NoFrame)
        self.reply_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.reply_scroll.setStyleSheet("background: transparent;")
        self.reply_scroll.viewport().setStyleSheet("background: transparent;")
        self.sources_label = QLabel("")
        self.sources_label.setAlignment(Qt.AlignCenter)
        self.sources_label.setWordWrap(True)
        self.sources_label.setStyleSheet("color: rgba(140,190,255,160); font-size: 11px;")
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignRight)
        self.stats_label.setStyleSheet("color: rgba(255,255,255,90); font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16 + self.ARROW_H, 24, 14)
        layout.addWidget(self.orb)
        layout.addWidget(self.status)
        layout.addWidget(self.user_label)
        layout.addWidget(self.steps_widget)
        layout.addWidget(self.reply_scroll, stretch=1)
        layout.addWidget(self.sources_label)
        layout.addWidget(self.stats_label)

        self._reply_text = ""

        self._anim: QParallelAnimationGroup | None = None
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.slide_out)

        # фоновый рабочий поток
        self.worker = Worker(model, think)
        self.worker.state.connect(self._on_state)
        self.worker.level.connect(self.orb.set_level)
        self.worker.user_text.connect(self._on_user_text)
        self.worker.step.connect(self._add_step)
        self.worker.thinking_delta.connect(self._on_thinking_delta)
        self.worker.reply_delta.connect(self._on_reply_delta)
        self.worker.stats.connect(lambda tps: self.stats_label.setText(f"{tps:.0f} ток/с"))
        self.worker.sources.connect(self.sources_label.setText)
        self.worker.awake.connect(self.slide_in)
        self.worker.done.connect(lambda: self._hide_timer.start(self.HIDE_AFTER_MS))

        # «динамический остров» вокруг чёлки: раскрывается на имя «Кира»
        self.glow = NotchIsland()
        self.worker.awake.connect(self.glow.show_glow)
        self.worker.state.connect(self.glow.set_state)
        self.worker.level.connect(self.glow.set_level)
        self.worker.done.connect(self.glow.hide_glow)

        threading.Thread(target=self.worker.run, daemon=True).start()

    # --- контент ----------------------------------------------------------

    def _clear_steps(self) -> None:
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._thinking_step = None
        self._thinking_text = ""

    def _add_step(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: rgba(255,255,255,125); font-size: 12px;")
        self.steps_layout.addWidget(label)
        return label

    def _on_user_text(self, text: str) -> None:
        self.user_label.setText(f"«{text}»")
        self._reply_text = ""
        self._clear_steps()
        self.reply_label.setText("")
        self.sources_label.setText("")

    def _on_thinking_delta(self, delta: str) -> None:
        if self._thinking_step is None:
            self._thinking_step = self._add_step("💭 думаю…")
            self._thinking_step.setStyleSheet(
                "color: rgba(190,160,255,150); font-size: 11px; font-style: italic;")
        self._thinking_text += delta
        tail = self._thinking_text[-120:].replace("\n", " ")
        prefix = "💭 …" if len(self._thinking_text) > 120 else "💭 "
        self._thinking_step.setText(prefix + tail)

    def _on_reply_delta(self, delta: str) -> None:
        self._reply_text += delta
        self.reply_label.setText(self._reply_text)
        # автопрокрутка к свежему тексту
        bar = self.reply_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # --- выезд из-под менюбара ---------------------------------------------

    def set_anchor_x(self, x: int) -> None:
        """Трей сообщает x-координату иконки, чтобы панель выезжала под ней."""
        self._anchor_x = x

    def _anchor_pos(self) -> QPoint:
        screen = QApplication.primaryScreen().availableGeometry()
        cx = self._anchor_x if self._anchor_x is not None else screen.right() - 200
        x = max(screen.left() + 8, min(cx - self.W // 2, screen.right() - self.W - 8))
        # клювик указывает ровно на иконку, но не заезжает на скругления
        margin = self.RADIUS + self.ARROW_W
        self._arrow_x = max(margin, min(cx - x, self.W - margin))
        return QPoint(x, screen.top() + 1)  # вплотную к менюбару

    def _animate(self, end_pos: QPoint, end_opacity: float, on_done=None) -> None:
        pos_anim = QPropertyAnimation(self, b"pos")
        pos_anim.setEndValue(end_pos)
        fade = QPropertyAnimation(self, b"windowOpacity")
        fade.setEndValue(end_opacity)
        group = QParallelAnimationGroup(self)
        for a in (pos_anim, fade):
            a.setDuration(300)
            a.setEasingCurve(QEasingCurve.OutCubic)
            group.addAnimation(a)
        if on_done:
            group.finished.connect(on_done)
        if self._anim is not None:
            self._anim.stop()
        self._anim = group
        group.start()

    def slide_in(self) -> None:
        """Выехать из-под менюбара (услышала имя или клик по иконке)."""
        self._hide_timer.stop()
        target = self._anchor_pos()
        if not self.isVisible():
            self.user_label.setText("")
            self._clear_steps()
            self.reply_label.setText("")
            self.sources_label.setText("")
            # старт чуть выше конечной точки, но не за пределами экрана
            top = QApplication.primaryScreen().geometry().top()
            self.move(target.x(), max(top, target.y() - self.SLIDE_PX))
            self.setWindowOpacity(0.0)
            self.show()
        self.raise_()  # процесс фоновый, без этого окно может остаться под другими
        self._animate(target, 1.0)

    def slide_out(self) -> None:
        """Уехать обратно под менюбар."""
        self._hide_timer.stop()
        pos = self.pos()
        self._animate(QPoint(pos.x(), pos.y() - self.SLIDE_PX), 0.0, on_done=self.hide)

    def toggle(self) -> None:
        if self.isVisible() and self.windowOpacity() > 0.5:
            self.slide_out()
        else:
            self.slide_in()

    # --- отрисовка и события ---------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, ah = self.width(), self.height(), self.ARROW_H
        body = QPainterPath()
        body.addRoundedRect(0.5, ah + 0.5, w - 1.0, h - ah - 1.0, self.RADIUS, self.RADIUS)
        beak = QPainterPath()  # стрелка к иконке в менюбаре, как у нативных поповеров
        ax = self._arrow_x
        beak.moveTo(ax - self.ARROW_W / 2, ah + 1.0)
        beak.quadTo(ax - 4, ah - 6, ax, 0.5)
        beak.quadTo(ax + 4, ah - 6, ax + self.ARROW_W / 2, ah + 1.0)
        beak.closeSubpath()
        shape = body.united(beak)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.setBrush(QBrush(QColor(28, 28, 33, 228)))
        p.drawPath(shape)

    def _on_state(self, state: str) -> None:
        self.orb.set_state(state)
        label = STATE_LABELS.get(state, "")
        if label:
            self.status.setText(label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        quit_action = QAction("Выйти", menu)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)
        menu.exec(event.globalPos())

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.slide_out()


def _orb_icon(color: QColor, size: int = 44) -> QIcon:
    """Нарисовать сферу-иконку для менюбара в цвете состояния."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    r = size * 0.42
    cx = cy = size / 2
    glow = QColor(color)
    glow.setAlpha(70)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(glow))
    p.drawEllipse(int(cx - r * 1.15), int(cy - r * 1.15), int(r * 2.3), int(r * 2.3))
    grad = QRadialGradient(cx - r * 0.3, cy - r * 0.35, r * 1.9)
    grad.setColorAt(0.0, QColor(255, 255, 255, 235))
    grad.setColorAt(0.45, color)
    grad.setColorAt(1.0, QColor(color.red() // 3, color.green() // 3, color.blue() // 3))
    p.setBrush(QBrush(grad))
    p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
    p.end()
    return QIcon(pm)


def _status_item_center_x() -> int | None:
    """Координата нашей иконки в менюбаре через AppKit.

    Qt на macOS не реализует QSystemTrayIcon.geometry(), но окно статус-иконки
    (NSStatusBarWindow) принадлежит нашему процессу — берём его позицию напрямую.
    """
    try:
        from AppKit import NSApp
        for w in NSApp.windows():
            if "StatusBarWindow" in str(w.className()):
                frame = w.frame()
                return int(frame.origin.x + frame.size.width / 2)
    except Exception:
        pass
    return None


class KiraTray(QSystemTrayIcon):
    """Сфера в менюбаре: постоянное присутствие Киры + цвет состояния."""

    def __init__(self, window: KiraWindow):
        super().__init__()
        self.window = window
        self._icons = {state: _orb_icon(color) for state, color in STATE_COLORS.items()}
        self.setIcon(self._icons["idle"])
        self.setToolTip("Кира — скажите «Кира» или кликните")

        # контекстное меню НЕ ставим через setContextMenu: тогда клик по иконке
        # открывал бы меню, а не панель. Левый клик — панель, правый — меню.
        self._menu = QMenu()
        quit_action = QAction("Выйти", self._menu)
        quit_action.triggered.connect(QApplication.quit)
        self._menu.addAction(quit_action)
        self.activated.connect(self._on_activated)

        window.worker.state.connect(self.on_state)

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Context:
            self._menu.popup(QCursor.pos())
        else:  # обычный клик — открыть/спрятать панель
            self._report_anchor()
            self.window.toggle()

    def _report_anchor(self) -> None:
        x = _status_item_center_x()
        if x is None:
            rect = self.geometry()
            if rect.isValid() and rect.width() > 0:
                x = rect.center().x()
        if x is not None:
            import os
            if os.environ.get("KIRA_DEBUG"):
                print(f"[tray] иконка в менюбаре: x={x}", flush=True)
            self.window.set_anchor_x(x)

    def on_state(self, state: str) -> None:
        self.setIcon(self._icons.get(state, self._icons["idle"]))
        if state != "idle":
            self._report_anchor()  # обновляем якорь панели, пока иконка видна


def run_app(model: str = agent.DEFAULT_MODEL, think: bool = True) -> None:
    import sys
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # панель прячется, Кира живёт в менюбаре
    window = KiraWindow(model, think)
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = KiraTray(window)
        tray.show()
        window._tray = tray  # держим ссылку, иначе иконку соберёт GC
        QTimer.singleShot(1500, tray._report_anchor)  # иконка занимает место не сразу
    # показать панель при старте (когда позиция иконки уже известна)
    QTimer.singleShot(1600, window.slide_in)
    QTimer.singleShot(1600, lambda: window._hide_timer.start(KiraWindow.HIDE_AFTER_MS))
    sys.exit(app.exec())
