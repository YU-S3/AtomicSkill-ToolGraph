"""全局监控进度条（实验运行 / 冻结评估共用，零第三方依赖）。

设计（与设计文档 §62 监控仪表一致）：
- 顶层单元 = 条件（baseline 子进程 / ours 条件 / 冻结条件一律等同）
- 条件内单元 = 任务（基线子进程通过解析其 stdout 的 ``[i/n]`` 行获得）
- 总体百分比 = (已完成条件 + 当前条件任务进度) / 总条件数

tty 下用 ``\\r`` 单行刷新（最多每秒一次）；非 tty（日志重定向）下每更新一行。
"""

from __future__ import annotations

import sys
import time

_BAR_WIDTH = 30

_ANCHOR_NOTES = ("开始", "完成", "全部完成")


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)


class ProgressMonitor:
    """全局进度条：条件级 + 任务级，跨条件类型统一呈现。"""

    def __init__(self, stream=None, enabled: bool = True) -> None:
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())
        # 控制台编码不兼容块状字符时（如 Windows GBK）退化为 ASCII
        try:
            "█░".encode(getattr(self.stream, "encoding", None) or "utf-8")
            self._chars = ("█", "░")
        except (UnicodeEncodeError, LookupError):
            self._chars = ("#", "-")
        self.total_conditions = 1
        self.completed_conditions = 0
        self.condition_name = ""
        self.task_total = 0
        self.task_done = 0
        self.t0 = time.time()
        self._last_render = 0.0

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def set_total_conditions(self, n: int) -> None:
        self.total_conditions = max(1, int(n))

    def condition_start(self, name: str, task_total: int = 0) -> None:
        self.condition_name = name
        self.task_total = max(0, int(task_total))
        self.task_done = 0
        self._render("开始")

    def task_update(self, done: int, note: str = "") -> None:
        self.task_done = max(self.task_done, int(done))
        self._render(note)

    def condition_finish(self) -> None:
        self.task_done = self.task_total
        self._render("完成")
        self.completed_conditions += 1

    def finish(self) -> None:
        if self.enabled:
            self._render("全部完成")
            self.stream.write("\n")
            self.stream.flush()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @property
    def fraction(self) -> float:
        if self.task_total > 0:
            frac = (self.completed_conditions
                    + min(self.task_done, self.task_total) / self.task_total
                    ) / self.total_conditions
        else:
            frac = self.completed_conditions / self.total_conditions
        return min(1.0, max(0.0, frac))

    def _render(self, note: str = "") -> None:
        if not self.enabled:
            return
        now = time.time()
        anchor = note in _ANCHOR_NOTES
        if self._tty and not anchor and now - self._last_render < 1.0:
            return
        self._last_render = now

        frac = self.fraction
        filled = int(_BAR_WIDTH * frac)
        fill_char, empty_char = self._chars
        bar = fill_char * filled + empty_char * (_BAR_WIDTH - filled)
        elapsed = now - self.t0
        eta = (elapsed / frac - elapsed) if frac > 0 else 0.0

        cond_text = "条件 %d/%d %s" % (
            min(self.completed_conditions + 1, self.total_conditions),
            self.total_conditions, self.condition_name)
        task_text = (" 任务 %d/%d" % (min(self.task_done, self.task_total),
                                      self.task_total)) if self.task_total > 0 else ""
        note_text = (" | %s" % note) if note else ""
        line = ("[%s] %5.1f%% | %s%s | 已用 %s 剩余 %s%s"
                % (bar, 100.0 * frac, cond_text, task_text,
                   _hms(elapsed), _hms(eta), note_text))
        end = "\r" if self._tty else "\n"
        self.stream.write(line + end)
        self.stream.flush()


def monitor_task_loop(monitor, items, runner, note_fn=None):
    """统一的任务循环包装：逐任务执行并回报进度。"""
    for index, item in enumerate(items, start=1):
        yield item
        if monitor is not None:
            monitor.task_update(index, note=note_fn(item, index) if note_fn else "")
