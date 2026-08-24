"""Terminal-UI-Primitive: Auswahlliste (Pfeiltasten), Status-Spinner,
Trennlinie, Prompt-Zeile. Keine externen Abhaengigkeiten."""

import sys
import shutil
import termios
import textwrap
import threading
import time
import tty

ESC = "\x1b"
RESET = "\x1b[0m"
DIM = "\x1b[2m"
INV = "\x1b[7m"
MARKER = "\u25b8"


def cols() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def hide_cursor() -> None:
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()


def show_cursor() -> None:
    sys.stdout.write("\x1b[?25h")
    sys.stdout.flush()


def clear() -> None:
    sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
    sys.stdout.flush()


def rule(char: str = "\u2500") -> None:
    sys.stdout.write(DIM + char * cols() + RESET + "\n")


def wrap(text: str, width: int | None = None, indent: str = "") -> str:
    width = width or min(cols(), 78)
    out = []
    for para in text.strip().split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.append(textwrap.fill(para.strip(), width=width,
                                 initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out)


class _Raw:
    """Terminal fuer die Dauer des Blocks in den Raw-Modus schalten."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def _key() -> str:
    ch = sys.stdin.read(1)
    if ch != ESC:
        return ch
    if sys.stdin.read(1) != "[":
        return ESC
    return {"A": "up", "B": "down"}.get(sys.stdin.read(1), "")


def select(title: str, labels: list[str]) -> int:
    """Pfeiltasten-Auswahl. Gibt den Index des gewaehlten Eintrags zurueck."""
    if not labels:
        raise SystemExit(f"{title}: nothing found.")

    width = cols() - 6
    view = [l if len(l) <= width else l[: width - 1] + "\u2026" for l in labels]

    print(title)
    print(DIM + "\u2191/\u2193 select \u00b7 Enter confirm \u00b7 q quit" + RESET)
    print()
    for _ in view:
        print()

    idx = 0
    hide_cursor()
    try:
        with _Raw():
            while True:
                sys.stdout.write(f"\x1b[{len(view)}A")
                for i, label in enumerate(view):
                    marker = MARKER if i == idx else " "
                    line = f"  {marker} {label} "
                    sys.stdout.write("\x1b[2K")
                    sys.stdout.write((INV + line + RESET if i == idx else line) + "\r\n")
                sys.stdout.flush()

                k = _key()
                if k == "up":
                    idx = (idx - 1) % len(view)
                elif k == "down":
                    idx = (idx + 1) % len(view)
                elif k in ("\r", "\n"):
                    return idx
                elif k in ("q", "\x03", "\x04"):
                    raise SystemExit(0)
    finally:
        show_cursor()


class Status:
    """Spinner. Zeigt ausschliesslich, DASS gerechnet wird."""

    FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        hide_cursor()
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r\x1b[2K{DIM}{frame}  {self.label}{RESET}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.08)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join()
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()
        show_cursor()


def ask(caret: str = "\u203a") -> str:
    """Eingabezeile. Leere Eingabe wird erneut abgefragt."""
    import readline  # noqa: F401  -- aktiviert Zeileneditierung

    while True:
        try:
            value = input(f"{caret} ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
        if value:
            return value
