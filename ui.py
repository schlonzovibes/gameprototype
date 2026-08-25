"""Terminal-Ausgabe. Einziger Besitzer von stdout.

=== Warum dieses Modul der einzige Schreiber ist ===

Sobald der Rahmen laeuft, wollen drei Beteiligte gleichzeitig auf denselben
Bildschirm schreiben: der Hauptthread (Szenen), der Spinner-Thread und der
Rahmen-Thread. Wuerden die einfach drauflosschreiben, wuerden sich ihre
Ausgaben ineinander schieben und das Bild zerreissen. Deshalb geht jeder
Schreibzugriff durch write() - und damit durch ein Lock, das immer nur einen
zur Zeit durchlaesst.

Genau deswegen gibt termimage.render() auch nur einen String zurueck, statt
selbst zu drucken.

=== ANSI-Escape-Sequenzen: der Spickzettel ===

Ein Terminal versteht Steuerbefehle, die als Text mitgeschickt werden. Sie
beginnen fast alle mit \\x1b (das ESC-Zeichen, Zeichen Nummer 27) gefolgt von
"[". Was danach kommt, ist der Befehl. Was hier vorkommt:

    \\x1b[2J        Bildschirm loeschen
    \\x1b[3J        Rueckscroll-Puffer loeschen
    \\x1b[H         Cursor nach ganz oben links
    \\x1b[Z;SH      Cursor in Zeile Z, Spalte S ("H" = Home)
    \\x1b[2K        aktuelle Zeile loeschen ("K" = Kill)
    \\x1b[NA        N Zeilen hoch ("A" = Above)
    \\x1b[O;Ur      Scroll-Region von Zeile O bis U festlegen (DECSTBM)
    \\x1b[r         Scroll-Region aufheben
    \\x1b7 / \\x1b8  Cursorposition sichern / wiederherstellen (ohne "["!)
    \\x1b[?25l/h    Cursor verstecken / zeigen
    \\x1b[0m        alle Farben und Effekte zuruecksetzen
    \\x1b[7m        invers (Vorder- und Hintergrund tauschen)
    \\x1b[38;2;R;G;Bm   Vordergrundfarbe als echtes RGB

=== Die Scroll-Region ist der Trick fuer den Rahmen ===

Normalerweise scrollt der ganze Bildschirm. Mit \\x1b[2;29r sagt man dem
Terminal: scrolle nur die Zeilen 2 bis 29. Zeile 1 und Zeile 30 bleiben dann
einfach stehen, egal wie viel Text durchlaeuft. Genau darauf sitzen unsere
Kopf- und Fusszeile.

Python-Konzepte hier: Threads, Locks, Events, Kontextmanager (__enter__ /
__exit__), Klassen mit Konstanten, und Signal-Handler.
"""

from __future__ import annotations

import shutil  # shutil.get_terminal_size() - wie gross ist das Fenster?
import signal  # Betriebssystem-Signale, hier: "Fenster wurde skaliert"
import sys  # sys.stdout / sys.stdin - die Standard-Ein-/Ausgabe
import termios  # Terminal-Einstellungen aendern (nur Linux/macOS!)
import textwrap  # Text auf eine Breite umbrechen
import threading  # Nebenlaeufigkeit: mehrere Dinge gleichzeitig
import tty  # Terminal in den "Raw"-Modus schalten (nur Linux/macOS!)

import gpu

# ACHTUNG: termios und tty gibt es unter Windows nicht. Dieses Programm laeuft
# im Linux-Container, deshalb ist das in Ordnung.

ESC = "\x1b"  # das ESC-Zeichen, Start jeder Sequenz
RESET = "\x1b[0m"  # Farben/Effekte zuruecksetzen
INV = "\x1b[7m"  # invers - markiert den gewaehlten Eintrag
GRAY = "\x1b[38;2;140;140;140m"  # definiertes Mittelgrau als echtes RGB.
# Besser als \x1b[2m ("dim"), weil das je
# nach Terminal-Theme voellig anders aussieht.
DARK_GRAY = "\x1b[38;2;80;80;80m"  # dunkler als GRAY - fuer die Fusszeile,
# die bewusst mehr zuruecktritt als der Kopf
MARKER = "▸"  # Pfeil vor dem gewaehlten Eintrag

# Der linke Rand, den JEDER Bildschirm dieses Spiels teilt: Kopfzeile
# ("RePlot"), Fusszeile ("RAM"), Auswahlmenues und Statuszeilen beginnen
# alle an dieser Spalte. Eine einzige Konstante statt verstreuter "  "
# an mehreren Stellen, damit sie nie auseinanderlaufen koennen.
INDENT = "  "

# Ein Lock ("Schloss") laesst immer nur einen Thread gleichzeitig hinein.
# Warum RLock und nicht das einfachere Lock? "R" steht fuer reentrant:
# derselbe Thread darf es mehrfach nehmen. Das brauchen wir, weil der
# SIGWINCH-Handler mitten im Hauptthread zuschlagen kann - moeglicherweise
# genau dann, wenn der das Lock schon haelt. Ein normales Lock wuerde sich
# dabei selbst blockieren (Deadlock), das Programm haengt fuer immer.
_LOCK = threading.RLock()


def write(text: str) -> None:
    """Der einzige Weg, etwas auf den Bildschirm zu bekommen."""
    # "with _LOCK:" nimmt das Lock beim Betreten und gibt es beim Verlassen
    # garantiert wieder frei - auch wenn drinnen ein Fehler auftritt.
    with _LOCK:
        sys.stdout.write(text)
        # flush() erzwingt die sofortige Ausgabe. Ohne das sammelt Python
        # Text in einem Puffer und schreibt erst spaeter - der Spinner wuerde
        # ruckeln und der Rahmen zu spaet erscheinen.
        sys.stdout.flush()


def size() -> tuple[int, int]:
    """(Spalten, Zeilen) des Terminals. Notfalls 80x24 als Annahme."""
    s = shutil.get_terminal_size((80, 24))
    return s.columns, s.lines


def cols() -> int:
    """Nur die Breite. [0] nimmt das erste Element des Tupels von size()."""
    return size()[0]


def hide_cursor() -> None:
    write("\x1b[?25l")


def show_cursor() -> None:
    write("\x1b[?25h")


def clear() -> None:
    """Ganzer Bildschirm inklusive Rueckscroll-Puffer (Setup, Titel)."""
    write("\x1b[2J\x1b[3J\x1b[H")


def clear_body() -> None:
    """Nur die Scroll-Region leeren - Kopf- und Fusszeile bleiben stehen.

    Der Cursor landet danach NICHT in Zeile 2, sondern in Zeile 3: Zeile 2
    bleibt als Abstand zur Kopfzeile leer. Diese eine Stelle setzt die Regel
    fuer jeden Bildschirmwechsel durch - kein Aufrufer muss mehr selbst an
    die fuehrende Leerzeile denken (und keiner kann sie mehr vergessen, so
    wie es bei der Backend- und Bildmodell-Auswahl passiert war, die direkt
    unter der Kopfzeile ohne jeden Abstand begannen).

    Warum umstaendlich zeilenweise statt einfach mit \\x1b[J? Weil \\x1b[J
    "loesche ab hier bis zum Bildschirmende" bedeutet - das wuerde die
    Fusszeile mitnehmen. Sie kaeme zwar eine Sekunde spaeter zurueck, aber
    genau das waere sichtbares Flackern bei jeder Szene.
    """
    # range(2, _bottom() + 1) zaehlt von 2 bis _bottom einschliesslich -
    # das obere Ende von range gehoert nie dazu, daher das "+ 1".
    # "".join(...) klebt alle erzeugten Stuecke zu einem String zusammen;
    # so geht alles in einem einzigen write() raus statt in vielen.
    # min(3, _bottom()): bei einem winzigen Fenster (_bottom() == 2) ist
    # kein Platz fuer die Leerzeile - dann eben ohne, statt eine Region zu
    # verlassen, die es gar nicht gibt.
    write(
        "".join(f"\x1b[{row};1H\x1b[2K" for row in range(2, _bottom() + 1))
        + f"\x1b[{min(3, _bottom())};1H"
    )


def wrap(text: str, width: int | None = None, indent: str = "") -> str:
    """Text auf eine Breite umbrechen und einruecken.

    textwrap.fill() kann das schon - aber es wirft dabei alle Absaetze
    zusammen. Deshalb behandeln wir jede Zeile einzeln und behalten die
    Leerzeilen zwischen Absaetzen.
    """
    # "a or b" liefert a, wenn a "wahr" ist (also nicht None, 0 oder leer),
    # sonst b. Eine kurze Art, einen Standardwert nachzureichen.
    width = width or min(cols(), 78)

    out = []
    for para in text.strip().split("\n"):
        if not para.strip():  # leere Zeile = Absatzgrenze, so belassen
            out.append("")
            continue  # continue springt zum naechsten Durchlauf
        out.append(
            textwrap.fill(
                para.strip(),
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )
        )
    return "\n".join(out)


# ------------------------------------------------------------------ Rahmen


def _bottom() -> int:
    """Letzte Zeile der Scroll-Region - darunter liegt die Fusszeile.

    Die Fusszeile ist zwei Zeilen hoch: eine leere Trennzeile, darunter die
    RAM/GPU-Anzeige. Deshalb "- 2", nicht "- 1" - die Scroll-Region endet
    eine Zeile frueher, damit fuer beide Platz bleibt.

    max(2, ...) verhindert Unsinn bei einem winzigen Fenster: waere die
    Rechnung dort kleiner als 2, entstuende eine verkehrt herum liegende
    Region (Ende vor dem Anfang), die das Terminal schlicht ignorieren wuerde.
    """
    return max(2, size()[1] - 2)


def _set_region(on: bool) -> None:
    """Scroll-Region einschalten (Zeile 2 bis _bottom) oder aufheben.

    Der Cursor wird drumherum gesichert (\\x1b7 ... \\x1b8), weil DECSTBM ihn
    als Nebenwirkung nach Zeile 1 setzt. Ohne das landet die naechste Ausgabe
    - typischerweise der Spinner - oben im Bild statt dort, wo der Text
    gerade steht. Der Spinner gehoert immer direkt unter die Eingabezeile.
    """
    # "A if B else C" ist Pythons einzeiliges if: ergibt A, wenn B wahr ist,
    # sonst C. Hier: einschalten oder zuruecksetzen.
    write("\x1b7" + (f"\x1b[2;{_bottom()}r" if on else "\x1b[r") + "\x1b8")


def _join(parts: list[str], width: int) -> str:
    """Felder mit " · " verbinden, solange sie ganz passen.

    Bei schmalem Terminal fallen hintere Felder weg - besser, als einen
    Balken mittendrin abzuschneiden. Passt nicht einmal das erste Feld,
    bleibt die Zeile leer; ein halber Balken sagt weniger aus als gar keiner.

    width kommt von aussen, weil die Einrueckung des Rahmens davon abgeht.
    """
    line = ""

    for part in parts:
        # Beim ersten Feld kein Trenner davor, danach schon.
        candidate = f"{line} · {part}" if line else part
        if len(candidate) > width:
            break  # break verlaesst die Schleife ganz
        line = candidate  # passt - uebernehmen und weiter probieren
    return line


class Frame:
    """Die festen Zeilen ausserhalb der Scroll-Region: Kopf, Trenner, Fuss.

    Zeile 1:              Titel, Szene, Sprachmodell - liefert das Spiel
                           per update()
    vorletzte Zeile:       leer - reiner Abstand, damit die Fusszeile nicht
                           am scrollenden Text klebt
    letzte Zeile:          RAM- und GPU-Balken - holt der Thread selbst,
                           einmal pro Sekunde, damit man beim Bildmodell
                           zusehen kann

    Alle drei Zeilen gehoeren einem einzigen Objekt und einem einzigen
    Thread. Mehrere getrennte Klassen muessten sich die Scroll-Region
    teilen - und Zustaendigkeit, die geteilt wird, gehoert zusammengelegt.

    Alle drei werden jede Sekunde neu gezeichnet: das kostet praktisch
    nichts und heilt Zeilen, die ein fremdes Escape doch einmal
    ueberschrieben hat.

    Der Rahmen laeuft ueber die gesamte Programmlaufzeit - schon waehrend
    Sprach- und Bildmodell laden, nicht erst ab der ersten Szene. Titel und
    Fusszeile stehen dann sofort, Szene und Sprachmodell tragen erst
    set_model() bzw. update() nach, sobald sie bekannt sind:

        frame = Frame("RePlot").start()
        ...                        # Modell waehlen, laden
        frame.set_model("qwen3:32b")
        ...                        # Geschichte starten
        frame.update(7, 15)
        ...
        frame.stop()               # nur einmal, ganz am Ende
    """

    # Variablen direkt in der Klasse (nicht in __init__) sind Konstanten, die
    # sich alle Objekte dieser Klasse teilen. Zugriff ueber self.BAR.
    BAR = 16  # Breite der Balken in Zeichen
    FULL, EMPTY = "█", "░"  # gefuelltes / leeres Balkensegment

    def __init__(self, title: str):
        """__init__ ist der Konstruktor - laeuft bei Frame(...) automatisch.

        "self" ist das Objekt selbst und immer der erste Parameter. Alles,
        was als self.x gespeichert wird, ueberlebt bis zum naechsten Aufruf.

        model und scene starten leer (None) statt mit einem erfundenen
        Platzhalter - _header() laesst leere Felder dann einfach weg, statt
        z.B. "Szene 1/15" zu zeigen, bevor ueberhaupt eine Szene existiert.
        """
        self.title = title
        self.model: str | None = None
        self.scene: tuple[int, int] | None = None

        # Ein Event ist ein Schalter, den mehrere Threads sehen koennen.
        # Der Thread prueft ihn; stop() legt ihn um; der Thread endet.
        # Man kann einen laufenden Thread nicht von aussen abschiessen -
        # man bittet ihn, selbst aufzuhoeren. Das ist der uebliche Weg.
        self._stop = threading.Event()

        # Noch kein Thread - der entsteht erst in start().
        self._thread: threading.Thread | None = None

    def set_model(self, model: str) -> None:
        """Sprachmodell-Namen eintragen, sobald er feststeht."""
        self.model = model
        self.draw()

    def update(self, number: int, max_scenes: int) -> None:
        """Neue Szenennummer setzen und sofort zeichnen (nicht erst in 1 s)."""
        self.scene = (number, max_scenes)
        self.draw()

    def reset_scene(self) -> None:
        """Szenenanzeige ausblenden - fuer den Titelbildschirm eines Neustarts.

        Ohne das wuerde beim naechsten "start a new story" kurz noch
        "Szene 15/15" von der vorigen Geschichte stehen, bis die erste neue
        Szene eintrifft.
        """
        self.scene = None
        self.draw()

    def start(self) -> Frame:
        """Region einrichten und den Zeichen-Thread starten.

        Gibt self zurueck, damit man verketten kann:
            frame = Frame(...).start()
        """
        _set_region(True)

        # Wird das Fenster skaliert, schickt das Betriebssystem SIGWINCH.
        # Dann stimmen Region und Fusszeilen-Position nicht mehr - neu setzen.
        # lambda *_: (...) ist eine namenlose Funktion, die beliebige
        # Argumente entgegennimmt und ignoriert (das "*_" schluckt sie).
        # Das Tupel dahinter fuehrt einfach beide Aufrufe nacheinander aus.
        signal.signal(signal.SIGWINCH, lambda *_: (_set_region(True), self.draw()))

        # target= ist die Funktion, die im neuen Thread laufen soll -
        # ohne Klammern! self._run() wuerde sie sofort hier ausfuehren,
        # self._run uebergibt sie nur.
        # daemon=True: Python wartet beim Beenden nicht auf diesen Thread.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Thread beenden und die Scroll-Region aufheben.

        Darf mehrfach aufgerufen werden ("idempotent"): main.py stoppt am
        Spielende, und das finally stoppt sicherheitshalber noch einmal.
        """
        if self._thread is None:
            return  # schon gestoppt - nichts mehr zu tun

        self._stop.set()  # Schalter umlegen: der Thread soll aufhoeren
        self._thread.join()  # warten, bis er wirklich fertig ist
        self._thread = None  # merken, dass gestoppt wurde

        signal.signal(signal.SIGWINCH, signal.SIG_DFL)  # Handler abmelden
        _set_region(False)  # Region aufheben

    def draw(self) -> None:
        """Kopfzeile, Trennzeile und Fusszeile neu malen.

        Drei Zeilen, nicht zwei: zwischen dem scrollenden Inhalt und der
        RAM/GPU-Anzeige bleibt immer eine leere Zeile Abstand - sonst klebt
        die Anzeige direkt an der letzten Textzeile.
        """
        # Der Ablauf: Cursor sichern (\x1b7), alle drei Zeilen schreiben,
        # Cursor zurueck (\x1b8). Dadurch bleibt der laufende Text dazwischen
        # - und eine halb getippte Eingabe - voellig unberuehrt.
        # Alles in EINEM write(), damit kein anderer Thread dazwischenfunkt.
        # Die Fusszeile ist bewusst dunkler als der Kopf - sie tritt zurueck.
        write(
            "\x1b7"
            + self._paint(1, self._header(), GRAY)  # Zeile 1
            + self._blank(size()[1] - 1)  # Trennzeile
            + self._paint(size()[1], self._footer(), DARK_GRAY)  # letzte Zeile
            + "\x1b8"
        )

    def _paint(self, row: int, text: str, color: str) -> str:
        """Eine Zeile ansteuern, leeren und in der gegebenen Farbe beschriften."""
        return f"\x1b[{row};1H\x1b[2K{INDENT}{color}{text}{RESET}"

    def _blank(self, row: int) -> str:
        """Eine Zeile nur leeren - der Abstand oberhalb der Fusszeile."""
        return f"\x1b[{row};1H\x1b[2K"

    def _width(self) -> int:
        """Platz fuer den Text - die Einrueckung geht von der Breite ab."""
        return cols() - len(INDENT)

    def _run(self) -> None:
        """Die Schleife des Hintergrund-Threads."""
        while not self._stop.is_set():
            self.draw()
            # wait(1.0) statt time.sleep(1.0): wait kehrt sofort zurueck,
            # sobald der Schalter umgelegt wird. Mit sleep muesste stop()
            # im schlimmsten Fall eine volle Sekunde warten.
            self._stop.wait(1.0)

    def _bar(self, fraction: float) -> str:
        """Einen Balken aus Bloecken bauen. fraction ist 0.0 bis 1.0."""
        # round() rundet auf eine ganze Zahl. min/max klemmen das Ergebnis
        # zwischen 0 und BAR - falls die GPU mal 101 % meldet oder ein Wert
        # negativ hereinkommt, laeuft der Balken trotzdem nicht ueber.
        filled = max(0, min(self.BAR, round(fraction * self.BAR)))
        # "*" wiederholt einen String: "█" * 3 ergibt "███".
        return self.FULL * filled + self.EMPTY * (self.BAR - filled)

    def _header(self) -> str:
        """Zeile 1: RePlot · Szene 7/15 · qwen3:32b

        Waehrend des Ladens sind scene und model noch None - dann steht dort
        nur "RePlot". Kein Feld wird mit einem erfundenen Wert gefuellt.
        """
        parts = [self.title]
        if self.scene:
            number, total = self.scene
            parts.append(f"Szene {number}/{total}")
        if self.model:
            parts.append(self.model)

        # Der Titel darf notfalls abgeschnitten werden - ein Balken nicht.
        # "or" springt ein, wenn _join() bei sehr schmalem Fenster "" liefert.
        return _join(parts, self._width()) or self.title[: self._width()]

    def _footer(self) -> str:
        """Letzte Zeile: RAM ███░░░ 46GB · GPU █████░ 62%

        RAM kommt aus 'free' (Hauptspeicher + Swap zusammengerechnet), nicht
        aus nvidia-smi/torch - auf Systemen mit gemeinsamem Speicher (GB10)
        ist das der Wert, der tatsaechlich knapp wird. Gezeigt wird nur der
        belegte Wert, nicht "x/y GB" - der Balken selbst traegt die Relation
        zum Gesamtspeicher bereits vor.

        Beide Teile einzeln optional: fehlt eine Quelle, faellt nur ihr
        Teil weg, nicht die ganze Zeile.
        """
        parts = []

        ram = gpu.ram_stats()
        if ram:
            used, total_mb = ram
            if total_mb:  # Schutz gegen Division durch 0
                # round() statt Abschneiden: 1500 MB soll "2GB" ergeben,
                # nicht truegerisch abgerundet "1GB".
                parts.append(f"RAM {self._bar(used / total_mb)} {round(used / 1024)}GB")

        util = gpu.gpu_util()
        if util is not None:
            # ":3d" = ganze Zahl auf drei Stellen rechtsbuendig, damit der
            # Wert beim Wechsel von 9 auf 10 nicht seitlich huepft.
            parts.append(f"GPU {self._bar(util / 100)} {util:3d}%")

        return _join(parts, self._width())


# ------------------------------------------------------------------ Eingabe


class _Raw:
    """Terminal fuer die Dauer des Blocks in den Raw-Modus schalten.

    Normalerweise sammelt das Terminal Eingaben zeilenweise und gibt sie erst
    bei Enter weiter. Fuer die Pfeiltasten-Auswahl brauchen wir aber jeden
    Tastendruck sofort - das ist der Raw-Modus.

    Diese Klasse ist ein Kontextmanager: __enter__ laeuft beim Betreten von
    "with", __exit__ beim Verlassen - garantiert, auch bei einem Fehler.
    Genau das ist hier lebenswichtig: bliebe das Terminal im Raw-Modus
    zurueck, waere die Shell nach dem Absturz unbenutzbar.
    """

    def __enter__(self):
        self.fd = sys.stdin.fileno()  # Nummer des Eingabekanals
        self.old = termios.tcgetattr(self.fd)  # alte Einstellungen merken
        tty.setraw(self.fd)  # umschalten
        return self

    def __exit__(self, *exc):
        # *exc schluckt die drei Argumente, die Python hier uebergibt
        # (Fehlerart, Fehler, Stacktrace) - wir brauchen sie nicht.
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)  # zurueck


def _key() -> str:
    """Einen Tastendruck lesen. Pfeiltasten werden zu "up"/"down".

    Pfeiltasten senden keine einzelnen Zeichen, sondern drei auf einmal:
    ESC, dann "[", dann "A" (hoch) bzw. "B" (runter). Deshalb lesen wir
    schrittweise weiter, sobald ein ESC hereinkommt.
    """
    ch = sys.stdin.read(1)
    if ch != ESC:
        return ch  # normale Taste - direkt zurueck
    if sys.stdin.read(1) != "[":
        return ESC  # ESC alleine gedrueckt
    # dict.get(schluessel, default) liefert den Default statt eines Fehlers,
    # wenn der Schluessel fehlt - hier bei Pfeil links/rechts ein leerer String.
    return {"A": "up", "B": "down"}.get(sys.stdin.read(1), "")


def _row(label: str, active: bool) -> str:
    """Eine Zeile der Auswahlliste - markiert oder nicht.

    INDENT bringt den Marker auf dieselbe Spalte wie "RePlot" im Kopf und
    "RAM" im Fuss. Das Label selbst rutscht durch Marker+Leerzeichen zwei
    Spalten weiter nach rechts - dafuer muss links Platz fuer den Pfeil sein.
    """
    line = f"{INDENT}{MARKER if active else ' '} {label} "
    return INV + line + RESET if active else line


def select(title: str, labels: list[str]) -> int:
    """Pfeiltasten-Auswahl. Gibt den Index des gewaehlten Eintrags zurueck.

    Der Index (0, 1, 2, ...) statt des Textes, weil der Aufrufer damit direkt
    in seine eigene Liste greifen kann: modelle[select(...)]
    """
    if not labels:
        # SystemExit beendet das Programm mit einer Meldung.
        raise SystemExit(f"{title}: nothing found.")

    # Zu lange Eintraege kuerzen und mit "…" markieren, damit die Liste nicht
    # umbricht - ein Umbruch wuerde die Cursor-Rechnung unten zerstoeren.
    width = cols() - 6
    view = [l if len(l) <= width else l[: width - 1] + "…" for l in labels]

    # Erst Titel und Hilfezeile, dann pro Eintrag eine Leerzeile. Die brauchen
    # wir, damit der Cursor gleich um genau so viele Zeilen hochspringen kann.
    # Beide Zeilen mit INDENT, damit sie mit dem Marker der Liste fluchten -
    # und mit Kopf- und Fusszeile des Rahmens.
    write(
        f"{INDENT}{title}\n{INDENT}{GRAY}↑/↓ select · Enter confirm · q quit"
        f"{RESET}\n\n" + "\n" * len(view)
    )

    idx = 0  # welcher Eintrag ist gerade markiert
    hide_cursor()
    try:
        with _Raw():
            while True:
                # Erst um len(view) Zeilen hoch, dann alle Zeilen neu malen.
                # So entsteht der Eindruck, dass sich die Markierung bewegt,
                # obwohl in Wahrheit die ganze Liste neu gezeichnet wird.
                # \r\n statt \n: im Raw-Modus bewegt \n nur nach unten, der
                # Wagenruecklauf \r an den Zeilenanfang muss dazu.
                # enumerate() liefert Position UND Wert: (0, "erster"), ...
                write(
                    f"\x1b[{len(view)}A"
                    + "".join(
                        "\x1b[2K" + _row(label, i == idx) + "\r\n"
                        for i, label in enumerate(view)
                    )
                )

                key = _key()
                if key == "up":
                    # Der Modulo-Operator % laesst die Auswahl umlaufen:
                    # oben angekommen springt sie nach ganz unten.
                    idx = (idx - 1) % len(view)
                elif key == "down":
                    idx = (idx + 1) % len(view)
                elif key in ("\r", "\n"):  # Enter
                    return idx
                elif key in ("q", "\x03", "\x04"):  # q, Strg+C, Strg+D
                    raise SystemExit(0)
    finally:
        # finally laeuft immer - auch bei return oder SystemExit. Ohne das
        # bliebe der Cursor unsichtbar zurueck.
        show_cursor()
        # Zwei Leerzeilen Abstand zu dem, was als Naechstes kommt. Ohne sie
        # klebt die naechste Ueberschrift ("Language model") direkt am
        # letzten Listeneintrag ("vllm"). Hier statt beim Aufrufer, damit
        # jede Liste gleich viel Luft nach unten hat.
        write("\n\n")


def ask(caret: str = "›") -> str:
    """Eingabezeile. Leere Eingabe wird erneut abgefragt."""
    # Import mitten in der Funktion, nicht oben: allein durch das Importieren
    # klinkt sich readline in input() ein und ermoeglicht Pfeiltasten,
    # Zeilenbearbeitung und History. Benutzt wird es sonst nirgends - daher
    # das "noqa", das dem Linter sagt: der unbenutzte Import ist Absicht.
    import readline  # noqa: F401

    while True:
        try:
            value = input(f"{caret} ").strip()  # strip() = Leerraum abschneiden
        except (EOFError, KeyboardInterrupt):
            # Strg+D bzw. Strg+C - das Programm sauber beenden.
            raise SystemExit(0)
        if value:  # leerer String ist "falsch" -> nochmal fragen
            return value


class Status:
    """Einzeilige Statusanzeige: Spinner, Text, optional ein Balken.

    Der Spinner allein zeigt nur, DASS gerechnet wird - das ist bei den
    meisten Vorgaengen auch alles, was ehrlich moeglich ist: weder Ollama
    noch das Bildmodell kann sagen, wie lange es noch braucht.

    Wo es echte Zahlen gibt - beim Laden der vLLM-Shards - kommt ein Balken
    dazu. Er erscheint nur, wenn tatsaechlich ein Wert gesetzt wurde, nie
    als Schaetzung. Ein Balken, der bei 80 % stehenbleibt, ist schlimmer
    als gar keiner.

    Kontextmanager, also so zu benutzen:
        with Status("the image model is working"):
            bild = male()

    Mit Fortschritt:
        with Status("loading model") as status:
            engine.load(status.update)

    Der Spinner startet beim Betreten und verschwindet beim Verlassen -
    auch dann, wenn dazwischen eine Exception fliegt.

    indent haengt links davor, wie bei select() - Default "": der Spinner
    waehrend einer laufenden Geschichte steht bewusst buendig unter der
    "›"-Eingabezeile, die ebenfalls keine Einrueckung traegt. Die Auswahl-
    Phase (Backend/Modell laden) uebergibt dagegen ui.INDENT, damit sie mit
    Kopf- und Fusszeile fluchtet.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # Braille-Zeichen, ergeben ein drehendes Muster
    BAR = 20
    FULL, EMPTY = "█", "░"

    def __init__(self, label: str, indent: str = ""):
        self.label = label
        self.indent = indent
        self.fraction: float | None = None  # None = kein Balken
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, fraction: float | None = None, label: str | None = None) -> None:
        """Fortschritt und/oder Text aendern.

        Die Signatur passt absichtlich genau auf den progress-Rueckruf in
        llm.py - main.py kann diese Methode direkt durchreichen, ohne eine
        Zwischenfunktion zu bauen.

        Der Thread zeichnet ohnehin zwoelfmal pro Sekunde neu; hier wird
        deshalb nur der Wert gesetzt, nicht gemalt.
        """
        self.fraction = fraction
        if label:
            self.label = label

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _text(self, frame: str) -> str:
        out = f"{self.indent}{frame}  {self.label}"
        if self.fraction is not None:
            filled = max(0, min(self.BAR, round(self.fraction * self.BAR)))
            bar = self.FULL * filled + self.EMPTY * (self.BAR - filled)
            # ":3.0f" = ohne Nachkommastellen, drei Stellen breit, damit
            # die Zahl beim Wechsel von 9 auf 10 nicht seitlich huepft.
            out += f"  {bar} {self.fraction * 100:3.0f}%"
        # Kuerzen, damit die Zeile nie umbricht - ein Umbruch wuerde die
        # \r-Technik unten zerstoeren und Reste stehen lassen.
        return out[: cols() - 1]

    def _run(self):
        hide_cursor()
        i = 0
        while not self._stop.is_set():
            # \r setzt den Cursor an den Zeilenanfang, \x1b[2K leert die
            # Zeile - so wird immer dieselbe Zeile ueberschrieben, statt den
            # Bildschirm mit Spinner-Zeilen vollzuschreiben.
            # i % len(FRAMES) laeuft endlos im Kreis durch die Zeichen.
            write(
                f"\r\x1b[2K{GRAY}{self._text(self.FRAMES[i % len(self.FRAMES)])}{RESET}"
            )
            i += 1
            self._stop.wait(0.08)  # ~12 Bilder pro Sekunde

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        write("\r\x1b[2K")  # Spinner-Zeile hinterlassen wir sauber
        show_cursor()
