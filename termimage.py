"""Bild in eine Terminal-Ausgabe uebersetzen.

=== Wie ein Bild in ein Textfenster kommt ===

Es gibt vier Wege, und moderne Terminals koennen nur manche davon:

    kitty      Das Terminal Kitty (und Ghostty) versteht ein eigenes
               Protokoll: das PNG wird base64-kodiert mitgeschickt und
               echt gerendert. Beste Qualitaet.
    iterm      iTerm2 und WezTerm koennen dasselbe, mit anderer Syntax.
    sixel      Ein altes DEC-Format aus den 80ern, das erstaunlich viele
               Terminals wiederentdeckt haben. Braucht das Programm img2sixel.
    halfblock  Der Notausgang, der ueberall funktioniert: fuer jede
               Textzelle werden ZWEI Pixel genommen. Das Zeichen ▀ faerbt
               seine obere Haelfte in der Vordergrundfarbe, seine untere in
               der Hintergrundfarbe. So bekommt man doppelte vertikale
               Aufloesung aus einer Zelle. Braucht Truecolor-Faehigkeit.

=== Warum render() einen String zurueckgibt ===

Frueher hat dieses Modul selbst auf den Bildschirm geschrieben. Das geht
nicht mehr: seit der Rahmen aus ui.py in einem eigenen Thread laeuft, darf
nur noch ui.write() schreiben (mit Lock), sonst schieben sich die Ausgaben
ineinander. Deshalb baut render() nur den fertigen Text und gibt ihn zurueck.
"""

from __future__ import annotations

import base64        # Binaerdaten als Text kodieren
import io            # Datei-aehnliche Objekte im Arbeitsspeicher
import os
import shutil
import subprocess


def protocol() -> str:
    """Herausfinden, was dieses Terminal kann.

    Es gibt keine saubere Abfrage dafuer - man erkennt Terminals an den
    Umgebungsvariablen, die sie setzen. Deshalb diese Rateleiter, von der
    besten zur schlechtesten Option.
    """
    # Manuelle Uebersteuerung, falls die Erkennung danebenliegt.
    forced = os.environ.get("AIGAME_IMAGE_PROTOCOL")
    if forced:
        return forced

    term = os.environ.get("TERM", "")
    if os.environ.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term.lower():
        return "kitty"
    if os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm"):
        return "iterm"
    if shutil.which("img2sixel"):   # ist das Hilfsprogramm installiert?
        return "sixel"
    return "halfblock"              # geht immer


def fit(image, max_cols: int, max_rows: int) -> tuple[int, int]:
    """Zellmasse berechnen, ohne das Seitenverhaeltnis zu verzerren.

    Eine Terminalzelle ist ungefaehr doppelt so hoch wie breit - daher
    ueberall das "/ 2" bzw. "* 2". Ohne diese Korrektur waeren alle Bilder
    doppelt so hoch wie gewollt.
    """
    aspect = image.width / image.height

    # Erst so breit wie erlaubt, Hoehe passend dazu.
    cols = max_cols
    rows = max(1, round(cols / aspect / 2))

    # Zu hoch geworden? Dann andersherum rechnen: Hoehe festnageln,
    # Breite passend dazu.
    if rows > max_rows:
        rows = max_rows
        cols = max(1, round(rows * 2 * aspect))
    return cols, rows


def _png(image) -> bytes:
    """Das Bild als PNG-Bytes - ohne Umweg ueber die Festplatte.

    io.BytesIO() ist ein "Puffer, der sich wie eine Datei verhaelt".
    image.save() schreibt hinein, .getvalue() holt die Bytes wieder heraus.
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _kitty(image, cols: int, rows: int, indent: str) -> str:
    """Kitty-Grafikprotokoll: base64-PNG in Haeppchen von 4 KB."""
    data = base64.standard_b64encode(_png(image))

    # Ein Bild kann Megabytes gross sein, das Protokoll will kleine Stuecke.
    # Diese Zeile schneidet die Daten in 4096-Byte-Bloecke: range(start,
    # stop, schritt) zaehlt in 4096er-Spruengen, data[i:i+4096] schneidet.
    chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)]

    parts = [indent]
    for i, chunk in enumerate(chunks):
        # m=1 bedeutet "es kommt noch mehr", m=0 "das war das letzte Stueck".
        more = 1 if i < len(chunks) - 1 else 0
        # Nur das erste Stueck traegt die vollen Angaben (Format, Groesse).
        ctrl = f"a=T,f=100,c={cols},r={rows},m={more}" if i == 0 else f"m={more}"
        parts.append(f"\x1b_G{ctrl};{chunk.decode()}\x1b\\")

    # Nach dem Bild den Cursor um rows Zeilen nach unten und an den
    # Zeilenanfang - sonst wuerde der Erzaehltext ins Bild geschrieben.
    parts.append(f"\x1b[{rows}B\r")
    return "".join(parts)


def _iterm(image, cols: int, rows: int, indent: str) -> str:
    """iTerm2-Protokoll: alles in einer einzigen Sequenz."""
    b64 = base64.standard_b64encode(_png(image)).decode()
    return (f"{indent}\x1b]1337;File=inline=1;width={cols};height={rows};"
            f"preserveAspectRatio=1:{b64}\x07\n")


def _sixel(image, cols: int, rows: int, indent: str) -> str:
    """Sixel ueber das externe Programm img2sixel."""
    try:
        proc = subprocess.run(["img2sixel", "-w", str(cols * 8)],
                              input=_png(image),      # PNG in die Standardeingabe
                              capture_output=True, timeout=20)
        return indent + proc.stdout.decode("utf-8", "ignore")
    except (OSError, subprocess.SubprocessError):
        # Klappt nicht? Dann eben Halfblock - lieber ein grobes Bild als keins.
        return _halfblock(image, cols, rows, indent)


def _halfblock(image, cols: int, rows: int, indent: str) -> str:
    """Der Notausgang: zwei Pixel pro Textzelle, mit ▀ und zwei Farben."""
    # Auf exakt die Zellmasse skalieren - mal zwei in der Hoehe, weil jede
    # Zelle zwei Pixel uebereinander darstellt.
    img = image.convert("RGB").resize((cols, rows * 2))
    px = img.load()   # schneller Zugriff auf einzelne Pixel

    lines = []
    for y in range(rows):
        parts = [indent]
        for x in range(cols):
            # Zwei uebereinanderliegende Pixel holen. Jeder liefert drei
            # Werte (rot, gruen, blau), die direkt entpackt werden.
            r1, g1, b1 = px[x, y * 2]        # oberer Pixel
            r2, g2, b2 = px[x, y * 2 + 1]    # unterer Pixel
            # 38;2;R;G;B setzt die Vordergrundfarbe (obere Haelfte von ▀),
            # 48;2;R;G;B die Hintergrundfarbe (untere Haelfte).
            parts.append(f"\x1b[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m▀")
        # Am Zeilenende Farben zuruecksetzen, sonst faerbt sich der Rest ein.
        lines.append("".join(parts) + "\x1b[0m")
    return "\n".join(lines) + "\n"


def render(image, max_cols: int, max_rows: int, margin: int = 0) -> str:
    """Bild als druckfertigen String, links um `margin` eingerueckt."""
    cols, rows = fit(image, max_cols, max_rows)

    # Ein dict als Verteiler statt einer if/elif-Kette: protocol() liefert
    # den Schluessel, die eckigen Klammern holen die passende Funktion, und
    # die Klammern dahinter rufen sie auf.
    return {"kitty": _kitty, "iterm": _iterm, "sixel": _sixel,
            "halfblock": _halfblock}[protocol()](image, cols, rows, " " * margin)

