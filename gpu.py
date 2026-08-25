"""Systemspeicher- und GPU-Telemetrie fuer die Fusszeile.

=== Warum "free" statt nvidia-smi/torch fuer den Speicher ===

Auf dem DGX Spark (GB10, Grace-Blackwell) teilen sich CPU und GPU denselben
physischen Speicher - es gibt keinen separaten VRAM. nvidia-smi und
torch.cuda.mem_get_info() melden dort fuer den GPU-Speicher deshalb oft nur
"[N/A]" oder wenig aussagekraeftige Werte.

"free" liest dagegen den tatsaechlichen Systemspeicher - genau den Pool,
der bei einem grossen Modell wirklich knapp wird. Das ist auf diesem
System einfacher UND treffender als der Umweg ueber die GPU-Werkzeuge.

Ausgewertet werden Hauptspeicher UND Swap zusammen: beide "used"-Werte
werden addiert und als ein einziger Wert gezeigt (z.B. "46GB"), nicht als
Bruch gegen die Gesamtgroesse - der Balken traegt die Relation bereits vor.

=== Die GPU-Auslastung in Prozent ===

Das kennt nur nvidia-smi. Dafuer bleibt es im Einsatz, aber ausschliesslich
fuer diesen einen Wert - kein Speicher-Umweg mehr darueber.

Python-Konzepte hier: subprocess (ein fremdes Programm aufrufen), ein
Modul-globaler Cache, und Typ-Hinweise mit "| None".
"""

# Diese Zeile erlaubt moderne Typ-Hinweise wie "int | None" auch in
# aelteren Python-Versionen. Sie muss ganz oben stehen, vor allen Imports.
from __future__ import annotations

import shutil        # shutil.which() sucht ein Programm im Suchpfad (PATH)
import subprocess    # startet fremde Programme und faengt deren Ausgabe ab

MB = 1024 * 1024   # Bytes pro Megabyte

# Modul-globale Variable: merkt sich, ob nvidia-smi ueberhaupt existiert.
# Drei moegliche Werte - True, False, und None fuer "noch nicht nachgesehen".
# Der Unterstrich am Anfang ist eine Konvention und bedeutet: privat, bitte
# von aussen nicht anfassen. Python erzwingt das nicht, es ist eine Bitte.
_smi_available: bool | None = None


def ram_stats() -> tuple[int, int] | None:
    """(belegt_MB, gesamt_MB) - Hauptspeicher UND Swap zusammengerechnet.

    Diese Funktion wirft nie eine Exception - sie laeuft im Hintergrund-
    Thread der Fusszeile, und eine Exception dort wuerde ihn still killen
    und die Anzeige fuer immer einfrieren lassen.

    "free -b" statt "free -h": -b liefert exakte Byte-Werte. -h ("human
    readable") wuerde Groessen wie "41Gi" oder "1,5T" ausgeben - mit
    wechselnden Einheiten und je nach Locale einem Komma statt Punkt als
    Dezimaltrennzeichen. Das waere unnoetig fehleranfaellig zu parsen, wo
    reine Ganzzahlen es nicht sind.
    """
    try:
        out = subprocess.run(["free", "-b"], capture_output=True,
                             text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None

    # Typische Ausgabe von "free -b":
    #               total        used        free      shared  buff/cache   available
    #   Mem:    134987...   43982...    12345...     ...         ...           ...
    #   Swap:     8589...           0    8589...
    mem = swap = None
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            if parts[0] == "Mem:":
                mem = (int(parts[1]), int(parts[2]))     # (total, used)
            elif parts[0] == "Swap:":
                swap = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue   # unerwartetes Format in dieser Zeile - ignorieren

    if mem is None:
        return None   # "free" lieferte gar keine brauchbare Mem-Zeile

    mem_total, mem_used = mem
    swap_total, swap_used = swap if swap else (0, 0)

    return (mem_used + swap_used) // MB, (mem_total + swap_total) // MB


def gpu_util() -> int | None:
    """GPU-Auslastung in Prozent - oder None, wenn nvidia-smi fehlt/schweigt."""
    global _smi_available

    # Nur beim allerersten Aufruf nachsehen - danach steht das Ergebnis fest.
    # "is None" statt "== None": is prueft auf Identitaet und ist bei None
    # die richtige Wahl. Ein blosses "if not _smi_available" wuerde auch bei
    # bereits ermitteltem False jedes Mal neu suchen.
    if _smi_available is None:
        _smi_available = shutil.which("nvidia-smi") is not None
    if not _smi_available:
        return None

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        # Erste Zeile = erste GPU; Mehr-GPU-Systeme sind hier nicht das Thema.
        # Erst float, dann int: je nach Treiberversion kommt "62.0" statt
        # "62", und int("62.0") wuerde einen Fehler werfen.
        return int(float(out.stdout.splitlines()[0].strip()))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None

