"""Welche Modelle stehen zur Verfuegung.

Dieses Modul beantwortet genau eine Frage: was kann der Spieler auswaehlen?
Es laedt nichts und startet nichts - es listet nur auf. Das Laden macht
llm.py (Sprachmodelle) bzw. diffusion.py (Bildmodelle).

Es gibt genau zwei Quellen:
    Ollama    - laeuft auf dem Host und meldet seine Modelle ueber /api/tags
    HF-Cache  - ein Ordner, in dem Sprach- und Bildmodelle gemischt liegen

Die entscheidende Idee: die Trennung im Cache laeuft ueber die Datei
model_index.json. Wer sie hat, ist eine Diffusers-Pipeline (also ein
Bildmodell), wer sie nicht hat, ist ein Sprachmodell. Diese eine Regel
ersetzt jede Namensheuristik ("steht 'flux' im Namen?") und jede Blockliste.

Aufbau des HF-Cache, den wir hier durchsuchen:

    cache/huggingface/hub/
      models--Qwen--Qwen3-32B/            <- ein Repo, "--" trennt org/name
        snapshots/
          a1b2c3.../                      <- eine Version, benannt nach Commit
            config.json
            model-00001-of-00004.safetensors
      models--black-forest-labs--FLUX.2/
        snapshots/
          d4e5f6.../
            model_index.json              <- diese Datei macht es zum Bildmodell
            transformer/, vae/, ...

Python-Konzepte hier: dataclass, Generator (yield), Pfade mit pathlib,
Comprehensions und HTTP ohne externe Bibliothek.
"""

from __future__ import annotations

import json               # JSON lesen/schreiben
import os                 # Zugriff auf Umgebungsvariablen
import urllib.error       # die Fehlerarten von urllib
import urllib.request     # HTTP-Anfragen, ohne 'requests' installieren zu muessen
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path  # Pfade als Objekte statt als Strings

# os.environ.get(name, default) liest eine Umgebungsvariable. Ist sie nicht
# gesetzt, kommt der zweite Wert zurueck. So kann docker-compose.yml die
# Adressen ueberschreiben, ohne dass hier etwas geaendert werden muss.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")

SEP = " · "   # Trenner zwischen Modellname und Groesse in der Auswahlliste


# @dataclass ist ein "Decorator" - er schreibt Code fuer uns. Aus den drei
# Zeilen darunter erzeugt Python automatisch __init__, __repr__ und __eq__.
# Ohne ihn muesste man all das von Hand tippen.
# frozen=True macht die Objekte unveraenderlich: model.ref = "x" wirft dann
# einen Fehler. Das ist hier richtig, weil ein gefundenes Modell ein Fakt ist
# und niemand ihn nachtraeglich verbiegen koennen soll.
@dataclass(frozen=True)
class Model:
    backend: str   # "ollama" | "vllm" | "diffusers" - wer laedt das?
    ref: str       # Modellname (Ollama), Repo-ID (vLLM) oder Pfad (Bild)
    label: str     # Anzeigetext in der Auswahlliste


def _human(size: float) -> str:
    """Bytes in etwas Lesbares: 4300000000 -> "4.3 GB"."""
    # 1e9 ist wissenschaftliche Schreibweise fuer 1000000000.
    # Das f vor dem String macht ihn zum f-String: Ausdruecke in {} werden
    # ausgerechnet und eingesetzt. ":.1f" heisst "eine Nachkommastelle".
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


# ------------------------------------------------------------------ HF-Cache

def cache_root() -> Path | None:
    """Der eine Cache-Ordner - oder None, wenn keiner existiert.

    Im Container zeigen beide Kandidaten auf dasselbe Volume (siehe die
    volumes-Eintraege in docker-compose.yml). Der erste, den es wirklich
    gibt, gewinnt.
    """
    env = os.environ.get("AIGAME_CACHE")

    # Ist AIGAME_CACHE gesetzt, gilt nur dieser Pfad. Sonst probieren wir
    # die zwei ueblichen Orte. Der "/"-Operator bei Path haengt Ordner an:
    # Path.home() / ".cache" ergibt z.B. /root/.cache
    candidates = [Path(env)] if env else [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.cwd() / "cache" / "huggingface" / "hub",
    ]

    # next(generator, default) nimmt das erste Element - oder den Default,
    # wenn es keins gibt. Der Ausdruck in den Klammern ist ein Generator:
    # er prueft die Kandidaten einzeln und hoert beim ersten Treffer auf.
    return next((c for c in candidates if c.is_dir()), None)


def _snapshots() -> Iterator[tuple[str, Path]]:
    """Liefert (repo_id, neuester_snapshot) fuer jedes Repo im Cache.

    Diese Funktion ist ein Generator: sie benutzt "yield" statt "return".
    Statt eine fertige Liste zu bauen, gibt sie ein Ergebnis nach dem anderen
    heraus. Man benutzt sie wie eine Liste ("for a, b in _snapshots():"),
    aber sie haelt nie alles gleichzeitig im Speicher.
    """
    root = cache_root()
    if root is None:
        return   # In einem Generator heisst nacktes "return": nichts mehr da.

    # glob("models--*") findet alle Ordner, die so anfangen. Das * ist ein
    # Platzhalter fuer beliebigen Text.
    for repo in sorted(root.glob("models--*")):
        try:
            snaps = sorted(
                # Alle Unterordner von snapshots/ ...
                (d for d in (repo / "snapshots").iterdir() if d.is_dir()),
                # ... sortiert nach Aenderungszeit, neueste zuerst.
                # key= sagt: sortiere nicht nach dem Objekt selbst, sondern
                # nach diesem Wert. lambda ist eine namenlose Mini-Funktion.
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            # Ordner fehlt oder ist nicht lesbar - dieses Repo ueberspringen.
            continue

        if snaps:
            # "models--Qwen--Qwen3-32B" -> "Qwen/Qwen3-32B"
            # Erst das Praefix weg, dann "--" durch "/" ersetzen.
            yield repo.name.replace("models--", "").replace("--", "/"), snaps[0]


def _weights(snap: Path) -> int:
    """Gesamtgroesse der Gewichte in Bytes - oder 0, wenn unvollstaendig.

    "Gewichte" sind die eigentlichen Modelldaten, meist als *.safetensors.
    Grosse Modelle sind in mehrere Dateien ("Shards") aufgeteilt.

    Warum die Vollstaendigkeitspruefung? Ein abgebrochener Download sieht im
    Ordner fast normal aus. Ohne diese Pruefung wuerde das Modell in der
    Liste auftauchen, der Spieler waehlt es - und der Fehler kommt erst nach
    15 Minuten Ladezeit. Lieber hier auffallen.
    """
    index = snap / "model.safetensors.index.json"
    if index.is_file():
        try:
            # Die Index-Datei enthaelt "weight_map": eine Zuordnung von
            # Tensor-Name -> Dateiname. Uns interessieren nur die Dateinamen,
            # also die Werte (.values()), und Duplikate stoeren nicht - daher
            # set(), das jeden Namen nur einmal behaelt.
            shards = set(json.loads(index.read_text(encoding="utf-8"))
                         ["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError):
            return 0   # Index kaputt -> als unvollstaendig behandeln

        # all(...) ist True, wenn JEDE Datei existiert. Fehlt eine einzige,
        # ist der Download angebrochen.
        if not all((snap / s).exists() for s in shards):
            return 0

    try:
        # rglob sucht rekursiv, also auch in Unterordnern. Bildmodelle legen
        # ihre Gewichte in transformer/, vae/ usw. ab - deshalb rglob und
        # nicht das flache glob.
        return sum(p.stat().st_size for p in snap.rglob("*.safetensors"))
    except OSError:
        return 0


# ------------------------------------------------------------------ Auswahl

def ollama_models() -> list[Model]:
    """Was Ollama auf dem Host gerade anbietet.

    Ollama hat eine kleine HTTP-Schnittstelle. GET /api/tags liefert JSON
    mit allen installierten Modellen. Wir durchsuchen hier keinen Ordner -
    die Modelle liegen in Ollamas eigener Verwaltung.
    """
    try:
        # "with" ist ein Kontextmanager: er raeumt am Blockende automatisch
        # auf (hier: schliesst die Netzwerkverbindung), auch wenn mittendrin
        # ein Fehler auftritt. Immer so, nie ohne.
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            data = json.load(r)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        # Ollama laeuft nicht oder antwortet nicht - leere Liste.
        # main.py macht daraus eine verstaendliche Fehlermeldung.
        return []

    # Eine "list comprehension": baut eine Liste in einem Ausdruck.
    # Gelesen: nimm jedes m aus data["models"], das einen Namen hat, und
    # mache daraus ein Model-Objekt.
    # data.get("models", []) statt data["models"]: .get() liefert den
    # Default, wenn der Schluessel fehlt, statt einen KeyError zu werfen.
    return [Model("ollama", m["name"], f"{m['name']}{SEP}{_human(m.get('size', 0))}")
            for m in data.get("models", []) if m.get("name")]


def vllm_models() -> list[Model]:
    """Sprachmodelle im Cache: config.json da, model_index.json nicht.

    vLLM laedt nicht aus einem Ordner, sondern bekommt die Repo-ID
    ("Qwen/Qwen3-32B") und findet den Cache selbst. Deshalb steckt in
    Model.ref hier die ID, nicht der Pfad.
    """
    out = []
    for repo_id, snap in _snapshots():
        # Zwei Ausschlusskriterien in einem if, verbunden mit "or":
        # entweder es ist ein Bildmodell, oder es fehlt die config.json.
        if (snap / "model_index.json").is_file() or not (snap / "config.json").is_file():
            continue

        size = _weights(snap)
        if size:   # 0 bedeutet unvollstaendig - nicht anbieten
            out.append(Model("vllm", repo_id, f"{repo_id}{SEP}{_human(size)}"))
    return out


def image_models() -> list[Model]:
    """Bildmodelle im Cache: alles mit model_index.json.

    Hier steht in Model.ref der Pfad zum Snapshot-Ordner, denn diffusion.py
    laedt direkt von der Platte.
    """
    return [Model("diffusers", str(snap), f"{repo_id}{SEP}{_human(_weights(snap))}")
            for repo_id, snap in _snapshots()
            if (snap / "model_index.json").is_file()]

