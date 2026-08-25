"""Sprachmodell-Backends und Szenen-Parsing.

=== Die Aufgabenteilung mit game_prompt.txt ===

Der Ausgabe-Contract liegt bewusst NICHT in game_prompt.txt, sondern hier:
die Datei traegt die WELT (wie die Geschichte funktioniert), der Code
erzwingt die FORM (wie die Antwort aussehen muss). So kann man am Spiel
schrauben, ohne den Client zu zerbrechen - und umgekehrt.

Den Szenenzaehler besitzt der Prompt, nicht der Code. Er kommt in
game.scene_number zurueck und speist nur die Kopfzeile. Der Code zaehlt
lediglich als Rueckfallebene mit, falls das Modell das Feld vergisst.

=== Was hier drin ist ===

    LLM         gemeinsame Basisklasse - definiert, was ein Backend koennen muss
    Ollama      spricht mit dem Ollama-Server auf dem Host
    VLLM        spricht mit dem vLLM-Server im zweiten Container
    Scene       das Ergebnis eines Zuges, aufgeraeumt
    parse_scene wandelt die JSON-Antwort in eine Scene

Python-Konzepte hier: Vererbung, abstrakte Methoden, Dataclasses, reguläre
Ausdruecke, Exception-Ketten (raise ... from) und HTTP-Fehlerbehandlung.
"""

from __future__ import annotations

import json
import os
import re                # regulaere Ausdruecke: Muster in Text suchen
import shlex             # Argumente sicher fuer eine Shell-Zeile quoten
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from models import OLLAMA_URL, VLLM_URL, Model

# Kontextfenster = wie viel Text das Modell gleichzeitig "sehen" kann,
# gemessen in Tokens (grob: Wortteile). Zu klein -> das Modell vergisst den
# Anfang der Geschichte, oder der Aufruf scheitert ganz.
#
# Ueberschlag, warum 40960 und nicht die frueheren 24576:
#     ~6500   game_prompt.txt + Start-Prompt (der System-Prompt)
#   ~15600   12 Zuege im Verlauf (HISTORY_TURNS), Szenen-JSON ~1300 je Zug
#    ~4096   Platz fuer die Antwort (MAX_TOKENS)
#   -------
#   ~26200   und damit bereits mehr, als 24576 hergaben
#
# Bei einem quantisierten Modell (NVFP4) sind die Gewichte klein und vom
# VLLM_GPU_UTIL-Budget bleibt reichlich fuer den KV-Cache - mehr Kontext
# kostet dort also kaum etwas. Gilt auch fuer Ollama (num_ctx).
NUM_CTX = int(os.environ.get("AIGAME_NUM_CTX", "40960"))

# Wie viele Tokens die Antwort hoechstens lang sein darf.
#
# Im Thinking-Modus muessen hier ZWEI Dinge hineinpassen: erst der
# Denkprozess, dann das vollstaendige Szenen-JSON mit state_update,
# Charakteren, Objekten und Dialog. Mit den frueheren 2048 wurde das JSON
# regelmaessig mittendrin abgeschnitten - parse_scene() landete dann auf
# dem Notfallpfad und die Szene war unbrauchbar.
#
# Betrifft nur vLLM: der Ollama-Payload setzt kein num_predict, dort ist
# die Antwortlaenge ohnehin nur durch NUM_CTX begrenzt.
MAX_TOKENS = int(os.environ.get("AIGAME_MAX_TOKENS", "4096"))

# Reasoning-Modus: manche Modelle koennen vor der Antwort "nachdenken".
# Das kostet Zeit, verbessert aber oft die Konsistenz. Der Denktext landet
# getrennt vom JSON und wandert nur ins Debug-Log.
# Der Vergleich mit einem Tupel erlaubt mehrere Schreibweisen in der env-Var.
THINK = os.environ.get("AIGAME_THINK", "0").lower() in ("1", "true", "on", "yes")

# Anteil des GPU-Speichers, den vLLM fuer sich reservieren darf - Gewichte
# UND KV-Cache zusammen. Auf dem DGX Spark (128 GB gemeinsamer Speicher)
# entsprechen 0.78 rund 100 GB.
#
# Zu niedrig ist gefaehrlicher als zu hoch: vLLM laedt die Gewichte trotzdem
# und scheitert erst danach beim KV-Cache - man sieht den Speicher volllaufen
# und dann komplett freigegeben werden. Bei 0.55 waeren es nur ~70 GB, also
# weniger als ein 80-GB-Modell allein schon braucht.
#
# Nach oben begrenzt das Bildmodell: es wird NACH dem Sprachmodell geladen
# und muss in den Rest passen. 0.78 laesst dafuer knapp 28 GB.
VLLM_GPU_UTIL = os.environ.get("AIGAME_VLLM_GPU_UTIL", "0.78")

# Zusaetzliche Flags fuer 'vllm serve', als eine Shell-Zeile.
#
# --enable-prefix-caching ist hier der eigentliche Gewinn. Der System-Prompt
# (game_prompt.txt + Start-Prompt, ~6500 Tokens) ist in JEDER Runde exakt
# derselbe und wurde bisher jedes Mal komplett neu durch den Prefill
# geschickt. Mit Prefix-Caching behaelt vLLM dessen KV-Cache und rechnet nur
# die neu hinzugekommenen Tokens.
#
# Das greift hier besonders gut: HISTORY_TURNS (12) liegt unter der
# Szenengrenze (15), der Verlauf wird also die meiste Zeit gar nicht
# gekuerzt - damit bleibt nicht nur der System-Prompt, sondern fast der
# ganze Kontext ein stabiler, wiederverwendbarer Praefix.
#
# In neueren vLLM-Versionen ist das ohnehin Standard; das Flag doppelt zu
# setzen schadet nicht. Sollte eine Version es NICHT kennen, weigert sich
# 'vllm serve' zu starten - dann diese Variable auf "" setzen (siehe
# docker-compose.yml). Der Fehler steht dank der Logdatei sofort sichtbar
# in der Meldung, statt erst nach dem Timeout aufzufallen.
VLLM_EXTRA_ARGS = os.environ.get("AIGAME_VLLM_ARGS", "--enable-prefix-caching")

MAX_SCENES = 15   # nur der Anzeige-Default, falls das Modell nichts meldet

# Suchmuster fuer pgrep/pkill, um den 'vllm serve'-Prozess zu finden.
#
# Warum die eckigen Klammern? pgrep -f durchsucht die kompletten Kommando-
# zeilen ALLER Prozesse - auch die der Shell, die pgrep gerade ausfuehrt.
# Stuende dort schlicht 'vllm serve', wuerde diese Shell sich selbst finden
# und immer "laeuft noch" melden, egal ob vLLM laengst gestorben ist. Die
# Absturzerkennung waere damit wirkungslos, und pkill wuerde obendrein
# seine eigene Shell abschiessen.
#
# 'vllm[ ]serve' ist als regulaerer Ausdruck gleichbedeutend mit
# "vllm serve", steht als Text in der Kommandozeile aber anders da - die
# Shell findet sich damit selbst nicht mehr. Ein klassischer Unix-Trick.
_SELF_SAFE = "'vllm[ ]serve'"

# Zum ENTLADEN reicht 'vllm serve' nicht: das ist nur der CLI-Einstieg.
# vLLM startet daneben einen eigenen Engine-Prozess (bei Tensor-Paralle-
# lismus mehrere Worker), und GENAU DIE halten den GPU-Speicher. Toetet man
# nur den Einstiegsprozess, verschwindet der Server zwar aus /v1/models,
# der Speicher bleibt aber belegt - und danach laesst sich kein Ollama mehr
# laden. Die Kindprozesse heissen je nach vLLM-Version anders (etwa
# "VLLM::EngineCore"), tragen aber alle "vllm" irgendwo in der Kommandozeile.
#
# -i macht die Suche gross-/kleinschreibungsunabhaengig, damit auch
# "VLLM::EngineCore" getroffen wird. Die eckigen Klammern sind derselbe
# Selbstschutz wie oben: die Shell, die pgrep ausfuehrt, traegt "vll[m]" in
# ihrer eigenen Kommandozeile und findet sich damit nicht selbst.
#
# Gefahrlos breit, weil das alles im PID-Namespace des vLLM-CONTAINERS
# laeuft - der sieht nur seine eigenen Prozesse. Das Bildmodell im
# Spielcontainer kann nicht versehentlich mitgetroffen werden.
_VLLM_ANY = "-i 'vll[m]'"

# Dieser Text wird an game_prompt.txt angehaengt (siehe story.system_prompt).
# Die dreifachen Anfuehrungszeichen erlauben mehrzeilige Strings.
# .strip() am Ende entfernt den Zeilenumbruch nach dem ersten """.
CONTRACT = """
AUSGABEFORMAT (verbindlich, keine Ausnahme):
Antworte ausschliesslich mit dem einzelnen JSON-Objekt aus dem Abschnitt
"REQUIRED OUTPUT FORMAT" mit den fuenf Top-Level-Feldern "game",
"state_update", "scene", "player_agency", "final_scene_output".
Kein Vorwort, kein Nachwort, keine Code-Fences, keine Kommentare im JSON.

Feldregeln fuer die vom Client weiterverarbeiteten Felder:
- game.scene_number: die Nummer DIESER Szene, beginnend bei 1.
- game.status: "active", oder "completed" sobald die Reise endet -
  spaetestens in Szene 15.
- final_scene_output.visual_scene_description: englische Bildbeschreibung,
  20-45 Woerter, nur Bildinhalt: Ort, Licht, Perspektive, Materialien,
  Atmosphaere. Keine Handlung, keine Sprache, kein Text im Bild.
- final_scene_output.narrator_text: deutscher Erzaehltext, 60-120 Woerter,
  zweite Person.
- scene.visual_prompt: englisch, direkt fuer ein Bildmodell nutzbar,
  konsistent mit der Bildbeschreibung und den persistenten Charakteren.
""".strip()

# Ein vorkompiliertes Suchmuster. Findet <think>...</think> samt Inhalt.
#   (.*?)      merkt sich, was dazwischen steht ("Gruppe 1")
#   re.DOTALL  laesst den Punkt auch Zeilenumbrueche treffen
# Einmal kompiliert statt bei jedem Aufruf neu - das ist schneller.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass(frozen=True)
class Scene:
    """Das Ergebnis eines Zuges - aufgeraeumt und fertig zur Anzeige."""
    narration: str    # deutscher Erzaehltext fuer den Bildschirm
    visual: str       # englische Bildbeschreibung fuer das Bildmodell
    raw: str          # das kanonische JSON, geht zurueck in den Chatverlauf
    number: int       # welche Szene ist das (1-15)
    max_scenes: int   # wie viele es insgesamt gibt
    completed: bool   # ist die Geschichte hier zu Ende?


class LLM:
    """Basisklasse: was jedes Sprachmodell-Backend koennen muss.

    Ollama und VLLM "erben" von dieser Klasse (class Ollama(LLM)) und
    uebernehmen dabei alles, was hier steht. Sie muessen nur complete()
    selbst ausfuellen - der Rest gilt fuer beide gleichermassen.

    Der Sinn: main.py und story.py muessen nicht wissen, welches Backend
    laeuft. Sie rufen .scene() auf, und das funktioniert immer.
    """

    last_thinking: str = ""   # Reasoning-Text des letzten complete()-Aufrufs

    def scene(self, messages: list[dict], fallback_number: int) -> Scene:
        """Einen Zug spielen: fragen und die Antwort auswerten."""
        return parse_scene(self.complete(messages), fallback_number)

    def load(self, progress=None) -> None:  # pragma: no cover
        """Modell bereitstellen.

        progress ist optional und wird - wenn uebergeben - waehrend des
        Ladens mit (anteil, beschriftung) aufgerufen. anteil ist 0.0 bis 1.0
        oder None, wenn gerade kein Fortschritt bekannt ist. Passt genau auf
        ui.Status.update(), sodass main.py einfach die Methode durchreicht.
        """
        raise NotImplementedError

    def complete(self, messages: list[dict]) -> str:  # pragma: no cover
        """Muss jede Unterklasse selbst implementieren.

        NotImplementedError ist das uebliche Signal fuer "hier fehlt noch
        etwas" - wer von LLM erbt und das vergisst, merkt es sofort.
        """
        raise NotImplementedError

    def unload(self) -> bool:
        """Modell wieder freigeben, falls das Backend das braucht.

        Anders als load()/complete(): kein NotImplementedError, sondern ein
        echter No-Op-Default. Ollama gibt seinen Speicher ueber keep_alive
        von selbst wieder frei - nur VLLM haelt einen eigenen Serverprozess
        am Laufen, der beim Beenden des Spiels explizit gestoppt werden
        muss, und ueberschreibt diese Methode entsprechend.

        Rueckgabe: True heisst "es ist nichts mehr belegt". Fuer Backends
        ohne eigenen Serverprozess ist das immer der Fall.
        """
        return True

    def _split_thinking(self, content: str) -> str:
        """<think>...</think> aus dem Content loesen.

        Denkmodelle schreiben ihren Denkprozess mitten in die Antwort. Das
        muss raus, sonst ist das JSON kaputt. Der Text wandert stattdessen
        nach self.last_thinking und von dort ins Debug-Log.
        """
        match = _THINK_RE.search(content)
        if not match:
            return content     # nichts gefunden - unveraendert zurueck

        block = match.group(1).strip()   # Gruppe 1 = das (.*?) im Muster

        # An vorhandenen Denktext anhaengen (manche Backends liefern beides:
        # ein eigenes Feld UND einen Inline-Block). strip() raeumt danach den
        # fuehrenden Umbruch weg, falls last_thinking vorher leer war.
        self.last_thinking = f"{self.last_thinking}\n{block}".strip()

        # sub() ersetzt das Gefundene durch "" - count=1 nur das erste Mal.
        return _THINK_RE.sub("", content, count=1).strip()


class Ollama(LLM):
    """Backend fuer den Ollama-Server auf dem Host.

    "(LLM)" bedeutet: diese Klasse erbt von LLM. Sie bekommt scene() und
    _split_thinking() geschenkt und fuellt nur complete() und load() aus.
    """

    def __init__(self, model: Model):
        self.name = model.ref        # z.B. "qwen3:32b"
        self.last_thinking = ""

    def load(self, progress=None) -> None:
        """Modell in den Speicher ziehen, ohne Text zu erzeugen.

        Ein leerer Prompt reicht Ollama als Aufforderung zum Laden.
        keep_alive sorgt dafuer, dass es 30 Minuten geladen bleibt - sonst
        wirft Ollama es nach kurzer Zeit wieder raus und jede Szene wartet.

        progress wird hier nicht benutzt: Ollama meldet keinen Fortschritt,
        der Aufruf blockiert einfach bis fertig. Der Parameter steht nur da,
        damit main.py beide Backends gleich behandeln kann.
        """
        self._post("/api/generate", {"model": self.name, "prompt": "",
                                     "keep_alive": "30m", "stream": False})

    def unload(self) -> bool:
        """Modell sofort aus dem Speicher werfen.

        keep_alive: 0 ist Ollamas Gegenstueck zum "30m" in load() - es weist
        den Server an, das Modell unmittelbar nach diesem Aufruf freizugeben,
        statt es noch minutenlang vorzuhalten.

        Ohne diese Methode wuerde Ollama den No-Op-Default der Basisklasse
        erben und "freigegeben" melden, ohne etwas zu tun - der Befehl waere
        bei diesem Backend also schlicht gelogen.
        """
        try:
            self._post("/api/generate", {"model": self.name, "prompt": "",
                                         "keep_alive": 0, "stream": False})
            return True
        except Exception:
            return False

    def complete(self, messages: list[dict]) -> str:
        """Den Verlauf schicken und die Antwort als Text zurueckbekommen."""
        # Ein dict ist Pythons Woerterbuch: Schluessel -> Wert. Das hier wird
        # gleich zu JSON und geht so an den Server.
        payload = {
            "model": self.name,
            "messages": messages,
            "stream": False,             # alles auf einmal, nicht Stueck fuer Stueck
            "format": "json",            # Ollama zwingt das Modell zu gueltigem JSON
            "options": {"temperature": 0.9, "num_ctx": NUM_CTX},
            # temperature 0.9 = recht kreativ. Niedriger waere braver und
            # vorhersehbarer - fuer eine generative Geschichte ungeeignet.
        }
        if THINK:
            payload["think"] = True

        try:
            data = self._post("/api/chat", payload)
        except RuntimeError as e:
            # Modelle ohne Reasoning-Support lehnen "think" mit HTTP 400 ab.
            # Statt zu scheitern: einmal ohne Denkprozess wiederholen.
            # str(e) macht aus der Exception ihren Text, .lower() macht den
            # Vergleich unabhaengig von Gross-/Kleinschreibung.
            if not THINK or "does not support thinking" not in str(e).lower():
                raise   # nacktes "raise" wirft den urspruenglichen Fehler weiter
            payload.pop("think", None)   # Feld entfernen, None = kein Fehler wenn weg
            data = self._post("/api/chat", payload)

        # Ollama meldet Fehler manchmal mit HTTP 200 - dann steht die Ursache
        # im error-Feld und der Content bleibt leer. Ohne diese Pruefung
        # bekaeme man nur ein raetselhaftes "leere Antwort".
        if data.get("error"):
            raise RuntimeError(f"Ollama: {data['error']}")

        message = data.get("message", {})
        self.last_thinking = message.get("thinking") or ""
        content = self._split_thinking(message.get("content", "").strip())

        if not content:
            raise RuntimeError(f"Ollama lieferte eine leere Antwort von "
                               f"{self.name} (num_ctx={NUM_CTX}).")
        return content

    def _post(self, path: str, payload: dict) -> dict:
        try:
            return _json_post(OLLAMA_URL + path, payload, timeout=600)
        except urllib.error.HTTPError as e:
            # Ollama steckt die Ursache in den Body der Fehlerantwort (z.B.
            # "does not support thinking") - ohne ihn bleibt nur "HTTP 400",
            # und man sucht sich zu Tode.
            detail = e.read().decode(errors="replace").strip()
            # "from None" unterdrueckt die urspruengliche Exception in der
            # Fehlerausgabe. Der HTTPError sagt nichts Zusaetzliches, und
            # eine kurze Meldung ist im Spiel besser lesbar.
            raise RuntimeError(f"Ollama: HTTP {e.code} - {detail[:300]}") from None


class VLLM(LLM):
    """Client fuer den OpenAI-kompatiblen Endpoint eines vLLM-Servers.

    Besonderheit: load() schaltet den Server bei Bedarf selbst um. Der
    Compose-Container "vllm" laeuft idle (sleep infinity) und bekommt das
    gewaehlte Modell per Docker-exec eingeschaltet - siehe docker-compose.yml.
    Dafuer ist der Docker-Socket in den Spielcontainer gemountet.
    """

    CONTAINER = os.environ.get("AIGAME_VLLM_CONTAINER", "vllm")
    TIMEOUT = int(os.environ.get("AIGAME_VLLM_TIMEOUT", "900"))   # 15 Minuten

    # Wohin die Ausgabe von 'vllm serve' umgeleitet wird. Ohne diese Datei
    # waere sie unrettbar verloren: ein detached exec-Prozess schreibt
    # nirgendwohin, und in 'docker logs vllm' steht er auch nicht - dort
    # erscheint nur der Hauptprozess des Containers (sleep infinity).
    # Aus dieser Datei kommen beides: der Ladefortschritt und, wenn vLLM
    # stirbt, die Ursache.
    LOG = "/tmp/aigame-vllm.log"

    def __init__(self, model: Model):
        self.name = model.ref     # HF-Repo-ID; muss dem Namen entsprechen,
        self.last_thinking = ""   # den vLLM spaeter serviert

    # ------------------------------------------------------------ Laden

    def load(self, progress=None) -> None:
        """Den Server auf das gewaehlte Modell bringen und darauf warten.

        Wartet nicht blind, sondern liest waehrenddessen das Log mit:
        - der Shard-Fortschritt wandert ueber progress() in die Statuszeile
        - stirbt der Prozess, bricht das hier sofort mit der echten Ursache
          ab, statt bis zum Timeout stumm zu warten
        """
        if self.name in self._served():
            return   # laeuft schon - nichts zu tun

        box = self._container()
        self._serve(box)

        # time.monotonic() ist eine Uhr, die nur vorwaerts laeuft und von
        # Zeitumstellungen unbeeindruckt bleibt - fuer Zeitmessung richtig,
        # anders als time.time().
        deadline = time.monotonic() + self.TIMEOUT
        tail = ""
        while time.monotonic() < deadline:
            if self.name in self._served():
                if progress:
                    progress(1.0, "vLLM ready")
                return

            alive, tail = self._probe(box)

            if progress:
                # Das Sternchen packt das zurueckgegebene Tupel wieder in
                # zwei Argumente aus.
                progress(*_vllm_progress(tail))

            if not alive:
                raise RuntimeError(
                    f"vLLM beendete sich beim Laden von {self.name}.\n\n"
                    f"{_last_lines(tail)}\n\n"
                    f"Vollstaendiges Log: docker exec {self.CONTAINER} "
                    f"cat {self.LOG}")

            time.sleep(2)

        raise RuntimeError(
            f"vLLM meldete {self.name} nicht innerhalb von "
            # "//" ist Ganzzahldivision: 900 // 60 ergibt 15, nicht 15.0
            f"{self.TIMEOUT // 60} Minuten als bereit.\n\n{_last_lines(tail)}")

    def _served(self) -> list[str]:
        """Welche Modelle serviert vLLM gerade? Leere Liste, wenn er schweigt."""
        try:
            with urllib.request.urlopen(VLLM_URL + "/v1/models", timeout=3) as r:
                return [m.get("id", "") for m in json.load(r).get("data", [])]
        except (urllib.error.URLError, OSError, json.JSONDecodeError,
                TimeoutError, ValueError):
            return []

    def _container(self):
        """Das Docker-Objekt des vLLM-Containers holen."""
        try:
            import docker
        except ImportError:
            raise RuntimeError(
                "Python package 'docker' is missing - install it or start "
                f"vLLM manually: vllm serve {self.name} --port 8000") from None
        try:
            return docker.from_env().containers.get(self.CONTAINER)
        except Exception as e:
            raise RuntimeError(
                f"Docker container '{self.CONTAINER}' is not reachable ({e}). "
                "Is the compose stack up and is the docker socket mounted?") from e
            # "from e" haengt den Originalfehler an - anders als oben, weil
            # die Docker-Meldung hier oft die eigentliche Ursache nennt.

    def _probe(self, box) -> tuple[bool, str]:
        """(laeuft noch?, letzte Logzeilen) - in einem einzigen exec.

        Zwei getrennte Aufrufe waeren zwei Runden ueber den Docker-Socket,
        alle zwei Sekunden. Eine Shell-Zeile erledigt beides.
        """
        try:
            result = box.exec_run(["sh", "-c",
                                   f"pgrep -f {_SELF_SAFE} >/dev/null "
                                   f"&& echo __ALIVE__; tail -c 4000 {self.LOG}"])
            out = result.output.decode(errors="replace")
        except Exception:
            return True, ""   # Socket zickt - im Zweifel weiterwarten
        return "__ALIVE__" in out, out.replace("__ALIVE__", "", 1)

    def _pkill(self, box, wait: int = 15) -> bool:
        """Alle vLLM-Prozesse im Container beenden und auf die Freigabe warten.

        Warum das mehr ist als ein pkill auf 'vllm serve':

        'vllm serve' ist nur der CLI-Einstieg. vLLM startet daneben einen
        eigenen Engine-Prozess (bei Tensor-Parallelismus sogar mehrere
        Worker), und GENAU DIE halten den GPU-Speicher. Toetet man nur den
        Einstiegsprozess, verschwindet der Server aus /v1/models, der
        Speicher bleibt aber belegt - dann laesst sich danach kein Ollama
        mehr laden.

        Statt die Namen dieser Kindprozesse zu raten (sie unterscheiden sich
        zwischen vLLM-Versionen), fragen wir nvidia-smi direkt, WER gerade
        GPU-Speicher belegt, und beenden genau diese PIDs. Das laeuft im
        PID-Namespace des vLLM-Containers - der sieht nur seine eigenen
        Prozesse, das Bildmodell im Spielcontainer kann also nicht
        versehentlich mitgetroffen werden.

        Ablauf: erst SIGTERM (freundlich, damit vLLM aufraeumen kann), dann
        warten, dann SIGKILL fuer alles, was sich weigert.

        Gibt True zurueck, wenn am Ende kein Prozess mehr GPU-Speicher haelt.
        """
        # Warum pgrep/pkill und NICHT nvidia-smi --query-compute-apps:
        # nvidia-smi meldet in einem Container die PIDs aus Sicht des HOSTS,
        # nicht die des Container-PID-Namespace. Ein kill auf diese Zahlen
        # ginge im Container ins Leere - oder traefe zufaellig einen ganz
        # anderen Prozess, der dieselbe Nummer hat. pgrep und pkill arbeiten
        # dagegen von Natur aus im richtigen Namespace.
        #
        # "|| true" haengt an jedem Kill, damit die Shell nicht abbricht,
        # falls ein Prozess in der Zwischenzeit schon von selbst weg ist.
        # Stirbt ein Prozess, gibt der Treiber seinen GPU-Speicher frei -
        # "kein vLLM-Prozess mehr da" heisst also "Speicher frei".
        script = f"""
        pkill -TERM -f {_VLLM_ANY} 2>/dev/null || true
        for i in $(seq 1 {wait}); do
            pgrep -f {_VLLM_ANY} >/dev/null 2>&1 || break
            sleep 1
        done
        if pgrep -f {_VLLM_ANY} >/dev/null 2>&1; then
            pkill -KILL -f {_VLLM_ANY} 2>/dev/null || true
            sleep 2
        fi
        sleep 1
        pgrep -f {_VLLM_ANY} >/dev/null 2>&1 || echo __FREED__
        """
        try:
            result = box.exec_run(["sh", "-c", script])
            return b"__FREED__" in result.output
        except Exception:
            return False   # Socket zickt - Aufrufer entscheidet, was das heisst

    def unload(self) -> bool:
        """vLLM beenden und den belegten GPU-Speicher wirklich freigeben.

        Wird beim Beenden des Spiels aufgerufen (main.py). Ohne das bliebe
        der ueber VLLM_GPU_UTIL reservierte Speicher (auf dem DGX Spark oft
        ~100 GB) belegt - und danach laesst sich kein Ollama mehr laden.

        Der Container selbst bleibt ausdruecklich laufen (kein 'compose
        down'): nur der Service endet, der Container wartet weiter in seinem
        'sleep infinity' auf den naechsten Spielstart.

        Rueckgabe True, wenn der Speicher nachweislich frei ist. False heisst
        "nicht bestaetigt" - main.py sagt das dann sichtbar, statt den Nutzer
        im Glauben zu lassen, er koenne jetzt Ollama starten.

        Kann kein Container gefunden werden (Docker schon weg, Verbindung
        tot): dann ist ohnehin nichts mehr zu stoppen - das Beenden des
        Spiels darf daran nicht scheitern.
        """
        try:
            box = self._container()
        except RuntimeError:
            return True   # kein Container = nichts belegt uns noch Speicher
        return self._pkill(box)

    def _serve(self, box) -> None:
        """'vllm serve <modell>' im Nachbarcontainer starten."""
        # Alte Instanz stoppen (beim Modellwechsel) und warten, BIS der
        # Speicher wirklich frei ist. _pkill() prueft das selbst per
        # nvidia-smi - deshalb hier kein blindes sleep mehr, das entweder
        # zu kurz waere (neues Modell scheitert an belegtem Speicher) oder
        # unnoetig lange bremst.
        self._pkill(box)

        # shlex.join baut aus der Argumentliste eine Shell-Zeile und quotet
        # dabei alles, was Sonderzeichen enthaelt. Noetig, weil wir wegen
        # der Umleitung ">" eine echte Shell brauchen und den Modellnamen
        # nicht ungeprueft hineinschreiben wollen.
        # shlex.split zerlegt die Zusatz-Flags so, wie eine Shell es taete
        # (respektiert also Anfuehrungszeichen); shlex.join setzt danach alles
        # wieder korrekt gequotet zusammen. Ist die Variable leer, kommt eine
        # leere Liste heraus und nichts wird angehaengt.
        command = shlex.join([
            "vllm", "serve", self.name,
            "--host", "0.0.0.0",
            # rsplit(":", 1) teilt am LETZTEN Doppelpunkt, [-1] nimmt das
            # letzte Stueck: aus "http://vllm:8000" wird "8000".
            "--port", VLLM_URL.rsplit(":", 1)[-1],
            "--gpu-memory-utilization", VLLM_GPU_UTIL,
            "--max-model-len", str(NUM_CTX),
        ] + shlex.split(VLLM_EXTRA_ARGS))
        try:
            # 2>&1 leitet auch die Fehlerausgabe in dieselbe Datei - genau
            # dort steht der Traceback, wenn vLLM beim Laden abbricht.
            box.exec_run(["sh", "-c", f"{command} > {self.LOG} 2>&1"],
                         detach=True)   # nicht warten - der Server laeuft dauerhaft
        except Exception as e:
            raise RuntimeError(
                f"Could not start 'vllm serve' inside container "
                f"'{self.CONTAINER}' ({e}). Is the container running?") from e

    # ----------------------------------------------------------- Abfragen

    def complete(self, messages: list[dict]) -> str:
        payload = {
            "model": self.name,
            "messages": messages,
            "temperature": 0.9,
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},   # JSON erzwingen
        }
        try:
            data = _json_post(VLLM_URL + "/v1/chat/completions", payload, timeout=600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace").strip()
            raise RuntimeError(f"vLLM: HTTP {e.code} - {detail[:300]}") from None

        # Hier ohne .get(): fehlt "choices", ist die Antwort so kaputt, dass
        # ein lauter Fehler ehrlicher ist als ein stiller Ersatzwert.
        message = data["choices"][0]["message"]
        self.last_thinking = message.get("reasoning_content") or ""
        content = self._split_thinking((message.get("content") or "").strip())

        if not content:
            raise RuntimeError(f"vLLM returned an empty response from {self.name}.")
        return content


# vLLM zeigt das Laden der Gewichte als tqdm-Balken:
#   Loading safetensors checkpoint shards:  45% Completed | 9/20 [00:12<00:15]
# tqdm ueberschreibt sich dabei mit \r, in der Logdatei stehen also alle
# Zwischenstaende hintereinander. Deshalb wird immer der LETZTE Treffer
# genommen. [^\n\r]*? bleibt bewusst innerhalb einer Zeile, damit das Muster
# nicht ueber mehrere Zwischenstaende hinweg zusammengesetzt wird.
_SHARD_COUNT_RE = re.compile(r"checkpoint shards:[^\n\r]*?(\d+)/(\d+)")
_SHARD_PCT_RE = re.compile(r"checkpoint shards:\s*(\d+)%")


def _last_lines(tail: str, count: int = 12) -> str:
    """Die letzten nicht-leeren Zeilen - fuer Fehlermeldungen.

    \\r wird zu \\n, sonst waeren alle tqdm-Zwischenstaende eine Riesenzeile.
    Danach wird jede Reihe solcher Zwischenstaende auf ihren letzten
    eingedampft: bei 17 Shards waeren das sonst 17 Rauschzeilen, die den
    eigentlichen Traceback aus dem Fenster draengen. Der letzte bleibt
    stehen, weil er zeigt, wie weit das Laden gekommen ist.
    """
    lines = [l.strip() for l in tail.replace("\r", "\n").splitlines() if l.strip()]

    kept = []
    for i, line in enumerate(lines):
        following = lines[i + 1] if i + 1 < len(lines) else ""
        if "checkpoint shards:" in line and "checkpoint shards:" in following:
            continue
        kept.append(line)

    return "\n".join(kept[-count:]) if kept else "(kein Log vorhanden)"


def _vllm_progress(tail: str) -> tuple[float | None, str]:
    """(Anteil 0..1 oder None, Beschriftung) aus dem vLLM-Log.

    Findet sich kein Shard-Fortschritt, wird die letzte Logzeile zur
    Beschriftung. Das ist waehrend der uebrigen Phasen sogar nuetzlicher
    als ein Balken - man sieht dann, ob vLLM gerade Speicher profiliert,
    den CUDA-Graph baut oder den Server hochfaehrt.
    """
    counts = _SHARD_COUNT_RE.findall(tail)
    if counts:
        done, total = int(counts[-1][0]), int(counts[-1][1])
        if total:
            return done / total, f"loading shards {done}/{total}"

    percents = _SHARD_PCT_RE.findall(tail)
    if percents:
        return int(percents[-1]) / 100, "loading shards"

    for line in reversed(tail.replace("\r", "\n").splitlines()):
        line = line.strip()
        if line:
            return None, line[:70]
    return None, "starting vLLM"


def _json_post(url: str, payload: dict, timeout: int) -> dict:
    """JSON hinschicken, JSON zurueckbekommen. Von beiden Backends benutzt."""
    req = urllib.request.Request(
        url,
        # .encode() macht aus dem String Bytes - das Netz transportiert
        # keine Strings, nur Bytes.
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def build(model: Model) -> LLM:
    """Aus einem Model-Eintrag das passende Backend-Objekt bauen.

    Ein dict als Ersatz fuer eine if/elif-Kette: der Schluessel waehlt die
    Klasse aus, die Klammern dahinter erzeugen davon ein Objekt.
    Gleichbedeutend mit:
        if model.backend == "ollama": return Ollama(model)
        elif model.backend == "vllm": return VLLM(model)
    """
    return {"ollama": Ollama, "vllm": VLLM}[model.backend](model)


def _int(value, default: int) -> int:
    """In eine ganze Zahl wandeln - oder den Default nehmen.

    Lokale Modelle liefern die Szenennummer mal als 7, mal als "7", mal gar
    nicht. Das faengt alle drei Faelle ab.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default   # None -> TypeError, "sieben" -> ValueError


def parse_scene(raw: str, fallback_number: int) -> Scene:
    """Die Modellantwort in eine Scene verwandeln.

    Diese Funktion faellt nie hart aus - die Szene muss weitergehen. Lokale
    Modelle halten sich nicht immer an das Format: mal steht das JSON in
    ```-Bloecken, mal mit "Hier ist deine Szene:" davor, mal ist es kaputt.
    Alles das wird hier abgefangen.

    Szenennummer und Limit kommen aus dem Modell; meldet es nichts
    Brauchbares, zaehlt der Client mit fallback_number selbst weiter.
    """
    text = raw.strip()

    # Fall 1: Das Modell hat Code-Fences drumgelegt (```json ... ```).
    if text.startswith("```"):
        # strip("`") entfernt alle Backticks vorne und hinten.
        # split("\n", 1)[-1] wirft die erste Zeile weg (meist "json").
        text = text.strip("`").split("\n", 1)[-1]

    # Fall 2: Prosa vor oder nach dem JSON. Wir schneiden von der ersten "{"
    # bis zur letzten "}" - alles ausserhalb ist Geschwaetz.
    # find() liefert -1, wenn nichts gefunden wurde.
    start, end = text.find("{"), text.rfind("}")
    obj = None
    if start != -1 and end > start:
        text = text[start:end + 1]   # +1, weil das Ende exklusiv ist
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None   # kaputtes JSON - unten faellt es auf den Notfall

    # Fall 3: Notfall. Gar kein brauchbares JSON - dann zeigen wir eben den
    # Rohtext als Erzaehltext an. Besser als ein leerer Bildschirm.
    if not isinstance(obj, dict):
        return Scene(raw.strip(), "", raw.strip(), fallback_number, MAX_SCENES, False)

    # Eine Funktion innerhalb einer Funktion. Sie sieht obj automatisch
    # (das nennt sich Closure) und spart drei fast gleiche Zeilen.
    def section(key: str) -> dict:
        """Ein Unterobjekt holen - oder ein leeres dict, wenn es fehlt
        oder etwas anderes als ein Objekt ist."""
        value = obj.get(key)
        return value if isinstance(value, dict) else {}

    game, scene, final = section("game"), section("scene"), section("final_scene_output")

    return Scene(
        # Die "or"-Ketten sind Rueckfallebenen: nimm final_scene_output, sonst
        # scene, sonst leer. Manche Modelle fuellen nur eins der beiden.
        # str(...) stellt sicher, dass wirklich Text herauskommt, auch wenn
        # das Modell dort eine Zahl oder eine Liste abgelegt hat.
        narration=str(final.get("narrator_text")
                      or scene.get("narrator_text") or "").strip(),
        visual=str(final.get("visual_scene_description")
                   or scene.get("visual_scene_description")
                   or scene.get("visual_prompt") or "").strip(),
        raw=text,
        number=_int(game.get("scene_number"), fallback_number),
        max_scenes=_int(game.get("max_scenes"), MAX_SCENES),
        # DIES ist das Spielende-Signal aus game_prompt.txt Abschnitt 38.
        # .lower() faengt "Completed" und "COMPLETED" mit ab.
        completed=str(game.get("status", "")).lower() == "completed",
    )

