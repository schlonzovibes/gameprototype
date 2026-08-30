"""Sprachmodell-Backends: reden mit Ollama und vLLM.

=== Die Aufgabenteilung ===

    game_prompts/   tragen die WELT - wie die Geschichte funktioniert
    schema.py       traegt die FORM - was eine gueltige Antwort ist
    diese Datei     bringt beides zum Server und holt die Antwort ab

Ein Spielsystem laesst sich damit austauschen, ohne den Client zu
zerbrechen - und umgekehrt.

=== Warum hier kein Parser mehr steht ===

Frueher gab es parse_scene(): eine Funktion, die aus freiem Modelltext mit
Rueckfallketten und Notfallpfaden eine Szene zu retten versuchte. Sie war
noetig, weil das Format nur ERBETEN war ("antworte in JSON"), nicht
erzwungen - gueltiges JSON war garantiert, das richtige nicht.

Jetzt geht das Schema als GRAMMATIK an den Server. Er uebersetzt es in einen
Automaten und maskiert bei jedem Schritt die Tokens weg, die zu einer
ungueltigen Antwort fuehren wuerden. Ein falsches Feld ist damit nicht
unwahrscheinlich, sondern unmoeglich. Was bleibt, ist Validierung - und die
macht Pydantic.

=== Was hier drin ist ===

    LLM         gemeinsame Basisklasse - definiert, was ein Backend koennen muss
    Ollama      spricht mit dem Ollama-Server auf dem Host
    VLLM        spricht mit dem vLLM-Server im zweiten Container, und haelt
                ihn am Leben (starten, ueberwachen, entladen)

Python-Konzepte hier: Vererbung, abstrakte Methoden, reguläre Ausdruecke,
Exception-Ketten (raise ... from) und HTTP-Fehlerbehandlung.
"""

from __future__ import annotations

import json
import os
import re                # regulaere Ausdruecke: Muster in Text suchen
import shlex             # Argumente sicher fuer eine Shell-Zeile quoten
import time
import urllib.error
import urllib.request

from models import OLLAMA_URL, VLLM_URL, Model

# Kontextfenster = wie viel Text das Modell gleichzeitig "sehen" kann,
# gemessen in Tokens (grob: Wortteile). Zu klein -> das Modell vergisst den
# Anfang der Geschichte, oder der Aufruf scheitert ganz.
#
# Ein Aufruf ist STATELESS: genau zwei Nachrichten (System-Prompt +
# gerenderter Weltzustand), KEIN mitwachsender Chatverlauf - die Kontinuitaet
# traegt world.render() im User-Teil. Grobe Groesse pro Aufruf:
#     ~2000   groesster Prompt (resolve_player.txt) + Feldbeschreibung
#     ~1500   gerenderter Weltzustand gegen Ende einer Geschichte
#   ~24576   Platz fuer Denkprozess + Antwort (MAX_TOKENS)
#   -------
#   ~28000   - also selbst mit reichlich Reserve weit unter 65536
#
# Warum trotzdem so grosszuegig: bei NVFP4 sind die Gewichte klein und mit
# --kv-cache-dtype fp8 kostet ein Token KV-Cache nur die Haelfte - der Platz
# ist praktisch geschenkt (im Log lag die KV-Cache-Auslastung bei <1 %).
# Gilt auch fuer Ollama (num_ctx).
NUM_CTX = int(os.environ.get("AIGAME_NUM_CTX", "65536"))

# Wie viele Tokens die Antwort hoechstens lang sein darf.
#
# Im Thinking-Modus muessen hier ZWEI Dinge hineinpassen: erst der
# Denkprozess, dann das vollstaendige Zug-JSON. Reisst das VOR dem JSON ab
# (finish_reason=length), war der content aus vLLM buchstaeblich leer - nicht
# kaputtes JSON, sondern gar keins - und das Spiel meldete "vLLM returned an
# empty response (finish_reason=length)".
#
# Der haeufigere Grund dafuer ist aber KEIN zu kleines Budget, sondern ein
# Denkprozess, der sich verrennt und im Kreis dreht, bis er das Budget - egal
# wie gross - auffrisst. Dagegen helfen TEMPERATURE (niedriger = terminiert
# eher) und der Wiederholungsversuch in structured(), nicht ein noch
# groesseres MAX_TOKENS. 24576 ist die Obergrenze fuer einen EHRLICH langen
# Denkprozess; wer es hoeher dreht, kaschiert nur eine Schleife.
#
# Betrifft nur vLLM: der Ollama-Payload setzt kein num_predict, dort ist die
# Antwortlaenge ohnehin nur durch NUM_CTX begrenzt - deshalb lief genau
# dasselbe Modell auf Ollama anstandslos und auf vLLM mit "empty response".
MAX_TOKENS = int(os.environ.get("AIGAME_MAX_TOKENS", "24576"))

# Sampling-Temperatur fuer BEIDE Backends. 0.9 war fuer die Erzaehlung bewusst
# kreativ gewaehlt - im Thinking-Modus ist das aber zu hoch: der Denkprozess
# findet dann schwerer zu einem Schluss und verrennt sich, bis MAX_TOKENS
# reisst (siehe oben). 0.6 ist Qwens eigene Empfehlung fuer den Thinking-Modus
# und laesst die Erzaehlung immer noch lebendig genug.
TEMPERATURE = float(os.environ.get("AIGAME_TEMPERATURE", "0.6"))

# Presence-Penalty: flacher Malus auf jedes Token, das im Kontext schon
# einmal vorkam. Qwen3 nennt genau das als Mittel gegen "endlose
# Wiederholungen" im Thinking-Modus - und exakt das war der Fehlerfall:
# der Denkprozess drehte sich in "Let's go with X... or Y... actually Z..."
# im Kreis, bis MAX_TOKENS riss und der content leer blieb
# ("finish_reason=length"). 1.5 ist Qwens Vorschlag, wenn man Schleifen
# sieht; deutlich hoeher kann Sprachmischung ausloesen. 0 schaltet es ab.
PRESENCE_PENALTY = float(os.environ.get("AIGAME_PRESENCE_PENALTY", "1.5"))

# top_k / min_p: der Rest von Qwens empfohlenem Sampling fuer den Thinking-
# Modus (top_k 20, min_p 0). Schneidet den unwahrscheinlichen Schwanz weg,
# an dem sich der Denkprozess sonst festhakt. Nur vLLM - Ollama nimmt beide
# ueber options entgegen, aber der Fehlerfall lag ausschliesslich bei vLLM.
TOP_K = int(os.environ.get("AIGAME_TOP_K", "20"))

# Zuletzt gemessene Generierungsrate (Tokens/Sekunde) der juengsten
# complete()-Anfrage - nur fuer die Footer-Anzeige (ui.py liest das). None =
# es lief noch keine Inferenz. structured() ruft complete() ggf. mehrfach
# auf; jeder Aufruf ueberschreibt, der Footer zeigt also stets die letzte.
last_tokens_per_sec: float | None = None


def _record_rate(tokens: int | None, seconds: float | None) -> None:
    """Rate ablegen, wenn beide Werte brauchbar sind - sonst unveraendert."""
    global last_tokens_per_sec
    if tokens and seconds and seconds > 0:
        last_tokens_per_sec = tokens / seconds


# Reasoning-Modus: manche Modelle koennen vor der Antwort "nachdenken".
# Das kostet Zeit, verbessert aber oft die Konsistenz. Der Denktext landet
# getrennt vom JSON und wandert nur ins Debug-Log.
# Der Vergleich mit einem Tupel erlaubt mehrere Schreibweisen in der env-Var.
THINK = os.environ.get("AIGAME_THINK", "0").lower() in ("1", "true", "on", "yes")

# Anteil des GPU-Speichers, den vLLM fuer sich reservieren darf - Gewichte
# UND KV-Cache zusammen. Auf dem DGX Spark (128 GB gemeinsamer Speicher)
# entsprechen 0.65 rund 83 GB - der Wert aus dem Arena-Rezept (siehe
# VLLM_EXTRA_ARGS unten).
#
# Zu niedrig ist gefaehrlicher als zu hoch: vLLM laedt die Gewichte trotzdem
# und scheitert erst danach beim KV-Cache - man sieht den Speicher volllaufen
# und dann komplett freigegeben werden. Bei 0.55 waeren es nur ~70 GB, also
# weniger als ein 80-GB-Modell allein schon braucht.
#
# Nach oben begrenzt das Bildmodell: es wird NACH dem Sprachmodell geladen
# und muss in den Rest passen. 0.65 laesst dafuer ~45 GB - und mit
# --kv-cache-dtype fp8 (VLLM_EXTRA_ARGS) reicht das kleinere Budget dem
# KV-Cache trotzdem locker.
VLLM_GPU_UTIL = os.environ.get("AIGAME_VLLM_GPU_UTIL", "0.65")

# Zusaetzliche Flags fuer 'vllm serve', als eine Shell-Zeile. shlex.split()
# in _serve() zerlegt sie wie eine echte Shell (Anfuehrungszeichen bleiben
# also erhalten - wichtig fuer das JSON in --speculative-config).
#
# Der Default ist das Durchsatz-Rezept aus der DGX-Spark-Arena
# (spark-arena.com) fuer nvidia/Qwen3.6-35B-A3B-NVFP4 - die Kombination mit
# der hoechsten Token-Rate auf dem GB10:
#
#   --kv-cache-dtype fp8            halbiert den KV-Cache
#   --attention-backend flashinfer  schnellster Attention-Kernel auf GB10
#   --moe-backend marlin           NVFP4-MoE ueber den Marlin-Kernel
#   --speculative-config mtp       Multi-Token-Prediction, 3 Tokens vorab
#   --load-format fastsafetensors  schnelleres Laden der Shards
#   --async-scheduling             Scheduler laeuft neben der GPU-Arbeit
#   --enable-chunked-prefill       langer Prefill in Haeppchen
#   --max-num-seqs / --max-num-batched-tokens   Batch-Fenster
#   --enable-prefix-caching        spart den Prefill des ueber den ganzen
#                                  Lauf IDENTISCHEN ~6500-Token-System-
#                                  Prompts in JEDER Runde - da kein
#                                  Chatverlauf mitwaechst, ist der cachebare
#                                  Anteil in Szene 15 so gross wie in Szene 1
#
# --reasoning-parser gehoert NICHT hierher (dafuer AIGAME_REASONING_PARSER),
# sonst haengt _serve() das Flag doppelt an.
#
# Kennt ein vLLM-Build ein Flag nicht, weigert sich 'vllm serve' zu starten
# ("unrecognized arguments") - dann die betroffene Option streichen (in
# docker-compose.yml AIGAME_VLLM_ARGS). Der Fehler steht dank der Logdatei
# sofort in der Meldung, statt erst nach dem Timeout aufzufallen.
VLLM_EXTRA_ARGS = os.environ.get(
    "AIGAME_VLLM_ARGS",
    "--trust-remote-code --kv-cache-dtype fp8 --attention-backend flashinfer "
    "--moe-backend marlin --max-num-seqs 4 --max-num-batched-tokens 32768 "
    "--enable-chunked-prefill --async-scheduling --enable-prefix-caching "
    "--load-format fastsafetensors "
    "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3,"
    "\"moe_backend\":\"triton\"}'")

# Name des Reasoning-Parsers von vLLM, passend zum Modell (qwen3,
# deepseek_r1, ...). Leer lassen, wenn ohne Thinking gefahren wird.
#
# WARUM DAS ZUSAMMEN MIT DER GRAMMATIK ZWINGEND IST: ohne diesen Parser
# greift die Grammatik ab dem ALLERERSTEN Token. Das Modell muesste also
# sofort mit "{" beginnen - ein <think>-Block waere damit unmoeglich, und
# der Denkprozess verschwaende ersatzlos, obwohl AIGAME_THINK gesetzt ist.
#
# Mit dem Parser trennt vLLM den Denkblock ab, liefert ihn in
# reasoning_content, und zwingt erst den Text NACH </think> in die
# Grammatik. Das Modell darf frei denken und muss danach exakt antworten.
VLLM_REASONING_PARSER = os.environ.get("AIGAME_REASONING_PARSER", "")

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

# Ein vorkompiliertes Suchmuster. Findet <think>...</think> samt Inhalt.
#   (.*?)      merkt sich, was dazwischen steht ("Gruppe 1")
#   re.DOTALL  laesst den Punkt auch Zeilenumbrueche treffen
# Einmal kompiliert statt bei jedem Aufruf neu - das ist schneller.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class EmptyResponse(RuntimeError):
    """Das Backend lieferte einen leeren content - meist ein Denkprozess,
    der sich verrannt und das Token-Budget aufgefressen hat, bevor das JSON
    kam. Eigene Klasse, damit structured() das von einem Schema-Fehler
    unterscheiden und einmal - mit einer Ermahnung zur Kuerze - wiederholen
    kann, statt hart abzubrechen."""


class LLM:
    """Basisklasse: was jedes Sprachmodell-Backend koennen muss.

    Ollama und VLLM "erben" von dieser Klasse (class Ollama(LLM)) und
    uebernehmen dabei alles, was hier steht. Sie muessen nur complete()
    selbst ausfuellen - der Rest gilt fuer beide gleichermassen.

    Der Sinn: story.py muss nicht wissen, welches Backend laeuft. Es ruft
    .structured() auf und bekommt ein validiertes Objekt - egal ob dahinter
    Ollama oder vLLM steckt.
    """

    last_thinking: str = ""   # Reasoning-Text des letzten complete()-Aufrufs

    def load(self, progress=None) -> None:  # pragma: no cover
        """Modell bereitstellen.

        progress ist optional und wird - wenn uebergeben - waehrend des
        Ladens mit (anteil, beschriftung) aufgerufen. anteil ist 0.0 bis 1.0
        oder None, wenn gerade kein Fortschritt bekannt ist. Passt genau auf
        ui.Status.update(), sodass main.py einfach die Methode durchreicht.
        """
        raise NotImplementedError

    def complete(self, messages: list[dict],
                 schema: dict | None = None) -> str:  # pragma: no cover
        """Muss jede Unterklasse selbst implementieren.

        schema ist ein JSON-Schema. Ist es gesetzt, wird die Antwort nicht
        erbeten, sondern ERZWUNGEN: der Server baut daraus einen Automaten
        und maskiert bei jedem Generierungsschritt die Tokens weg, die zu
        einer ungueltigen Antwort fuehren wuerden.

        Wichtig fuer das Verstaendnis: das Schema geht als eigener
        Request-Parameter mit, NICHT im Prompt. Es kostet also keine Tokens
        und stoert den Prefix-Cache nicht - der System-Prompt bleibt
        byteweise identisch, egal welches Schema gerade gilt.

        NotImplementedError ist das uebliche Signal fuer "hier fehlt noch
        etwas" - wer von LLM erbt und das vergisst, merkt es sofort.
        """
        raise NotImplementedError

    def structured(self, messages: list[dict], model_cls, retries: int = 1):
        """Fragen und ein VALIDIERTES Pydantic-Objekt zurueckbekommen.

        Der einzige Weg, auf dem story.py mit einem Modell spricht. Was hier
        herauskommt, hat Grammatik und Validierung passiert - der Aufrufer
        muss nichts mehr pruefen und nichts mehr retten.

        Warum trotz Grammatik noch ein Reparaturversuch? Weil nicht jedes
        Backend sie beherrscht: ein aelteres Ollama ignoriert das Schema und
        liefert freies JSON. Bei vLLM greift dieser Zweig nie - er ist die
        Rueckfallebene fuer den Fall, dass die Zusage nicht eingehalten wird.

        Der Reparaturversuch haengt die fehlerhafte Antwort UND den
        Validierungsfehler an den Verlauf. Das Modell sieht damit genau,
        woran es gescheitert ist - eine blosse Wiederholung derselben Frage
        wuerde meist denselben Fehler erzeugen.
        """
        # Spaeter Import: schema.py zieht pydantic, und llm.py wird auch
        # dort geladen, wo nur die Modellverwaltung gebraucht wird.
        from pydantic import ValidationError

        schema = model_cls.model_json_schema()
        attempt = list(messages)

        for remaining in range(retries, -1, -1):
            try:
                raw = self.complete(attempt, schema)
            except EmptyResponse as e:
                # Leerer content - fast immer ein Denkprozess, der sich
                # verrannt hat. Ein blosses Wiederholen liefe genauso; wir
                # haengen deshalb eine ausdrueckliche Ermahnung zur Kuerze an
                # und lassen die naechste Runde damit laufen.
                if remaining == 0:
                    raise RuntimeError(
                        f"Model gave an empty answer after "
                        f"{retries + 1} attempts ({e}).") from None
                attempt = attempt + [
                    {"role": "user",
                     "content": "Your previous reply was empty - the "
                                "reasoning ran too long. Think briefly, then "
                                "output only the JSON object now."},
                ]
                continue
            try:
                return model_cls.model_validate_json(_json_slice(raw))
            except ValidationError as e:
                if remaining == 0:
                    raise RuntimeError(
                        f"Model output did not match the schema after "
                        f"{retries + 1} attempts:\n{e}") from None
                attempt = attempt + [
                    {"role": "assistant", "content": raw},
                    {"role": "user",
                     "content": "That response was rejected by the schema "
                                f"validator:\n{e}\n\nAnswer again. Return "
                                "only the corrected object."},
                ]

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
        """Den Denkprozess aus dem Content loesen.

        Mit --reasoning-parser trennt vLLM das Denken schon serverseitig ab
        (reasoning_content) und content ist blankes JSON - dann tut diese
        Methode nichts. Die drei Faelle hier sind die Rueckfallebene fuer den
        Fall, dass doch ein <think> in den content durchsickert: bei Ollama,
        bei einem Parser-Aussetzer oder wenn der Denkprozess ins Token-Limit
        lief. Der Text wandert nach self.last_thinking und von dort ins Log.
        """
        # Fall 1: vollstaendiger Block <think>...</think>.
        match = _THINK_RE.search(content)
        if match:
            block = match.group(1).strip()   # Gruppe 1 = das (.*?) im Muster
            # An vorhandenen Denktext anhaengen (manche Backends liefern
            # beides: ein eigenes Feld UND einen Inline-Block). strip() raeumt
            # danach den fuehrenden Umbruch weg, falls last_thinking leer war.
            self.last_thinking = f"{self.last_thinking}\n{block}".strip()
            # sub() ersetzt das Gefundene durch "" - count=1 nur das erste Mal.
            return _THINK_RE.sub("", content, count=1).strip()

        # Fall 2: nur ein schliessendes </think>, kein oeffnendes. Qwens
        # Chat-Vorlage legt das oeffnende <think> schon in den Prompt - im
        # generierten Text steht dann alles VOR dem ersten </think> als
        # Denkprozess, ohne Tag davor.
        head, sep, rest = content.partition("</think>")
        if sep:
            self.last_thinking = f"{self.last_thinking}\n{head.strip()}".strip()
            return rest.strip()

        # Fall 3: ein oeffnendes <think> ohne Abschluss - der Denkprozess lief
        # ins Token-Limit, das JSON kam nie. Alles ist Denktext; "" macht
        # daraus in complete() eine saubere EmptyResponse zum Wiederholen,
        # statt den halben Gedankengang als Szene anzuzeigen.
        if content.lstrip().startswith("<think>"):
            self.last_thinking = f"{self.last_thinking}\n{content}".strip()
            return ""

        return content     # nichts gefunden - unveraendert zurueck


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

    def complete(self, messages: list[dict],
                 schema: dict | None = None) -> str:
        """Den Verlauf schicken und die Antwort als Text zurueckbekommen."""
        # Ein dict ist Pythons Woerterbuch: Schluessel -> Wert. Das hier wird
        # gleich zu JSON und geht so an den Server.
        payload = {
            "model": self.name,
            "messages": messages,
            "stream": False,             # alles auf einmal, nicht Stueck fuer Stueck
            # Frueher stand hier fest "json" - das erzwang GUELTIGES JSON,
            # aber nicht das RICHTIGE. Ein Schema ist die staerkere Zusage:
            # es legt die Felder selbst fest. Ohne Schema bleibt es beim
            # blossen "json", damit ein Aufruf ohne Modellklasse weiterhin
            # brauchbar antwortet.
            "format": schema if schema else "json",
            "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
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

        # Ollama misst selbst: eval_count Tokens in eval_duration Nanosekunden
        # (reine Generierung, ohne Prompt-Verarbeitung) - genauer als eine
        # eigene Wall-Clock-Messung.
        dur = data.get("eval_duration")
        _record_rate(data.get("eval_count"), dur / 1e9 if dur else None)

        message = data.get("message", {})
        self.last_thinking = message.get("thinking") or ""
        content = self._split_thinking(message.get("content", "").strip())

        if not content:
            raise EmptyResponse(f"Ollama lieferte eine leere Antwort von "
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

    @staticmethod
    def check_requirements() -> None:
        """Prueft, ob dieses Backend ueberhaupt betriebsbereit ist.

        Nur das Python-Paket - der Container selbst wird NICHT geprueft.
        Der laeuft womoeglich noch nicht, wenn compose ihn gerade erst
        gestartet hat, und das Spiel wartet beim Laden ohnehin auf ihn.

        Absichtlich statisch (@staticmethod): main.py ruft das VOR der
        Modellauswahl auf, es existiert also noch gar kein VLLM-Objekt.

        Warum ueberhaupt vorab? Fehlt das Paket im Image (siehe
        _require_docker_package), faellt das sonst erst auf, nachdem der
        Spieler ein Modell ausgesucht hat.
        """
        _require_docker_package()

    def _container(self):
        """Das Docker-Objekt des vLLM-Containers holen."""
        docker = _require_docker_package()
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
        ]
            # Nur anhaengen, wenn gesetzt - ein leerer Wert waere ein
            # ungueltiges Argument, kein "kein Parser".
            + (["--reasoning-parser", VLLM_REASONING_PARSER]
               if VLLM_REASONING_PARSER else [])
            + shlex.split(VLLM_EXTRA_ARGS))
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

    def complete(self, messages: list[dict],
                 schema: dict | None = None) -> str:
        payload = {
            "model": self.name,
            "messages": messages,
            "temperature": TEMPERATURE,
            # Qwens vollstaendige Sampling-Empfehlung fuer den Thinking-Modus:
            # top_p 0.95, top_k 20 - schneiden den unwahrscheinlichen Schwanz
            # weg, an dem sich der Denkprozess verheddert. presence_penalty
            # ist Qwens explizites Gegenmittel gegen endlose Wiederholungen
            # (siehe PRESENCE_PENALTY oben) - der eigentliche Hebel gegen den
            # "finish_reason=length"-Fehlerfall.
            "top_p": 0.95,
            "top_k": TOP_K,
            "presence_penalty": PRESENCE_PENALTY,
            "max_tokens": MAX_TOKENS,
        }
        if schema:
            # strict=True heisst: keine zusaetzlichen Felder, keine
            # Abweichung. vLLM uebersetzt das Schema in eine Grammatik
            # (xgrammar) und maskiert waehrend der Generierung alle Tokens
            # weg, die daraus herausfuehren wuerden.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema,
                                "strict": True},
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            data = _json_post(VLLM_URL + "/v1/chat/completions", payload, timeout=600)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace").strip()
            raise RuntimeError(f"vLLM: HTTP {e.code} - {detail[:300]}") from None

        # vLLM liefert keine Timing-Felder - Wall-Clock ist die einzige Quelle.
        # completion_tokens zaehlt auch die Reasoning-Tokens mit; als grober
        # Durchsatzwert fuer die Fussleiste ist das genau richtig.
        _record_rate((data.get("usage") or {}).get("completion_tokens"),
                     time.perf_counter() - t0)

        # Hier ohne .get(): fehlt "choices", ist die Antwort so kaputt, dass
        # ein lauter Fehler ehrlicher ist als ein stiller Ersatzwert.
        choice = data["choices"][0]
        message = choice["message"]
        # Das abgetrennte Denken steht je nach vLLM-Version/Parser in
        # "reasoning_content" (DeepSeek-Konvention) ODER "reasoning" (0.27er-
        # Build mit --reasoning-parser qwen3). Beide pruefen, sonst faellt der
        # Denktext still unter den Tisch und das [THINKING]-Log bleibt leer.
        self.last_thinking = (message.get("reasoning_content")
                              or message.get("reasoning") or "")
        content = self._split_thinking((message.get("content") or "").strip())

        if not content:
            # finish_reason mit in die Meldung: "length" heisst, MAX_TOKENS
            # wurde erreicht, bevor etwas ausserhalb von reasoning_content
            # stand - der Denkprozess hat sich also verrannt. structured()
            # faengt EmptyResponse und wiederholt einmal mit einer Ermahnung
            # zur Kuerze; last_thinking traegt den Rattenschwanz fuers Log.
            reason = choice.get("finish_reason", "unknown")
            raise EmptyResponse(
                f"vLLM returned an empty response from {self.name} "
                f"(finish_reason={reason}).")
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


def _require_docker_package():
    """Das Python-Paket 'docker' importieren - oder verstaendlich scheitern.

    An einer Stelle statt an zweien, damit die Vorabpruefung
    (VLLM.check_requirements) und der spaetere echte Zugriff (_container)
    garantiert denselben Text zeigen. Zwei getrennte Formulierungen desselben
    Problems laufen sonst frueher oder spaeter auseinander.

    Das Paket steckt heute IM IMAGE (siehe build.dockerfile_inline in
    docker-compose.yml) statt per pip bei jedem Containerstart nachinstalliert
    zu werden - das war frueher so, ist es aber nicht mehr. Fehlt es
    trotzdem, ist meist das Image veraltet: es wurde nach einer Aenderung an
    dockerfile_inline nicht neu gebaut.
    """
    try:
        import docker
    except ImportError:
        raise RuntimeError(
            "Python package 'docker' is missing - the vLLM backend needs it "
            "to start the server in the neighbouring container.\n\n"
            "Fix it inside this container with:  pip install docker\n\n"
            "It is baked into the image at build time (see "
            "build.dockerfile_inline in docker-compose.yml), so this usually "
            "means the image is out of date - rebuild it with:  "
            "docker compose build") from None
    return docker


def _json_slice(raw: str) -> str:
    """Aus einer Antwort das JSON-Objekt herausschneiden.

    Guertel UND Hosentraeger: greift die Grammatik, ist hier nichts zu tun -
    die Antwort ist dann bereits ein blankes JSON-Objekt. Ein Ollama-Backend
    ohne Schema-Unterstuetzung liefert aber freies JSON, mitunter in
    ```-Bloecken oder mit einem Satz davor. Ohne diesen Schnitt faellt es
    dort hart aus, wo es sich noch retten liesse.

    Zwei Schritte: Code-Fences abstreifen, dann von der ersten "{" bis zur
    letzten "}" schneiden.
    """
    text = raw.strip()

    if text.startswith("```"):
        # strip("`") entfernt alle Backticks an beiden Enden; das Abtrennen
        # der ersten Zeile wirft die Sprachangabe weg (meist "json").
        text = text.strip("`").split("\n", 1)[-1]

    start, end = text.find("{"), text.rfind("}")
    # find() liefert -1, wenn nichts da ist. Dann geben wir den Text
    # unveraendert zurueck und lassen die Validierung den Fehler melden -
    # sie formuliert ihn besser, als wir es hier koennten.
    return text[start:end + 1] if start != -1 and end > start else text


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
