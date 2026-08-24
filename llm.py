
"""LLM-Backends und Szenen-Parsing.

Der Ausgabe-Contract liegt bewusst NICHT in game_prompt.txt, sondern hier:
die Datei traegt die Welt, der Code erzwingt die Form.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from discovery import OLLAMA_URL, VLLM_URL, Model

# Kontextfenster: game_prompt.txt allein braucht schon ~9K Tokens,
# dazu Start-Prompt, Verlauf und Generierungsraum.
NUM_CTX = int(os.environ.get("AIGAME_NUM_CTX", "24576"))

# Reasoning-Modus: Denkprozess der Modelle einschalten (kostet Zeit,
# landet getrennt vom JSON in message.thinking bzw. <think>-Block).
THINK = os.environ.get("AIGAME_THINK", "0").lower() in ("1", "true", "on", "yes")

CONTRACT = """
AUSGABEFORMAT (verbindlich, keine Ausnahme):
Antworte ausschliesslich mit dem einzelnen JSON-Objekt aus dem Abschnitt
"REQUIRED OUTPUT FORMAT" mit den fuenf Top-Level-Feldern "game",
"state_update", "scene", "player_agency", "final_scene_output".
Kein Vorwort, kein Nachwort, keine Code-Fences, keine Kommentare im JSON.

Feldregeln fuer die vom Client weiterverarbeiteten Felder:
- final_scene_output.visual_scene_description: englische Bildbeschreibung,
  20-45 Woerter, nur Bildinhalt: Ort, Licht, Perspektive, Materialien,
  Atmosphaere. Keine Handlung, keine Sprache, kein Text im Bild.
- final_scene_output.narrator_text: deutscher Erzaehltext, 60-120 Woerter,
  zweite Person.
- scene.visual_prompt: englisch, direkt fuer ein Bildmodell nutzbar,
  konsistent mit der Bildbeschreibung und den persistenten Charakteren.
""".strip()


class LLM:
    last_thinking: str = ""   # Reasoning-Text des letzten complete()-Aufrufs

    def scene(self, messages: list[dict]) -> dict:
        raw = self.complete(messages)
        return parse_scene(raw)

    def complete(self, messages: list[dict]) -> str:  # pragma: no cover
        raise NotImplementedError


class Ollama(LLM):
    def __init__(self, model: Model):
        self.name = model.ref
        self.last_thinking = ""

    def load(self) -> None:
        """Modell in den Speicher ziehen, ohne Text zu erzeugen."""
        self._post("/api/generate", {"model": self.name, "prompt": "",
                                     "keep_alive": "30m", "stream": False})

    def complete(self, messages: list[dict]) -> str:
        payload = {
            "model": self.name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.9, "num_ctx": NUM_CTX},
        }
        if THINK:
            payload["think"] = True
        try:
            data = self._post("/api/chat", payload)
        except RuntimeError as e:
            # Standardmodelle ohne Reasoning-Support lehnen "think" mit
            # HTTP 400 ab - einmal ohne Denkprozess wiederholen.
            if not THINK or "does not support thinking" not in str(e).lower():
                raise
            payload.pop("think", None)
            data = self._post("/api/chat", payload)
        # Ollama meldet Fehler manchmal mit HTTP 200 - dann steht
        # die Ursache im error-Feld und der Content bleibt leer.
        if data.get("error"):
            raise RuntimeError(f"Ollama: {data['error']}")
        message = data.get("message", {})
        self.last_thinking = message.get("thinking") or ""
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError(
                f"Ollama lieferte eine leere Antwort von {self.name} "
                f"(num_ctx={NUM_CTX}).")
        return content

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            OLLAMA_URL + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # Ollama steckt die Ursache in den Body (z.B. "does not
            # support thinking") - ohne ihn bleibt nur "HTTP 400".
            detail = e.read().decode(errors="replace").strip()
            raise RuntimeError(
                f"Ollama: HTTP {e.code} - {detail[:300]}") from None


class LlamaCpp(LLM):
    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(self, model: Model):
        self.path = model.ref
        self.llm = None
        self.last_thinking = ""

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "Python package 'llama-cpp-python' is missing - GGUF models "
                "need it. Install inside the container with: "
                "CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native' "
                "FORCE_CMAKE=1 pip install llama-cpp-python") from None

        self.llm = Llama(
            model_path=self.path,
            n_ctx=NUM_CTX,
            n_gpu_layers=int(os.environ.get("AIGAME_GPU_LAYERS", "-1")),
            verbose=False,
        )

    def complete(self, messages: list[dict]) -> str:
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.9,
            max_tokens=int(os.environ.get("AIGAME_MAX_TOKENS", "2048")),
            response_format={"type": "json_object"},
        )
        content = out["choices"][0]["message"]["content"]
        # Denkmodelle geben <think>...</think> inline im Content aus.
        self.last_thinking = ""
        if THINK:
            match = self._THINK_RE.search(content)
            if match:
                self.last_thinking = match.group(1).strip()
                content = self._THINK_RE.sub("", content, count=1).strip()
        return content


class VLLM(LLM):
    """Client fuer den OpenAI-kompatiblen Endpoint eines vLLM-Servers.

    load() schaltet bei Bedarf selbst um: Der Compose-Container "vllm"
    laeuft idle (sleep infinity) und bekommt das gewaehlte Modell per
    Docker-exec eingeschaltet - siehe docker-compose.yml.
    """

    CONTAINER = os.environ.get("AIGAME_VLLM_CONTAINER", "vllm")
    LOAD_TIMEOUT = int(os.environ.get("AIGAME_VLLM_TIMEOUT", "900"))  # Sekunden

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(self, model: Model):
        self.name = model.ref   # HF-Repo-ID, z.B. "org/name" - muss dem
        self.last_thinking = ""  # Namen entsprechen, den vLLM serviert

    # ------------------------------------------------------------ Laden

    def load(self) -> None:
        """Server auf das gewaehlte Modell bringen und Start abwarten."""
        if self.name in self._served_models():
            return
        self._exec_serve()
        deadline = time.monotonic() + self.LOAD_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(5)
            if self.name in self._served_models():
                return
        raise RuntimeError(
            f"vLLM did not report {self.name} as served within "
            f"{self.LOAD_TIMEOUT // 60} minutes. Check 'docker logs {self.CONTAINER}'.")

    def _served_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(VLLM_URL + "/v1/models", timeout=3) as r:
                data = json.load(r)
        except (urllib.error.URLError, OSError, json.JSONDecodeError,
                TimeoutError, ValueError):
            return []
        return [m.get("id", "") for m in data.get("data", [])]

    def _exec_serve(self) -> None:
        try:
            import docker
        except ImportError:
            raise RuntimeError(
                "Python package 'docker' is missing - install it or start "
                f"vLLM manually: vllm serve {self.name} --port 8000") from None
        try:
            box = docker.from_env().containers.get(self.CONTAINER)
        except Exception as e:
            raise RuntimeError(
                f"Docker container '{self.CONTAINER}' is not reachable ({e}). "
                "Is the compose stack up and is the docker socket mounted?") from e
        # Alte serve-Instanz stoppen (Modellwechsel), VRAM freigeben lassen.
        try:
            box.exec_run(["sh", "-c",
                          "pkill -f 'vllm serve'; sleep 2"])
        except Exception:
            pass
        time.sleep(3)
        try:
            box.exec_run(
                ["vllm", "serve", self.name,
                 "--host", "0.0.0.0", "--port",
                 VLLM_URL.rsplit(":", 1)[-1],
                 "--gpu-memory-utilization",
                 os.environ.get("AIGAME_VLLM_GPU_UTIL", "0.55"),
                 "--max-model-len", str(NUM_CTX)],
                detach=True)
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
            "max_tokens": int(os.environ.get("AIGAME_MAX_TOKENS", "2048")),
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            VLLM_URL + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace").strip()
            raise RuntimeError(f"vLLM: HTTP {e.code} - {detail[:300]}") from None
        message = data["choices"][0]["message"]
        self.last_thinking = message.get("reasoning_content") or ""
        content = (message.get("content") or "").strip()
        # Denkmodelle geben <think>...</think> inline im Content aus.
        match = self._THINK_RE.search(content)
        if match:
            if self.last_thinking:
                self.last_thinking += "\n" + match.group(1).strip()
            else:
                self.last_thinking = match.group(1).strip()
            content = self._THINK_RE.sub("", content, count=1).strip()
        if not content:
            raise RuntimeError(
                f"vLLM returned an empty response from {self.name}.")
        return content


def build(model: Model) -> LLM:
    return {"ollama": Ollama, "llama-cpp": LlamaCpp,
            "vllm": VLLM}[model.backend](model)


def parse_scene(raw: str) -> dict:
    """Robustes Parsing. Faellt nie hart aus - die Szene muss weitergehen.

    Liefert: visual (Bildprompt-Basis), narration (Anzeigetext),
    raw (kanonisches JSON fuer den Chatverlauf), completed (Spielende).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            scene = obj.get("scene") if isinstance(obj.get("scene"), dict) else {}
            final = (obj.get("final_scene_output")
                     if isinstance(obj.get("final_scene_output"), dict) else {})
            game = obj.get("game") if isinstance(obj.get("game"), dict) else {}
            # Neu: final_scene_output. Fallbacks halten aeltere Formate lauffaehig.
            visual = str(final.get("visual_scene_description")
                         or scene.get("visual_scene_description")
                         or scene.get("visual_prompt")
                         or obj.get("visual", "")).strip()
            narration = str(final.get("narrator_text")
                            or scene.get("narrator_text")
                            or obj.get("narration", "")).strip() or text
            return {
                "visual": visual,
                "narration": narration,
                "raw": text[start:end + 1],
                "completed": str(game.get("status", "")).lower() == "completed",
            }
    return {"visual": "", "narration": raw.strip(),
            "raw": raw.strip(), "completed": False}
