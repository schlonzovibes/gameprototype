
"""LLM-Backends und Szenen-Parsing.

Der Ausgabe-Contract liegt bewusst NICHT in game_prompt.txt, sondern hier:
die Datei traegt die Welt, der Code erzwingt die Form.
"""

from __future__ import annotations

import json
import os
import urllib.request

from discovery import OLLAMA_URL, Model

# Kontextfenster: game_prompt.txt allein braucht schon ~9K Tokens,
# dazu Start-Prompt, Verlauf und Generierungsraum.
NUM_CTX = int(os.environ.get("AIGAME_NUM_CTX", "24576"))

CONTRACT = """
AUSGABEFORMAT (verbindlich, keine Ausnahme):
Antworte ausschliesslich mit dem in Abschnitt 37 (REQUIRED OUTPUT FORMAT)
definierten einzelnen JSON-Objekt mit den vier Top-Level-Feldern
"game", "state_update", "scene", "player_agency".
Kein Vorwort, kein Nachwort, keine Code-Fences, keine Kommentare im JSON.

Feldregeln fuer die vom Client weiterverarbeiteten Felder:
- scene.visual_scene_description: englische Bildbeschreibung, 20-45 Woerter,
  nur Bildinhalt: Ort, Licht, Perspektive, Materialien, Atmosphaere.
  Keine Handlung, keine Sprache, kein Text im Bild.
- scene.narrator_text: deutscher Erzaehltext, 60-120 Woerter, zweite Person.
- scene.visual_prompt: englisch, direkt fuer ein Bildmodell nutzbar,
  konsistent mit visual_scene_description und den persistenten Charakteren.
""".strip()


class LLM:
    def scene(self, messages: list[dict]) -> dict:
        raw = self.complete(messages)
        return parse_scene(raw)

    def complete(self, messages: list[dict]) -> str:  # pragma: no cover
        raise NotImplementedError


class Ollama(LLM):
    def __init__(self, model: Model):
        self.name = model.ref

    def load(self) -> None:
        """Modell in den Speicher ziehen, ohne Text zu erzeugen."""
        self._post("/api/generate", {"model": self.name, "prompt": "",
                                     "keep_alive": "30m", "stream": False})

    def complete(self, messages: list[dict]) -> str:
        data = self._post("/api/chat", {
            "model": self.name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.9, "num_ctx": NUM_CTX},
        })
        # Ollama meldet Fehler manchmal mit HTTP 200 - dann steht
        # die Ursache im error-Feld und der Content bleibt leer.
        if data.get("error"):
            raise RuntimeError(f"Ollama: {data['error']}")
        content = data.get("message", {}).get("content", "").strip()
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
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)


class LlamaCpp(LLM):
    def __init__(self, model: Model):
        self.path = model.ref
        self.llm = None

    def load(self) -> None:
        from llama_cpp import Llama

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
        return out["choices"][0]["message"]["content"]


def build(model: Model) -> LLM:
    return {"ollama": Ollama, "llama-cpp": LlamaCpp}[model.backend](model)


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
            game = obj.get("game") if isinstance(obj.get("game"), dict) else {}
            visual = str(scene.get("visual_scene_description")
                         or obj.get("visual", "")).strip()
            narration = str(scene.get("narrator_text")
                            or obj.get("narration", "")).strip() or text
            return {
                "visual": visual,
                "narration": narration,
                "raw": text[start:end + 1],
                "completed": str(game.get("status", "")).lower() == "completed",
            }
    return {"visual": "", "narration": raw.strip(),
            "raw": raw.strip(), "completed": False}
