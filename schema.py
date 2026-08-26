"""Die einzige Quelle des Ausgabeformats.

=== Warum es diese Datei gibt ===

Frueher existierte der Ausgabe-Contract dreifach: als Prosa in llm.CONTRACT,
als "REQUIRED OUTPUT FORMAT"-Abschnitt in jeder Prompt-Datei, und implizit in
den Feldnamen des Parsers. Drei Orte fuer dieselbe Wahrheit laufen zwangs-
laeufig auseinander - und sie waren es bereits.

Hier steht das Format genau einmal. Daraus entstehen ZWEI Dinge:

    model_json_schema()  geht als Grammatik an den Server und erzwingt die
                         FORM - ungueltige Tokens werden gar nicht erst
                         erzeugt
    describe()           geht als Text in den Prompt und erklaert die
                         BEDEUTUNG - denn das Modell sieht die Grammatik nie

Beide stammen aus derselben Klassendefinition. Sie koennen nicht mehr
auseinanderlaufen.

=== Konventionen, die fuer JEDES Modell hier gelten ===

extra="forbid"     erfundene Felder sind ein Validierungsfehler, kein
                   stiller Zusatz
alle Felder Pflicht  kein Optional, kein None, keine Union ausser Literal.
                   "Kein Ziel" ist das Literal "stay", "kein Text" der leere
                   String. Ein fehlendes Feld waere ein Sonderfall im
                   Client; ein leerer Wert ist keiner.
Field(description=)  englisch, weil er im Prompt landet
Feldreihenfolge    ist Generierungsreihenfolge und damit Design. Das Modell
                   fuellt von oben nach unten, jedes Feld sieht die davor
                   bereits erzeugten. Deshalb steht das Zustandsdelta VOR
                   der Szene: der Erzaehltext wird auf einem schon
                   festgeschriebenen Delta konditioniert und kann ihm nicht
                   widersprechen.

=== Warum Ids als Literal und nicht als freier String ===

Die dynamischen Modelle werden pro Zug aus den TATSAECHLICH existierenden
Ids gebaut. Damit kann das Modell einen Knoten, den es nicht gibt, nicht
einmal buchstabieren - die Grammatik laesst die Tokens nicht zu. Das ist
staerker als jede Bitte im Prompt und billiger als jede Pruefung danach.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Gemeinsame Konfiguration. Als Modul-Konstante statt in jeder Klasse
# wiederholt - so kann keine Klasse sie versehentlich anders setzen.
STRICT = {"extra": "forbid"}

# Ids sind absichtlich einstellig: n1-n9, c1-c9. Das haelt sie kurz (ein
# Token je Id) und deckelt die Weltgroesse dort, wo sie noch ueberschaubar
# bleibt. Beim Init existieren die Ids noch nicht, deshalb dort ein Regex -
# in den dynamischen Modellen spaeter die echte Aufzaehlung.
NODE_ID = r"^n[1-9]$"
CHAR_ID = r"^c[1-9]$"


# ------------------------------------------------------- statisch: INIT

class InitExit(BaseModel):
    model_config = STRICT

    to: str = Field(
        pattern=NODE_ID,
        description="id of the node this exit leads to")
    one_way: bool = Field(
        description="true if this connection cannot be travelled back")
    justification: str = Field(
        description="if one_way is true: the physical reason why return is "
                    "impossible, in English. Height alone is not a reason - "
                    "state what blocks the way back. Empty string if the "
                    "connection is not one-way.")


class InitNode(BaseModel):
    model_config = STRICT

    id: str = Field(
        pattern=NODE_ID,
        description="unique id, n1 through n9")
    name: str = Field(
        description="short place name, in the language of the start prompt")
    anchor: str = Field(
        description="permanent physical description in English. Only things "
                    "that can be named and touched, and that will still be "
                    "true in twenty scenes. No mood, no events, no people.")
    exits: list[InitExit] = Field(
        description="connections leading out of this node")


class InitCharacter(BaseModel):
    model_config = STRICT

    id: str = Field(
        pattern=CHAR_ID,
        description="unique id, c1 through c9")
    name: str = Field(
        description="name, in the language of the start prompt")
    at: str = Field(
        pattern=NODE_ID,
        description="id of the node this character starts at")
    agenda: str = Field(
        description="one English sentence, physically checkable: the "
                    "world-state this character wants to bring about. This "
                    "is the only reason they will ever act, and it never "
                    "changes once play begins.")
    aim: str = Field(
        description="one English sentence: the first step toward the "
                    "agenda. During play the character replaces this every "
                    "round themselves - write only the opening move, not a "
                    "plan.")


class InitWorld(BaseModel):
    """Der einmalige Weltaufbau samt Eroeffnungsszene."""

    model_config = STRICT

    language: str = Field(
        description="BCP-47 tag of the language the player wrote the start "
                    "prompt in, e.g. de or en. Every player-facing text in "
                    "this game must use it.")
    nodes: list[InitNode] = Field(
        description="7 to 9 locations forming a connected graph")
    characters: list[InitCharacter] = Field(
        description="2 to 4 characters, none of them the player")
    facts: list[str] = Field(
        description="short English clauses that are true of the whole world "
                    "and cannot be seen at a single place")
    player_at: str = Field(
        pattern=NODE_ID,
        description="id of the node the player starts at")
    opening_narration: str = Field(
        description="scene 1, in the language of the start prompt, second "
                    "person, 60-120 words. No player action has happened "
                    "yet, so nothing here may react to one.")
    opening_image_prompt: str = Field(
        description="English image description of the opening scene: place, "
                    "light, materials, perspective. Nameable physical things "
                    "only. No style words, no negations, no action, no text.")


# ------------------------------------------------- dynamisch: pro Zug

def decide_model(node_ids: tuple[str, ...]) -> type[BaseModel]:
    """Was EIN Charakter als Naechstes will - aus seiner Sicht allein.

    Bewusst klein: vier Felder, kein Zustandsdelta, kein Erzaehltext. Diese
    Figur entscheidet nicht, ob ihr Vorhaben gelingt - das tut erst der
    Resolver, der den ganzen Zustand kennt. Sie will nur etwas.

    aim steht bewusst ZUERST, vor intent: die Figur setzt ihren naechsten
    Schritt neu (oder bestaetigt den alten), und der intent dieser Runde
    folgt DARAUS - nicht umgekehrt. Genau wie beim Zustandsdelta vor der
    Szene ist die Feldreihenfolge hier Design, nicht Zufall.
    """
    NodeId = Literal[node_ids]                  # type: ignore[valid-type]
    MoveTo = Literal[node_ids + ("stay",)]      # type: ignore[valid-type]

    class Decide(BaseModel):
        model_config = STRICT

        aim: str = Field(
            description="your next step toward your agenda, replacing your "
                        "previous one - English, one sentence")
        intent: str = Field(
            description="one short English clause: what you are trying to do "
                        "right now, following from the aim above. Physical "
                        "and checkable, not a feeling.")
        utterance: str = Field(
            description="what you say out loud, in the game language. Empty "
                        "string if you stay silent.")
        move_to: MoveTo = Field(
            description="an exit of the node you are standing in, or stay")

    # NodeId wird nur zur Herleitung von MoveTo gebraucht; die Zuweisung
    # haelt Linter davon ab, sie als unbenutzt zu melden.
    Decide.__doc__ = f"decision of one character among nodes {node_ids}"
    del NodeId
    return Decide


def resolve_model(node_ids: tuple[str, ...], char_ids: tuple[str, ...],
                  actor_exits: tuple[str, ...]) -> type[BaseModel]:
    """Das Zustandsdelta EINES Akteurzuges, samt der Ereignisse, die er
    hinterlaesst.

    Ein einziges Schema fuer Spieler- UND NPC-Zuege - wer gerade handelt,
    steht im Kontext (der Aufrufer baut die Nachricht entsprechend), nicht
    im Schema. Anders als frueher schreibt dieser Aufruf keinen Erzaehltext
    mehr und faellt kein Urteil ueber das Spielende - beides zieht in einen
    eigenen, spaeteren Aufruf (narrate_model), der nur noch das sieht, was
    der Spieler an seinem Ort wahrnehmen konnte.

    Alles Neue kommt ueber offene Listen herein (marks_added, facts_added,
    events, ...). Das Schema selbst waechst dabei NIE - es beschreibt immer
    nur die Form eines Deltas, nie den Umfang der Welt. Deshalb bleibt die
    Grammatik in Szene 15 genauso klein wie in Szene 2.

    actor_exits sind die Ausgaenge des Knotens, an dem der AKTUELL handelnde
    Akteur GERADE steht (Spieler oder eine Figur) - anders als bei Move.to
    (andere Figuren stehen woanders, ein gemeinsames Literal waere dort
    nicht moeglich) kennen wir die Position des Akteurs beim Bauen der
    Grammatik bereits genau. actor_move_to bekommt deshalb sein eigenes,
    engeres Literal statt des allgemeinen MoveTo: die Beschreibung sagt
    immer schon "nur ein Ausgang des aktuellen Knotens", die Grammatik kann
    es jetzt auch nicht mehr anders.
    """
    # Literal[()] wirft einen TypeError - ein leeres Tupel ist keine
    # gueltige Aufzaehlung. Gibt es keine Charaktere (alle tot, oder eine
    # Welt ohne NPCs), setzen wir einen Platzhalter ein, den es als Id nicht
    # gibt. Die Grammatik bleibt damit baubar; die Listen, die auf CharId
    # verweisen (moves, status_changes), muessen in diesem Fall leer
    # bleiben - etwas anderes koennte der Resolver gar nicht sinnvoll
    # fuellen, und World.apply_turn() wuerde es ohnehin ablehnen.
    char_ids = char_ids or ("none",)

    NodeId = Literal[node_ids]                  # type: ignore[valid-type]
    CharId = Literal[char_ids]                  # type: ignore[valid-type]
    MoveTo = Literal[node_ids + ("stay",)]      # type: ignore[valid-type]
    ActorMoveTo = Literal[actor_exits + ("stay",)]  # type: ignore[valid-type]
    Status = Literal["active", "disabled", "dead"]

    class Move(BaseModel):
        model_config = STRICT
        character: CharId = Field(description="who moves")
        to: NodeId = Field(
            description="target node, must be an exit of where they stand")

    class Mark(BaseModel):
        model_config = STRICT
        node: NodeId = Field(description="where the trace is left")
        clause: str = Field(
            description="short English physical clause describing a lasting "
                        "change to this place, e.g. 'the hatch stands open'. "
                        "Only what actually remains.")

    class StatusChange(BaseModel):
        model_config = STRICT
        character: CharId = Field(description="whose condition changes")
        status: Status = Field(
            description="active: acts normally. disabled: present but unable "
                        "to act. dead: gone for good.")

    class Event(BaseModel):
        model_config = STRICT
        node: NodeId = Field(
            description="where this is visible or audible")
        clause: str = Field(
            description="short English physical clause, checkable - what "
                        "happened, not how it felt or who noticed it")

    class Resolve(BaseModel):
        model_config = STRICT

        actor_move_to: ActorMoveTo = Field(
            description="where the acting party ends up, or stay. Only an "
                        "exit of their current node.")
        moves: list[Move] = Field(
            description="OTHER characters moved as a side effect this turn "
                        "- not the acting party, they use actor_move_to")
        status_changes: list[StatusChange] = Field(
            description="characters whose condition changes this turn")
        marks_added: list[Mark] = Field(
            description="lasting physical traces added to places")
        facts_added: list[str] = Field(
            description="short English clauses that became true of the whole "
                        "world and cannot be seen at a single place")
        events: list[Event] = Field(
            description="1 to 4 short English physical clauses, each tied to "
                        "the node where it is visible or audible - what "
                        "actually happened this turn. You do not decide who "
                        "witnesses it, only where and what.")

    return Resolve


def narrate_model() -> type[BaseModel]:
    """Was der Spieler zu lesen bekommt - erst NACHDEM alle Akteure dieser
    Runde aufgeloest sind.

    Statisch, ohne Ids: dieser Aufruf kennt keine Knoten- oder Figuren-Ids,
    weil er sie nicht braucht - er sieht nur noch die gefilterte
    Ereignisliste und den Ort des Spielers, beides bereits als Text (siehe
    story.Game._narrate). Das ist Absicht: es gibt so keinen Weg, versehentlich
    eine Id oder ein Feld zu referenzieren, das der Spieler nicht sehen darf.
    """
    class Narrate(BaseModel):
        model_config = STRICT

        can_end: bool = Field(
            description="true only if the situation has closed on its own. "
                        "This is a report, not a decision - whether the game "
                        "actually ends is decided elsewhere.")
        narrator_text: str = Field(
            description="the scene as the player reads it, in the game "
                        "language, second person, 60-120 words. Report only "
                        "what you were given - nothing else happened, as far "
                        "as the player is concerned.")
        image_prompt: str = Field(
            description="English image description: place, light, materials, "
                        "perspective. Nameable physical things only. No style "
                        "words, no negations, no action, no text.")

    return Narrate


# ------------------------------------------------------------- Zugriffe

def narration_text(narrate) -> tuple[str, str]:
    """(Erzaehltext, Bildprompt) einer erzaehlten Szene.

    Warum ein Zugriff statt narrate.narrator_text beim Aufrufer? Damit die
    FELDNAMEN des Ausgabeformats ausschliesslich in dieser Datei stehen.
    Genau darum ging der ganze Umbau: der Contract existierte frueher an
    drei Orten und lief auseinander. Ein Feldname, der auch nur in einer
    zweiten Datei auftaucht, ist der Anfang desselben Problems.

    Wer das Feld umbenennt, aendert es hier - und sonst nirgends.
    """
    return narrate.narrator_text, narrate.image_prompt


def opening_text(init) -> tuple[str, str]:
    """(Erzaehltext, Bildprompt) der Eroeffnungsszene - siehe narration_text()."""
    return init.opening_narration, init.opening_image_prompt


# ------------------------------------------------------------- describe

def describe(model_cls: type[BaseModel]) -> str:
    """Das Schema als eingerueckte Feldliste fuer den Prompt.

    Die Grammatik erzwingt die Form, aber das Modell BEKOMMT sie nie zu
    sehen - sie wirkt beim Server, nicht im Kontext. Ohne diese Beschreibung
    wuesste es zwar, dass ein Feld "anchor" existieren muss, aber nicht, was
    hineingehoert. Deshalb wandert die Ausgabe hier an den System-Prompt.

    Beides aus derselben Klasse: die Beschreibung kann nicht veralten,
    solange niemand sie von Hand woanders hinschreibt.

    Die Ausgabe ist deterministisch - dieselbe Klasse ergibt byteweise
    dieselbe Zeichenkette. Das ist keine Kosmetik: der System-Prompt muss
    ueber den ganzen Lauf identisch bleiben, sonst faellt der Prefix-Cache
    des Servers bei jedem Zug aus. Deshalb keine Mengen, keine Sortierung
    nach Zufall - die Feldreihenfolge kommt aus der Klassendefinition.
    """
    schema = model_cls.model_json_schema()
    return "\n".join(_field_lines(schema, schema.get("$defs", {}), 0))


def _resolve(node: dict, defs: dict) -> dict:
    """$ref durch das Zielmodell ersetzen.

    Pydantic legt verschachtelte Modelle unter "$defs" ab und verweist mit
    "$ref" darauf. Zusaetzlich kann AM VERWEIS eine eigene description
    stehen - die des Feldes. Die ist genauer als die des Zielmodells (sie
    beschreibt diese Verwendung, nicht die Klasse allgemein) und gewinnt
    deshalb.
    """
    if "$ref" not in node:
        return node

    target = dict(defs.get(node["$ref"].rsplit("/", 1)[-1], {}))
    for key, value in node.items():
        if key != "$ref":
            target[key] = value
    return target


def _type_label(node: dict, defs: dict) -> str:
    """Kurzer Typhinweis: "string", "array of object", "one of: n1, n2, stay"."""
    # Literal mit mehreren Werten wird zu "enum", mit genau einem zu "const".
    if "enum" in node:
        return "one of: " + ", ".join(str(v) for v in node["enum"])
    if "const" in node:
        return f"one of: {node['const']}"

    if node.get("type") == "array":
        return "array of " + _type_label(_resolve(node.get("items", {}), defs), defs)

    return node.get("type") or "object"


def _field_lines(node: dict, defs: dict, depth: int) -> list[str]:
    """Eine Zeile je Feld, verschachtelte Modelle eingerueckt darunter."""
    lines = []
    # dict behaelt in Python die Einfuegereihenfolge, und pydantic fuellt
    # properties in Felddefinitionsreihenfolge - die Ausgabe folgt also der
    # Generierungsreihenfolge, was fuer das Verstaendnis genau richtig ist.
    for name, raw in node.get("properties", {}).items():
        field = _resolve(raw, defs)

        # description am Feld hat Vorrang vor der des Zielmodells.
        text = raw.get("description") or field.get("description") or ""
        lines.append(f"{'  ' * depth}{name} ({_type_label(field, defs)})"
                     + (f" - {text}" if text else ""))

        # Bei Listen interessiert die Struktur der ELEMENTE, nicht die der
        # Liste - deshalb hier eine Ebene tiefer greifen.
        inner = field
        if field.get("type") == "array":
            inner = _resolve(field.get("items", {}), defs)
        if "properties" in inner:
            lines.extend(_field_lines(inner, defs, depth + 1))

    return lines
