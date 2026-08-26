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


# ------------------------------------------------------- statisch: INIT

def init_model() -> type[BaseModel]:
    """Der einmalige Weltaufbau - jetzt nur noch EIN Raum, keine Figuren.

    Frueher erzeugte INIT einen vollstaendigen Graphen (7-9 Knoten) samt
    Charakteren VOR dem ersten Spielzug - eine im Voraus geplante
    Levelgeometrie, die ein Sprachmodell danach nur noch abschreitet.
    Playtesting zeigte: entweder haelt sich das Modell daran (dann war das
    Sprachmodell fuer die Navigation ueberfluessig), oder es weicht ab (dann
    war der Graph verschwendete Arbeit).

    Jetzt entsteht die Welt WAEHREND des Spielens: INIT liefert nur den
    Startraum, jeder weitere Knoten entsteht als Nebenprodukt von
    resolve_model()s new_room-Feld, sobald die Erzaehlung ihn braucht (siehe
    dort). Der Startraum bekommt seine Id ("n1") vom Client
    (state.World.from_init), nie vom Modell - deshalb kein id-Feld hier.

    Parameterlos wie narrate_model() - fuer Namenssymmetrie ueber alle vier
    Aufruftypen (init/decide/resolve/narrate), auch wenn hier nichts
    Dynamisches injiziert werden muss.
    """
    class Init(BaseModel):
        model_config = STRICT

        language: str = Field(
            description="BCP-47 tag of the language the player wrote the "
                        "start prompt in, e.g. de or en. Every player-facing "
                        "text in this game must use it.")
        start_node_name: str = Field(
            description="short place name, in the language of the start "
                        "prompt")
        start_node_anchor: str = Field(
            description="permanent physical description in English. Only "
                        "things that can be named and touched, and that "
                        "will still be true in twenty scenes. No mood, no "
                        "events, no people.")
        opening_narration: str = Field(
            description="scene 1, in the language of the start prompt, "
                        "second person, 60-120 words. No player action has "
                        "happened yet, so nothing here may react to one.")
        opening_image_prompt: str = Field(
            description="English image description of the opening scene: "
                        "place, light, materials, perspective. Nameable "
                        "physical things only. No style words, no "
                        "negations, no action, no text.")

    return Init


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
                  actor_exits: tuple[str, ...],
                  mode: Literal["player", "agentic"]) -> type[BaseModel]:
    """Das Zustandsdelta EINES Akteurzuges, samt der Ereignisse, die er
    hinterlaesst - und, bei mode="player", der Raeume/Figuren, die dabei
    neu entstehen.

    ZWEI echte Klassen statt eines optionalen Feldes: nur der Spielerzug
    darf neue Figuren einfuehren (characters_introduced) - der agentische
    NPC ist raeumlich so frei wie der Spieler (new_room in beiden Modi),
    aber die Fuenf-Zuege-Garantie fuer neue Figuren haengt an dem, was der
    SPIELER erlebt, nicht am Nebenschauplatz eines NPC. "Optional" bliebe
    hier verboten (siehe Modul-Docstring "alle Felder Pflicht") - deshalb
    zwei Klassen, keine Optional[list]-Krücke.

    Anders als frueher schreibt dieser Aufruf keinen Erzaehltext mehr und
    faellt kein Urteil ueber das Spielende - beides zieht in narrate_model,
    das nur noch sieht, was der Spieler an seinem Ort wahrnehmen konnte.

    Alles Neue kommt ueber offene Listen herein (marks_added, facts_added,
    events, characters_introduced, ...). Das Schema selbst waechst dabei
    NIE - es beschreibt immer nur die Form eines Deltas, nie den Umfang der
    Welt. Deshalb bleibt die Grammatik in Szene 15 genauso klein wie in
    Szene 2 - GENAU SO wenig waechst sie mit der Anzahl der Knoten: neue
    Raeume erhoehen node_ids fuer den NAECHSTEN Aufruf, nicht dieses Schema.

    actor_exits sind die Ausgaenge des Knotens, an dem der AKTUELL handelnde
    Akteur GERADE steht (Spieler oder die eine agentische Figur) - anders
    als bei Move.to (andere Figuren stehen woanders, ein gemeinsames
    Literal waere dort nicht moeglich) kennen wir die Position des Akteurs
    beim Bauen der Grammatik bereits genau. actor_move_to bekommt deshalb
    sein eigenes, engeres Literal statt des allgemeinen MoveTo.
    """
    # Literal[()] wirft einen TypeError - ein leeres Tupel ist keine
    # gueltige Aufzaehlung. Gibt es keine Charaktere (ganz am Spielanfang,
    # oder alle tot), setzen wir einen Platzhalter ein, den es als Id nicht
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

    class NewRoom(BaseModel):
        model_config = STRICT
        # Pflichtfeld statt NewRoom | None: "kein neuer Raum" ist der leere
        # String bei name, exakt das Muster, das utterance/justification
        # schon nutzen ("kein Text" ist der leere String, kein None). Ein
        # echtes Optional-Feld waere die einzige Ausnahme von "alle Felder
        # Pflicht, kein Optional" in diesem Schema gewesen.
        name: str = Field(
            description="in the language of the start prompt. Empty string "
                        "if you are not proposing a new room this turn.")
        anchor: str = Field(
            description="permanent physical description in English, only "
                        "nameable/touchable things. Empty string if name is "
                        "empty.")
        one_way: bool = Field(
            description="true if the connection back to where you came from "
                        "is impossible")
        justification: str = Field(
            description="if one_way is true: the physical reason return is "
                        "impossible, in English. Height alone is not a "
                        "reason. Empty string otherwise.")

    # Feldreihenfolge ist Generierungsreihenfolge (siehe Modul-Docstring):
    # new_room ZUERST - er entscheidet, ob actor_move_to ueberhaupt einen
    # bestehenden Ausgang meint oder "stay" (die Ankunft in einem neuen Raum
    # laeuft NIE ueber actor_move_to - der neue Raum steht zum Zeitpunkt
    # dieses Aufrufs noch nicht in actor_exits, kann also in ActorMoveTos
    # Literal gar nicht auftauchen; new_room mit einem Namen IMPLIZIERT
    # deshalb selbst die Ankunft dort, siehe state.World.apply_turn).
    if mode == "player":
        class NewCharacter(BaseModel):
            model_config = STRICT
            name: str = Field(
                description="in the language of the start prompt")
            at: NodeId = Field(
                description="the node they appear at - an existing node, "
                            "normally the one you are standing in. Cannot "
                            "be a room you are proposing in this same call "
                            "(new_room) - that room does not exist until "
                            "after this call returns.")
            agenda_draft: str = Field(
                description="English, one sentence, physically checkable - "
                            "raw material for their agenda, not a finished "
                            "character sheet")
            agenda_target_hint: str = Field(
                description="English, one noun or short noun phrase: what "
                            "this agenda points toward. Used by the client "
                            "to decide things - you never see it again.")

        class ResolvePlayer(BaseModel):
            model_config = STRICT

            new_room: NewRoom = Field(
                description="a new room, or an empty-named one if you "
                            "propose none this turn. If you propose one, "
                            "its name cannot yet be an exit, so set "
                            "actor_move_to to 'stay' - arriving there "
                            "happens automatically.")
            actor_move_to: ActorMoveTo = Field(
                description="where the player ends up, or stay. Only an "
                            "exit of their current node - or 'stay' if you "
                            "are proposing a new room this turn.")
            moves: list[Move] = Field(
                description="OTHER characters moved as a side effect this "
                            "turn - not the player")
            status_changes: list[StatusChange] = Field(
                description="characters whose condition changes this turn")
            marks_added: list[Mark] = Field(
                description="lasting physical traces added to places")
            facts_added: list[str] = Field(
                description="short English clauses that became true of the "
                            "whole world and cannot be seen at a single "
                            "place")
            characters_introduced: list[NewCharacter] = Field(
                description="0 to 2 new characters this scene introduces")
            events: list[Event] = Field(
                description="1 to 4 short English physical clauses, each "
                            "tied to the node where it is visible or "
                            "audible - what actually happened this turn. "
                            "You do not decide who witnesses it, only "
                            "where and what.")

        return ResolvePlayer

    class ResolveAgentic(BaseModel):
        model_config = STRICT

        new_room: NewRoom = Field(
            description="a new room, or an empty-named one if you propose "
                        "none this turn. If you propose one, its name "
                        "cannot yet be an exit, so set actor_move_to to "
                        "'stay' - arriving there happens automatically.")
        actor_move_to: ActorMoveTo = Field(
            description="where you end up, or stay. Only an exit of your "
                        "current node - or 'stay' if you are proposing a "
                        "new room this turn.")
        moves: list[Move] = Field(
            description="OTHER characters moved as a side effect this turn "
                        "- not you, you use actor_move_to")
        status_changes: list[StatusChange] = Field(
            description="characters whose condition changes this turn")
        marks_added: list[Mark] = Field(
            description="lasting physical traces added to places")
        facts_added: list[str] = Field(
            description="short English clauses that became true of the "
                        "whole world and cannot be seen at a single place")
        events: list[Event] = Field(
            description="1 to 4 short English physical clauses, each tied "
                        "to the node where it is visible or audible - what "
                        "actually happened this turn. You do not decide who "
                        "witnesses it, only where and what.")

    return ResolveAgentic


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
