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
    """Der einmalige Weltaufbau - die lokale Nachbarschaft und 0-2 Figuren darin.

    Frueher erzeugte INIT einen vollstaendigen Graphen (7-9 Knoten) samt
    Charakteren VOR dem ersten Spielzug - eine im Voraus geplante
    Levelgeometrie, die ein Sprachmodell danach nur noch abschreitet.
    Playtesting zeigte: entweder haelt sich das Modell daran (dann war das
    Sprachmodell fuer die Navigation ueberfluessig), oder es weicht ab (dann
    war der Graph verschwendete Arbeit). Deshalb baut INIT KEIN volles Level
    mehr - aber die unmittelbare Umgebung: der Startraum PLUS jeder Raum, der
    direkt mit ihm verbunden ist (Tiefe 1, 2-4 Knoten, 1-3 Verbindungen je
    Knoten). Alles weiter entfernte entsteht als Nebenprodukt von
    resolve_model()s new_room-Feld, sobald die Erzaehlung es braucht.

    FIGUREN sind zurueck, aber anders: nicht als Levelplanung, sondern weil
    die Interaktion mit den Agenten der Kern des Spiels ist - allein in
    einem leeren Raum zu starten ist die Ausnahme, nicht die Norm. Die
    Figuren stehen alle im Startraum (dem ersten in `nodes`), deshalb
    kein at-Feld - der Client setzt "n1" ein (state.World.from_init).

    DIRECTION legt VORAB fest, worauf die Geschichte hinauslaeuft - aber nur
    abstrakt (pull = wohin gezogen wird, pressure = was draengt). Durch die
    Handlungen von Spieler und Figuren wird das im Spielverlauf konkret. Es
    bleibt dem Spieler verborgen (nur RESOLVE/DECIDE sehen es, nie NARRATE
    im Wortlaut - wie shared_target).

    Parameterlos wie narrate_model() - fuer Namenssymmetrie ueber alle vier
    Aufruftypen (init/decide/resolve/narrate).
    """
    class Direction(BaseModel):
        model_config = STRICT

        pull: str = Field(
            description="English, one clause. What this story is being pulled "
                        "toward - see THE DIRECTION in the prompt for how "
                        "concrete or abstract to pitch it.")
        pressure: str = Field(
            description="English, one clause. What makes the pull matter now "
                        "and makes standing still a bad option - see THE "
                        "DIRECTION in the prompt.")

    class StartNode(BaseModel):
        model_config = STRICT

        name: str = Field(
            description="short place name, in the language of the start "
                        "prompt")
        anchor: str = Field(
            description="permanent physical description in English. Only "
                        "things that can be named and touched, and that "
                        "will still be true in twenty scenes. No mood, no "
                        "events, no people.")

    class StartLink(BaseModel):
        model_config = STRICT

        from_name: str = Field(
            description="name of one room in `nodes` this connection starts "
                        "from - copied exactly")
        to_name: str = Field(
            description="name of the room in `nodes` it connects to - copied "
                        "exactly. A plain door, corridor or stair between the "
                        "two; both directions are passable.")

    class StartCharacter(BaseModel):
        model_config = STRICT

        name: str = Field(
            description="a real personal name, in the language of the "
                        "start prompt - not a role or a description")
        agenda_draft: str = Field(
            description="English, one sentence, physically checkable - what "
                        "this character wants. It must put them in contact "
                        "or conflict with the player. Raw material, not a "
                        "finished character sheet - you will not see it again.")
        agenda_target_hint: str = Field(
            description="English, one noun or short noun phrase: what the "
                        "agenda points toward (a place, an object, a "
                        "person). Used by the client to make a decision you "
                        "never see.")
        carries: list[str] = Field(
            description="0-3 short English item names this person is holding "
                        "when the scene opens. Empty for most people. This is "
                        "where the pull object goes if someone starts with it.")

    class Init(BaseModel):
        model_config = STRICT

        language: str = Field(
            description="BCP-47 tag of the language the player wrote the "
                        "start prompt in, e.g. de or en. Every player-facing "
                        "text in this game must use it.")
        nodes: list[StartNode] = Field(
            description="2 to 4 rooms: the room the player starts in FIRST, "
                        "then every room directly connected to it. Depth 1 "
                        "only - do not build a whole level, distant rooms "
                        "appear later through play.")
        connections: list[StartLink] = Field(
            description="undirected links between rooms in `nodes`. Every "
                        "room has 1 to 3 connections, the whole set is "
                        "reachable from the start room, and the start room "
                        "has at least one.")
        direction: Direction = Field(
            description="the stakes this story runs on, fixed abstractly "
                        "before anyone acts. Hidden from the player - the "
                        "narrator never states it; the player infers it "
                        "from what people and the world do.")
        starting_characters: list[StartCharacter] = Field(
            description="0 to 2 characters already in the start room. "
                        "Include at least one whenever the start prompt "
                        "gives you anyone to work with - a companion, an "
                        "opponent, a bystander the situation implies; the "
                        "game is built around dealing with them. Empty only "
                        "for a prompt that is explicitly about being alone.")
        opening_narration: str = Field(
            description="scene 1, in the language of the start prompt, "
                        "second person, 60-120 words. Describe ONLY the "
                        "start room (the first in `nodes`) and the people in "
                        "it - not the connected rooms. Open on a person or "
                        "the one thing the scene turns on; do NOT open on or "
                        "mention the air, a smell, a hum, the light, or 'the "
                        "silence'. No player action has happened yet, so "
                        "nothing here may react to one. End the text with a "
                        "complete sentence.")
        opening_image_prompt: str = Field(
            description="English image description of the opening scene: "
                        "place, light, materials, perspective, and anyone "
                        "present. Nameable physical things only. No style "
                        "words, no negations, no action, no text.")

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

    actor_move_to darf JEDEN bestehenden Knoten nennen, nicht nur die
    Nachbarn: das Modell gibt das ZIEL an ("ich will zum Amt"), und
    World.apply_turn rechnet daraus den Ein-Schritt-Zug auf dem kuerzesten
    Weg (path_step). Frueher war das Literal auf die Nachbar-Exits verengt;
    dann konnte das Modell ein 2-Hop-Ziel gar nicht ausdruecken, xgrammar
    erzwang einen Zufalls-Exit oder "stay", und Figuren blieben kleben oder
    teleportierten. Mehrzuegige Wege laufen jetzt einfach ueber mehrere
    Runden.
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
    ActorMoveTo = Literal[node_ids + ("stay",)]  # type: ignore[valid-type]
    Status = Literal["active", "disabled", "dead"]

    class Move(BaseModel):
        model_config = STRICT
        character: CharId = Field(description="who moves")
        to: NodeId = Field(
            description="the room they are heading for - any room, not only "
                        "a neighbour. Not adjacent: they cover one step "
                        "toward it this turn.")

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

    ItemTo = Literal[                                    # type: ignore[valid-type]
        char_ids + node_ids + ("player", "gone")]

    class ItemMove(BaseModel):
        model_config = STRICT
        item: str = Field(
            description="short English name of the object, e.g. 'the "
                        "encrypted radio', 'the ledger'")
        to: ItemTo = Field(
            description="where it ends up: 'player', a character id, a room "
                        "id (set down there, no longer carried), or 'gone' "
                        "(destroyed or lost). Once carried it stays with that "
                        "holder until another item_move shifts it.")

    class NewRoom(BaseModel):
        model_config = STRICT
        # Pflichtfeld statt NewRoom | None: "kein neuer Raum" ist der leere
        # String bei name, exakt das Muster, das utterance/justification
        # schon nutzen ("kein Text" ist der leere String, kein None). Ein
        # echtes Optional-Feld waere die einzige Ausnahme von "alle Felder
        # Pflicht, kein Optional" in diesem Schema gewesen.
        name: str = Field(
            description="a short place name in the LANGUAGE OF THE START "
                        "PROMPT (same language as the room names you were "
                        "given - never English unless the game itself is in "
                        "English). Empty string if you are not proposing a "
                        "new room this turn.")
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
    # bestehenden Knoten meint oder "stay" (die Ankunft in einem neuen Raum
    # laeuft NIE ueber actor_move_to - der neue Raum hat noch keine Id, kann
    # in ActorMoveTos Literal also gar nicht auftauchen; new_room mit einem
    # Namen IMPLIZIERT selbst die Ankunft dort, siehe state.World.apply_turn).
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
            carries: list[str] = Field(
                description="0-3 short English item names this character "
                            "arrives holding. Empty for most.")

        class ResolvePlayer(BaseModel):
            model_config = STRICT

            new_room: NewRoom = Field(
                description="a new room, or an empty-named one if you "
                            "propose none this turn. If you propose one, "
                            "its name cannot yet be an exit, so set "
                            "actor_move_to to 'stay' - arriving there "
                            "happens automatically.")
            actor_move_to: ActorMoveTo = Field(
                description="the room the player is HEADING FOR - any room on "
                            "the map, not only a neighbouring one. If it is "
                            "not adjacent, they cover one step toward it this "
                            "turn and keep going next turn; you never "
                            "teleport them. 'stay' (or the id of the room "
                            "they are already in) whenever they did not move "
                            "at all - talked, read, handed something over, "
                            "stood still. Most turns are 'stay'. 'stay' as "
                            "well if you are proposing a new room this turn.")
            moves: list[Move] = Field(
                description="OTHER characters who move this turn - not the "
                            "player. Use this when the player's action takes "
                            "someone here WITH them ('we go', 'I drive us "
                            "there', 'come on') - add that person with the "
                            "same target as actor_move_to so they travel "
                            "together. Also for someone the player sends off, "
                            "or who leaves on their own in reaction.")
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
            item_moves: list[ItemMove] = Field(
                description="objects that changed hands, were picked up, set "
                            "down, or destroyed this turn. Usually empty. "
                            "Keeps carried objects from teleporting.")

        return ResolvePlayer

    class ResolveAgentic(BaseModel):
        model_config = STRICT

        new_room: NewRoom = Field(
            description="a new room, or an empty-named one if you propose "
                        "none this turn. If you propose one, its name "
                        "cannot yet be an exit, so set actor_move_to to "
                        "'stay' - arriving there happens automatically.")
        actor_move_to: ActorMoveTo = Field(
            description="the room you are HEADING FOR - any room on the map, "
                        "not only a neighbouring one. If it is not adjacent "
                        "you cover one step toward it this turn and keep "
                        "going next turn; you never teleport. 'stay' (or the "
                        "id of the room you are already in) whenever you did "
                        "not move at all. 'stay' as well if you are proposing "
                        "a new room.")
        moves: list[Move] = Field(
            description="OTHER characters who move this turn - not you (you "
                        "use actor_move_to). Use it for someone you lead or "
                        "send somewhere, or who leaves in reaction - add them "
                        "with their target node.")
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
        item_moves: list[ItemMove] = Field(
            description="objects that changed hands, were picked up, set "
                        "down, or destroyed this turn. Usually empty. Keeps "
                        "carried objects from teleporting.")

    return ResolveAgentic


def normalize_model() -> type[BaseModel]:
    """Die Spieler-Eingabe in die Ich-Perspektive gebracht, bevor sie in den
    Ledger geht.

    Der Spieler steuert nur die eigene Figur. Schreibt er "Du nimmst das
    Radio" oder "Er geht zur Tuer", ist das trotzdem SEINE Handlung - das
    Modell soll sie nicht einer anderen Figur zuschreiben. Text in
    Anfuehrungszeichen ist woertliche Spielerrede und bleibt unangetastet.

    Ein billiger Aufruf ohne Denkprozess (call="normalize", nie in
    THINK_CALLS). Statisch, ohne Ids - reine Textumformung.
    """
    class Normalized(BaseModel):
        model_config = STRICT

        text: str = Field(
            description="the player's action, rewritten so every action is "
                        "in the first person ('I take the radio', 'I go to "
                        "the door'). Keep any \"...\" quoted speech word for "
                        "word. Same language as the input. Add nothing, drop "
                        "nothing, resolve nothing - only change the "
                        "grammatical person.")

    return Normalized


def normalized_text(normalized) -> str:
    """Der umgeformte Eingabetext - siehe narration_text()."""
    return normalized.text


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
