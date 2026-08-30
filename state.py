"""Der Weltzustand - gewoehnliche Python-Objekte, kein Pydantic.

=== Warum hier dataclasses und nicht Pydantic ===

Pydantic validiert Fremdeingaben. Hier kommt aber nichts Fremdes an: was in
diesen Objekten landet, hat die Grammatik des Servers und danach die
Validierung in schema.py bereits passiert. Zu validieren gaebe es nichts
mehr - zu VERWALTEN dagegen viel. Dafuer sind dataclasses das leichtere
Werkzeug.

=== Wo die Wahrheit liegt ===

Frueher lebte der Weltzustand im Chatverlauf: jede Antwort des Modells blieb
als assistant-Nachricht stehen, und HISTORY_TURNS schnitt den Verlauf ab
Szene 13 vorne ab. Die Welt vergass damit ihren eigenen Anfang, ohne dass es
irgendwo auffiel.

Jetzt liegt sie hier, in Python. Das Sprachmodell hat kein Gedaechtnis mehr
und braucht keins: es bekommt bei JEDEM Aufruf den vollstaendigen Zustand
frisch gerendert mitgeschickt. Was es zurueckgibt, ist nur ein Delta.

=== Die Arbeitsteilung, die diese Datei traegt ===

Das Schema verhindert Ids, die es nicht gibt - schon beim Erzeugen.
apply_turn() verhindert Bewegungen, die es nicht geben KANN - beim Anwenden.

Beides sind harte Regeln, und beide stehen in deterministischem Code, nicht
in einer Bitte an das Modell. Was apply_turn() ablehnt, wandert ins Debug-Log
statt still zu passieren.
"""

from __future__ import annotations

import copy
import os
import types
from dataclasses import dataclass, field
from typing import NamedTuple

# Wie viele Erinnerungen eine Figur behaelt. Begrenzt, weil der Zustand bei
# JEDEM Aufruf komplett mitgeschickt wird - eine unbegrenzte Liste liesse
# den Kontext mit jeder Szene wachsen.
MEMORY_LIMIT = 8

# Wie viele Figuren HOECHSTENS eigene DECIDE/RESOLVE-Zuege bekommen (Design-Doc
# §7). Die ersten so vielen eingefuehrten Figuren werden agentisch, spaeter
# eingefuehrte nicht mehr - ihr Verhalten faellt dann in NARRATE. story.py
# faechert die DECIDE-Aufrufe dieser Figuren parallel gegen vLLM; bei einer
# Runde mit regelmaessig >4 Agenten sollte --max-num-seqs entsprechend hoch
# stehen (siehe story.LLM_CONCURRENCY).
MAX_AGENTIC = int(os.environ.get("AIGAME_MAX_AGENTIC", "4"))

# Wie viele fruehere Narrationen im Zustand bleiben. Sie ersetzen die
# assistant-Turns des alten Chatverlaufs und dienen allein der stilistischen
# Verankerung - inhaltlich steht alles Wichtige im Graphen.
RECENT_LIMIT = 3

# Obergrenze im CLIENT, nicht im Prompt (siehe World.apply_turn): ein
# Resolve-Aufruf darf hoechstens so viele Events melden. Ein Modell, das
# mehr liefert, wird gekappt statt abgewiesen - der Ueberschuss landet in
# der Ablehnungsliste fuers Debug-Log.
MAX_EVENTS = 4

# Wie viele Gegenstaende der Spieler oder eine Figur HOECHSTENS traegt. Ein
# leichtes Modell gegen das Teleportieren von Objekten (Playtest "Geheimagent
# im McDonald's": das Funkgeraet wanderte Tasche -> Tisch -> Kueche -> NPC ->
# weg, ohne dass irgendein Feld es festhielt). item_moves im Resolve-Delta
# verschiebt Objekte zwischen diesen Listen.
INVENTORY_LIMIT = 3


@dataclass
class Exit:
    to: str
    one_way: bool
    justification: str


@dataclass
class Node:
    id: str
    name: str
    anchor: str
    exits: list[Exit]
    # marks sind append-only: was einmal geschehen ist, bleibt sichtbar.
    # Genau das unterscheidet einen Ort, an dem der Spieler war, von einem,
    # an dem er nicht war.
    marks: list[str] = field(default_factory=list)


@dataclass
class Character:
    id: str
    name: str
    at: str
    # agenda ist unveraenderlich (aus INIT, der einzige Grund, warum diese
    # Figur je handelt) - aim ist es nicht: die Figur ersetzt ihn in jedem
    # DECIDE-Aufruf selbst durch den naechsten Schritt. Getrennt, weil beide
    # verschiedene Fragen beantworten: agenda gibt Richtung ueber den ganzen
    # Lauf, aim gibt Fortschritt.
    agenda: str
    aim: str
    status: str = "active"
    # Hoechstens MAX_AGENTIC Figuren duerfen True sein - World.add_character()
    # erzwingt das. Eine agentische Figur bekommt pro Runde einen eigenen
    # DECIDE/RESOLVE-Zug; die Reihenfolge ergibt sich aus der Einfuege-
    # reihenfolge von World.characters (= Spawn-Reihenfolge).
    is_agentic: bool = False
    # Das pro-Figur verborgene Ziel (aus agenda_target_hint): woraufhin diese
    # Figur zieht, ohne es je auszusprechen. Nur in render_for() DIESER Figur
    # sichtbar, nie in render()/NARRATE. Leer bei nicht-agentischen Figuren
    # und beim Notanker (spawn_fallback_character).
    hidden_target: str = ""
    memory: list[str] = field(default_factory=list)
    # Was diese Figur bei sich traegt (hoechstens INVENTORY_LIMIT). Wird ueber
    # item_moves im Resolve-Delta gefuellt/geleert; in render_for() nur der
    # Figur selbst gezeigt, in render() allen.
    inventory: list[str] = field(default_factory=list)


class RoundEntry(NamedTuple):
    """Ein Eintrag im Rundenprotokoll: wer hat wo was ausgeloest.

    player_was_at wird beim ANWENDEN festgehalten, nicht nachtraeglich
    berechnet - bewegt sich der Spieler mitten in der Runde, entscheidet
    seine Position ZUM ZEITPUNKT dieses einen Events, ob er es mitbekommen
    hat (siehe World.apply_turn und visible()).

    Ein NamedTuple statt eines nackten Tupels: kostet zur Laufzeit nichts,
    verhaelt sich weiterhin wie ein Tupel, gibt Aufrufern aber .node/
    .player_was_at statt magischer Indizes.
    """
    actor: str          # "player" oder eine CharId
    node: str
    clause: str
    player_was_at: str


def visible(round_log: list[RoundEntry]) -> list[RoundEntry]:
    """Nur die Eintraege, die der Spieler an seinem damaligen Ort miterlebt
    hat - in Reihenfolge. Alles andere ist geschehen und bleibt ungenannt.

    Das ist die eigentliche Filterung, die autonome NPCs erst sinnvoll
    macht (siehe Modul-Docstring): der Erzaehler bekommt nur, was diese
    Funktion durchlaesst, kann also nur darueber schreiben.
    """
    return [e for e in round_log if e.node == e.player_was_at]


@dataclass
class World:
    language: str
    nodes: dict[str, Node]
    characters: dict[str, Character]
    facts: list[str]
    player_at: str
    # Der Szenenzaehler lebt AUSSCHLIESSLICH hier. Frueher fragte der Client
    # das Modell danach und zaehlte nur hilfsweise selbst mit - eine Zahl,
    # die er die ganze Zeit selbst wusste.
    scene_number: int = 0
    recent: list[str] = field(default_factory=list)
    # Was der Spieler bei sich traegt (hoechstens INVENTORY_LIMIT) - das
    # Gegenstueck zu Character.inventory. Es gibt kein Spieler-Objekt, deshalb
    # ein eigenes Feld hier.
    player_carries: list[str] = field(default_factory=list)
    # Knoten, in denen der Spieler in einer FRUEHEREN Runde schon war (nach
    # from_init: der Startraum). NARRATE bekommt daraus ein "erstes Mal hier"
    # / "schon mal hier"-Signal (render_player_place), damit bestehende Raeume
    # nicht jede Runde neu mit Luft/Licht/Geruch etabliert werden. Der aktuelle
    # Ort kommt erst NACH erfolgreichem NARRATE dazu (story.Game.advance).
    # Liste, keine Menge: asdict()/json.dumps im JSON-Log kann kein set.
    visited: list[str] = field(default_factory=list)
    # Rundenzustand, KEINE Weltwahrheit - siehe copy(). Faengt bei jeder
    # Runde leer an und sammelt, was diese eine Runde an Events ausgeloest
    # hat, mitsamt der Spielerposition zum jeweiligen Zeitpunkt.
    round_log: list[RoundEntry] = field(default_factory=list)

    # Obergrenze fuer wachsende Raeume (siehe add_node/can_grow) - jenseits
    # davon bewegt sich der Spieler nur noch in der bestehenden Topologie,
    # damit Handlung durch Wiederkehr statt endloses Wachstum entsteht.
    max_nodes: int = 12
    # Die abstrakte Richtung der Geschichte, bei INIT gesetzt: pull = worauf
    # alle zugezogen werden, pressure = was draengt. Zieht Spieler UND
    # Figuren von Anfang an in eine Richtung; wird durch die Handlungen im
    # Verlauf konkret. Geht an RESOLVE und DECIDE (render/render_for), NIE
    # an NARRATE im Wortlaut - secret_leaked() prueft das mit.
    pull: str = ""
    pressure: str = ""
    # Obergrenze fuer agentische Figuren (siehe Character.is_agentic /
    # MAX_AGENTIC). Als Feld statt nur als Konstante, damit ein Testfall sie
    # gezielt hochsetzen kann.
    max_agentic: int = field(default_factory=lambda: MAX_AGENTIC)

    # ------------------------------------------------------------ Aufbau

    @classmethod
    def from_init(cls, init) -> World:
        """Aus dem validierten Init-Objekt eine lebende Welt bauen.

        INIT liefert Sprache, die lokale Nachbarschaft (2-4 Raeume: der
        Startraum plus jeder direkt verbundene, siehe schema.init_model) und
        0-2 Figuren, die schon im Startraum stehen. Weiter entfernte Knoten
        und spaetere Figuren entstehen erst waehrend des Spiels
        (World.add_node / _introduce_characters).

        Die Startfiguren laufen durch dieselbe Maschinerie wie eine spaeter
        eingefuehrte Figur: _introduce_characters() vergibt die Ids, befoerdert
        die ersten MAX_AGENTIC Figuren zu agentischen und setzt je ihr
        hidden_target. INIT kennt keine Knoten-Id, deshalb werden n1..nk hier
        in `nodes`-Reihenfolge vergeben (n1 = der Startraum).
        """
        raw_nodes = list(getattr(init, "nodes", None) or [])
        if not raw_nodes:
            # Rueckfall auf die aeltere Init-Form (start_node_name/-anchor),
            # damit alte Stubs/Fixtures nicht brechen.
            raw_nodes = [types.SimpleNamespace(
                name=getattr(init, "start_node_name", "Room"),
                anchor=getattr(init, "start_node_anchor", ""))]

        nodes: dict[str, Node] = {}
        name_to_id: dict[str, str] = {}
        for i, rn in enumerate(raw_nodes[:4], start=1):
            nid = f"n{i}"
            nodes[nid] = Node(id=nid, name=rn.name, anchor=rn.anchor, exits=[])
            name_to_id.setdefault(rn.name, nid)

        wired: set[frozenset] = set()
        for link in getattr(init, "connections", None) or []:
            a = name_to_id.get(getattr(link, "from_name", ""))
            b = name_to_id.get(getattr(link, "to_name", ""))
            if a is None or b is None or a == b:
                continue                       # unbekannter Name / Selbstlink
            pair = frozenset((a, b))
            if pair in wired:
                continue                       # Duplikat
            if len(nodes[a].exits) >= 3 or len(nodes[b].exits) >= 3:
                continue                       # 1-3 Verbindungen je Knoten
            wired.add(pair)
            nodes[a].exits.append(Exit(to=b, one_way=False, justification=""))
            nodes[b].exits.append(Exit(to=a, one_way=False, justification=""))

        direction = getattr(init, "direction", None)
        world = cls(
            language=init.language,
            nodes=nodes,
            characters={},
            facts=[],
            player_at="n1",
            visited=["n1"],
            pull=getattr(direction, "pull", "") or "",
            pressure=getattr(direction, "pressure", "") or "",
        )
        starters = [
            types.SimpleNamespace(name=c.name, at="n1",
                                  agenda_draft=c.agenda_draft,
                                  agenda_target_hint=c.agenda_target_hint,
                                  carries=list(getattr(c, "carries", []) or []))
            for c in getattr(init, "starting_characters", [])
        ]
        world._introduce_characters(starters, [])
        return world

    # ------------------------------------------------------------ Abfragen

    def node_ids(self) -> tuple[str, ...]:
        """Alle Knoten-Ids - so wie sie in die Grammatik des naechsten Zuges
        gehen. Als Tupel, weil Literal[] etwas Unveraenderliches braucht."""
        return tuple(self.nodes)

    def active_ids(self) -> tuple[str, ...]:
        """Nur handlungsfaehige Figuren.

        Tote und Ausgeschaltete kommen gar nicht erst in die Grammatik -
        damit kann der Resolver sie nicht versehentlich bewegen lassen.
        """
        return tuple(c.id for c in self.characters.values()
                     if c.status == "active")

    def resolvable_ids(self) -> tuple[str, ...]:
        """Figuren, die ein Resolve-Delta REFERENZIEREN darf: aktive UND
        ausgeschaltete, aber keine toten.

        Eine disabled Figur kann nicht laufen (apply_turn lehnt moves weiter
        ab), aber ein Kampf muss sie zu Ende bringen koennen (disabled ->
        dead) oder sie sich erholen lassen (disabled -> active). Baute die
        Grammatik nur aus active_ids(), waere eine ausgeschaltete Figur fuer
        immer eingefroren - im Playtest "Schneesturm" lag Thomas so 7 Szenen
        als stumme Requisite, weil sein Tod-Delta nie in die Grammatik passte.
        """
        return tuple(c.id for c in self.characters.values()
                     if c.status != "dead")

    def player_stranded(self) -> bool:
        """Steht der Spieler in einem Raum, aus dem KEIN Weg zurueck zum
        Startknoten n1 fuehrt?

        Dann hat er den Kernschauplatz ueber eine Einbahn verlassen - "raus,
        wo alles spielte, und kein Zurueck" ist ein legitimes Spielende
        (story.advance wertet das zusammen mit MIN_SCENES aus). BFS ueber die
        gerichteten Exits, kein Fund von "n1" -> gestrandet.
        """
        if self.player_at == "n1" or "n1" not in self.nodes:
            return False
        seen, stack = {self.player_at}, [self.player_at]
        while stack:
            node = stack.pop()
            for nxt in self.exits_from(node):
                if nxt == "n1":
                    return False
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return True

    def agentic_count(self) -> int:
        """Wie viele agentische Slots belegt sind.

        Tote geben ihren Slot frei (apply_turn setzt is_agentic beim Tod auf
        False, siehe dort) - eine spaeter eingefuehrte Figur kann dann
        nachruecken. Disabled zaehlt weiter: eine ausgeschaltete Figur kann
        sich erholen, dann waere der Slot sonst ueberbelegt.
        """
        return sum(1 for c in self.characters.values() if c.is_agentic)

    def agentic_actors(self) -> list[Character]:
        """Die AKTIVEN agentischen Figuren in Zug-Reihenfolge (= Spawn-
        Reihenfolge, weil dict die Einfuegereihenfolge haelt).

        story.Game.advance() geht diese Liste pro Runde durch: je Figur ein
        DECIDE (parallel gefaechert) und danach ein RESOLVE (seriell, damit
        Figur N+1 das angewandte Delta von Figur N schon sieht). Disabled/tote
        Figuren fallen raus - ein toter NPC ist zudem schon nicht mehr
        is_agentic.
        """
        return [c for c in self.characters.values()
                if c.is_agentic and c.status == "active"]

    def perceivers(self, node_id: str) -> list[Character]:
        """Wer an diesem Ort etwas wahrnehmen koennte - das gesamte
        Wahrnehmungsmodell.

        Gleiche Position heisst Wahrnehmung, sonst nicht. Bewusst grob:
        Schall durch Kanten, Sichtlinien, Dunkelheit sind spaetere
        Erweiterungen, kein Teil dieses Umbaus.
        """
        return [c for c in self.characters.values()
                if c.status == "active" and c.at == node_id]

    def exits_from(self, node_id: str) -> tuple[str, ...]:
        """Wohin man von hier aus kommt.

        Einbahnkanten brauchen keine Sonderbehandlung: sie stehen schlicht
        nur in der Exit-Liste des Ausgangsknotens, nicht in der des Ziels.
        Die Richtung steckt also schon im Graphen.
        """
        node = self.nodes.get(node_id)
        return tuple(e.to for e in node.exits) if node else ()

    # ------------------------------------------------------------ Wachstum

    def can_grow(self) -> bool:
        """Darf ueberhaupt noch ein neuer Raum entstehen?

        Getrennt von add_node() selbst: can_grow() ist die Entscheidung
        (der Aufrufer fragt VOR dem Versuch), add_node() nur die Ausfuehrung
        und kennt max_nodes gar nicht.
        """
        return len(self.nodes) < self.max_nodes

    def add_node(self, name: str, anchor: str, from_node: str,
                one_way: bool, justification: str) -> str:
        """Einen neuen Raum anhaengen, verbunden mit from_node.

        Vergibt die naechste freie Id fortlaufend (n1, n2, ...) - Raeume
        verschwinden nie (siehe Modul-Docstring), deshalb ist
        len(self.nodes) + 1 garantiert eine frische, nie zuvor vergebene Id.

        Verknuepft automatisch beidseitig, ausser one_way=True - dann fuehrt
        nur der Weg von from_node zum neuen Raum, keine Rueckkante.
        """
        node_id = f"n{len(self.nodes) + 1}"
        self.nodes[node_id] = Node(id=node_id, name=name, anchor=anchor,
                                   exits=[])
        self.nodes[from_node].exits.append(
            Exit(to=node_id, one_way=one_way, justification=justification))
        if not one_way:
            self.nodes[node_id].exits.append(
                Exit(to=from_node, one_way=False, justification=""))
        return node_id

    def add_character(self, name: str, at: str, agenda: str,
                      is_agentic: bool) -> str:
        """Eine neue Figur anlegen, vergibt die naechste freie Id (c1, c2, ...).

        Wirft, wenn is_agentic=True verlangt wird, obwohl schon max_agentic
        agentische Figuren existieren - der Aufrufer (_introduce_characters /
        spawn_fallback_character) prueft das selbst und uebergibt dann False;
        kommt hier trotzdem True an, ist es sein Programmierfehler, kein
        Spielereignis zum stillschweigenden Korrigieren.
        """
        if is_agentic and self.agentic_count() >= self.max_agentic:
            raise ValueError(
                f"already at max_agentic ({self.max_agentic}) agentic "
                f"characters")
        char_id = f"c{len(self.characters) + 1}"
        # aim startet leer - die Figur setzt ihren ersten Schritt selbst im
        # naechsten DECIDE-Aufruf (render_for() zeigt dafuer einen
        # Platzhaltertext statt einer leeren Zeile, siehe dort).
        self.characters[char_id] = Character(
            id=char_id, name=name, at=at, agenda=agenda, aim="",
            is_agentic=is_agentic)
        return char_id

    def character_quota_status(self, turn_number: int) -> str:
        """Der Regiehinweis fuer RESOLVE(actor=player): fehlt noch eine
        Figur, und ist die Frist (Zug 5) schon ueberschritten?

        Meist steht hier nichts: INIT setzt schon 0-2 Figuren in den
        Startraum (from_init), die Quote ist dann von Anfang an erfuellt.
        Der Hinweis greift nur, wenn WIRKLICH niemand da ist - weiche
        Fuehrung bis Zug 5 ("STORY DIRECTION"), harte Pflicht danach
        ("MANDATORY"). Reine Arithmetik, kein Modellzugriff.
        """
        if self.characters:      # schon mindestens eine Figur -> erfuellt
            return ""

        if turn_number > 5:
            # Das Modell setzt is_agentic NICHT selbst (das Feld existiert
            # im Schema gar nicht - die Auswahl trifft der Client in
            # _introduce_characters). Der Pflichttext verlangt nur eine
            # Figur mit brauchbarem agenda_target_hint.
            return "MANDATORY: this call MUST introduce a character."

        turns_left = max(0, 5 - turn_number)
        return (f"STORY DIRECTION: a character must appear by turn 5. "
                f"{turns_left} turn(s) remain. Introduce one where the "
                f"scene gives a natural occasion.")

    def spawn_fallback_character(self, names_pool: list[str]) -> str:
        """Client-seitiger Notanker, wenn selbst die Notbremse (3 gescheiterte
        RESOLVE-Versuche) keine neue Figur hervorbringt.

        Kein Modellaufruf - ein schmuckloser NPC ist besser als ein
        gebrochenes Versprechen an den Spieler. Ohne echten Ziel-Hinweis
        bleibt hidden_target leer statt geraten - der agentische Zug faellt
        fuer diese Figur inhaltlich schwach aus, aber das Spiel bricht nicht
        ab.
        """
        used = {c.name for c in self.characters.values()}
        name = next((n for n in names_pool if n not in used), "Stranger")
        make_agentic = self.agentic_count() < self.max_agentic
        # hidden_target bleibt leer: ohne echten agenda_target_hint gibt es
        # nichts sinnvoll abzuleiten - der agentische Zug faellt fuer diese
        # Figur inhaltlich schwach aus, aber das Spiel bricht nicht ab.
        return self.add_character(name=name, at=self.player_at,
                                  agenda="", is_agentic=make_agentic)

    # ------------------------------------------------------------ Rendern

    def render(self) -> str:
        """Der volle Zustand als strukturierter KLARTEXT, nicht als JSON.

        Zwei Gruende gegen JSON als EINGABE:

        Erstens kostet es 30-40 % mehr Tokens fuer Anfuehrungszeichen,
        Klammern und Kommata, die keine Bedeutung tragen.

        Zweitens - und das wiegt schwerer - verleitet ein JSON-Eingabeblock
        das Modell dazu, die Struktur zu SPIEGELN statt sie zu lesen: es
        beginnt, Felder der Eingabe in der Ausgabe zu wiederholen. Klartext
        laesst diese Verwechslung gar nicht erst zu; die Form der Ausgabe
        bestimmt allein die Grammatik.

        Die Ids bleiben maschinell (n3, c1), weil genau sie im Output als
        Literal zurueckkommen muessen.
        """
        lines = []

        for node in self.nodes.values():
            here = "  [player here]" if node.id == self.player_at else ""
            lines.append(f'NODE {node.id} "{node.name}"{here}')
            lines.append(f"  {node.anchor}")
            lines.extend(f"  {mark}" for mark in node.marks)
            lines.append(f"  exits: {self._exit_text(node)}")

        # Die EINE verlaessliche Quelle fuer "wer ist gerade wo". Die
        # remembers:-Zeilen unten sind Vergangenheit ("fuhr zum Friedhof",
        # "stieg aus") und sagen NICHTS ueber die aktuelle Position - das
        # Modell hat das im Playtest wiederholt verwechselt.
        lines.append("POSITIONS NOW (read positions ONLY here, never from "
                     "events or memories):")
        lines.append(f'  player: {self._place_name(self.player_at)}')
        for char in self.characters.values():
            if char.status != "dead":
                lines.append(f"  {char.name}: {self._place_name(char.at)}")

        for char in self.characters.values():
            lines.append(self._char_line(char))
            if char.inventory:
                lines.append("  carries: " + ", ".join(char.inventory))
            lines.extend(f"  did/heard earlier: {m}" for m in char.memory)

        if self.player_carries:
            lines.append("PLAYER CARRIES: " + " | ".join(self.player_carries))
        if self.facts:
            lines.append("FACTS: " + " | ".join(self.facts))
        lines.extend(self._direction_lines())
        if self.recent:
            lines.append("RECENTLY: " + " | ".join(self.recent))

        return "\n".join(lines)

    def _direction_lines(self) -> list[str]:
        """Die abstrakte Story-Richtung als Regieblock - fuer RESOLVE und
        DECIDE, damit Spieler UND Figuren in dieselbe Richtung streben. NIE
        von render_player_place() aufgerufen: NARRATE bekommt sie nicht."""
        out = []
        if self.pull:
            out.append(f"PULL (never say this - move everyone toward it): {self.pull}")
        if self.pressure:
            out.append(f"PRESSURE (never say this - let it bear down): {self.pressure}")
        return out

    def render_for(self, char: Character) -> str:
        """Der gefilterte Kontext fuer den DECIDE-Aufruf.

        DAS IST DER KERN DER AGENTENIDEE: Das Nichtwissen dieser Figur ist
        eine Eigenschaft des Kontextfensters, keine Bitte im Prompt. Was
        hier nicht steht, kann das Modell nicht verwenden - es muss sich
        nicht daran erinnern, etwas nicht zu wissen.

        Ausdruecklich NICHT enthalten:
          - andere Knoten (die Figur kennt nur, wo sie steht)
          - agenda oder aim anderer Figuren (sonst spielt sie gegen ihr Wissen)
          - die globalen facts (die sieht man nirgends "an einem Ort")
          - die Spielerposition, wenn er woanders ist
          - das Rundenprotokoll (das ist Erzaehler-Material, nicht ihres)

        EINE Ausnahme, nur fuer agentische Figuren mit gesetztem
        hidden_target: eine zusaetzliche Zeile mit dem verborgenen Ziel, auf
        das diese Figur zieht. Sie geht NUR hierher - nie in render() (das
        sehen RESOLVE UND der Spieler-Kontext) und nie in narrate_model()s
        Kontext. Ohne sie koennte die Figur ihr Ziel nicht konsistent
        verfolgen; mit ihr NUR hier bleibt es fuer jeden anderen Aufruf
        unsichtbar. Jede agentische Figur hat ihr eigenes.
        """
        node = self.nodes[char.at]

        lines = [
            f"YOU ARE {char.id} {char.name}",
            f"YOUR AGENDA: {char.agenda}",
            f"YOUR CURRENT AIM: {char.aim or '(not yet set - decide one now)'}",
            # Explizit, weil der Rest des Blocks englisches Arbeitsmaterial ist
            # und ein DECIDE ohne Denkprozess sonst die utterance auf Englisch
            # setzt statt in der Spielsprache.
            f"LANGUAGE: say your lines (utterance) in {self.language}",
        ]
        if char.is_agentic and char.hidden_target:
            lines.append(f"YOUR HIDDEN AIM RELATES TO: {char.hidden_target}")
        # Die Story-Richtung geht an JEDE Figur (nicht nur die agentische):
        # sie soll spueren, wohin es zieht und was draengt, auch wenn ihre
        # eigene agenda etwas anderes will. Nie aussprechen.
        lines.extend(self._direction_lines())
        lines += [
            "",
            f'YOU ARE AT {node.id} "{node.name}"',
            f"  {node.anchor}",
        ]
        lines.extend(f"  {mark}" for mark in node.marks)
        lines.append(f"  exits: {self._exit_text(node)}")

        if char.inventory:
            lines.append("YOU ARE CARRYING: " + ", ".join(char.inventory))

        # Andere Anwesende - nur Name, sonst nichts. Was jemand will, sieht
        # man ihm nicht an.
        others = [c for c in self.characters.values()
                  if c.id != char.id and c.at == char.at and c.status == "active"]
        if others:
            lines.append("ALSO HERE: "
                         + ", ".join(f"{c.id} {c.name}" for c in others))
        if self.player_at == char.at:
            lines.append("THE PLAYER IS HERE")
            if self.player_carries:
                lines.append("THE PLAYER IS HOLDING: "
                             + ", ".join(self.player_carries))
        else:
            lines.append("THE PLAYER IS NOT HERE")

        if char.memory:
            # Bewusst "happened earlier", nicht "remember": die Zeilen sind
            # Vergangenheit ("player drove off", "got out of the car") und
            # sagen NICHTS darueber, wo jetzt jemand steht - dafuer zaehlt nur
            # YOU ARE AT / THE PLAYER IS (NOT) HERE oben.
            lines.append("WHAT HAPPENED EARLIER (past - not where anyone is now):")
            lines.extend(f"  {m}" for m in char.memory)

        return "\n".join(lines)

    def render_player_place(self) -> str:
        """Der Ort des Spielers als Text - der Kontext fuer NARRATE.

        Bewusst eng, spiegelbildlich zu render_for(): NARRATE bekommt
        weder agenda/aim/intent noch memory noch facts noch andere Knoten -
        nur, wo der Spieler gerade steht, und (getrennt, siehe
        story.Game._narrate) die Events, die er dort wahrnehmen konnte.
        """
        node = self.nodes[self.player_at]
        # KEINE Knoten-Id hier: NARRATE erzeugt kein Delta und referenziert
        # nie einen Knoten - eine Id wie "n1" waere fuer den Erzaehler nur
        # Text, den er versehentlich abschreiben kann (und getan hat). Auch
        # die Ausgaenge nur als Anzahl + Einbahn-Begruendungen, nicht als
        # Id-Liste.
        seen = "YOU HAVE BEEN HERE BEFORE" if self.player_at in self.visited \
            else "FIRST TIME IN THIS PLACE"
        lines = [seen, node.name, f"  {node.anchor}"]
        lines.extend(f"  {mark}" for mark in node.marks)
        if self.player_carries:
            lines.append("  you are carrying: " + ", ".join(self.player_carries))
        if node.exits:
            oneways = [e.justification for e in node.exits
                       if e.one_way and e.justification]
            extra = f" ({'; '.join(oneways)})" if oneways else ""
            lines.append(f"  ways out: {len(node.exits)}{extra}")
        else:
            lines.append("  ways out: none yet")
        present = [c for c in self.characters.values()
                  if c.status == "active" and c.at == self.player_at]
        if present:
            lines.append("PRESENT: " + ", ".join(c.name for c in present))
        return "\n".join(lines)

    def secret_leaked(self, narrator_text: str) -> bool:
        """Ist ein verborgenes Figuren-Ziel ODER die Story-Richtung
        (pull/pressure) woertlich in narrator_text durchgesickert?

        Heisst bewusst secret_leaked und nicht *_target_*: der Methodenname
        darf den Feldnamen (hidden_target) nicht als Teilstring enthalten,
        sonst schluege der Grep-Test (tests/test_no_leaked_field_names.py)
        auf story.py an, das diese Methode aufruft.

        Diese Pruefung lebt bewusst HIER und nicht in story.py, obwohl sie
        story.Game._narrate aufruft: der Feldname des verborgenen Ziels
        darf ausserhalb von schema.py/state.py nirgends im Wortlaut
        auftauchen (Debug-Log-Grep, siehe tests/test_no_leaked_field_names.py)
        - waere die Pruefung in story.py, muesste sie dort auf den Namen
        zugreifen und ihn damit selbst schreiben.

        Bewusst nur ein einfacher Teilstring-Vergleich, keine Nominalphrasen-
        Extraktion - Letzteres waere ein eigenes Textverstehens-Problem. Bei
        pull/pressure (ganze Klauseln) ist ein woertlicher Treffer selten,
        aber wenn er kommt, ist er ein echtes Leck.
        """
        haystack = narrator_text.lower()
        secrets = [c.hidden_target for c in self.characters.values()]
        secrets += [self.pull, self.pressure]
        return any(s and s.lower() in haystack for s in secrets)

    def remember(self, narration: str) -> None:
        """Eine neue Erzaehlung als stilistischen Anker vormerken.

        Frueher geschah das inline in apply(), sobald die Szene feststand.
        Jetzt entsteht die Erzaehlung erst NACH allen apply_turn()-Aufrufen
        einer Runde (im separaten NARRATE-Aufruf) - deshalb ruft der
        Aufrufer (Game.advance) das hier einmal je Runde separat auf,
        sobald NARRATE erfolgreich war.
        """
        self.recent.append(narration)
        del self.recent[:-RECENT_LIMIT]

    def copy(self) -> World:
        """Tiefe Kopie fuer die transaktionale Arbeitskopie einer Runde.

        Game.advance() arbeitet auf genau so einer Kopie und committet sie
        erst nach NARRATE (siehe dort) - schlaegt irgendein Aufruf davor
        fehl, bleibt self.world exakt wie vorher, ohne eigenes Aufraeumen.

        round_log wird bewusst NICHT mitkopiert: es ist Rundenzustand, keine
        Weltwahrheit - eine neue Runde faengt immer mit einem leeren
        Protokoll an, egal was in der Quelle noch stand.
        """
        new = copy.deepcopy(self)
        new.round_log = []
        return new

    def _exit_text(self, node: Node) -> str:
        """"n2 | n7 (one-way, down the shaft)"."""
        parts = []
        for exit_ in node.exits:
            if exit_.one_way and exit_.justification:
                parts.append(f"{exit_.to} (one-way, {exit_.justification})")
            elif exit_.one_way:
                parts.append(f"{exit_.to} (one-way)")
            else:
                parts.append(exit_.to)
        return " | ".join(parts) if parts else "none"

    def _place_name(self, node_id: str) -> str:
        """'Friedhof (n3)' - Name zuerst, Id in Klammern. Das Modell soll in
        Klartext den Namen nutzen; die Id braucht es nur fuer die paar
        Felder, die ein Literal verlangen."""
        node = self.nodes.get(node_id)
        return f'{node.name} ({node_id})' if node else node_id

    def _char_line(self, char: Character) -> str:
        """"CHAR c1 Vogel @Cellar(n3) active   wants: ... | now: ...".

        agenda und aim stehen nur bei handlungsfaehigen Figuren - bei einer
        toten waeren sie bestenfalls verwirrend.
        """
        agentic = " agentic" if char.is_agentic else ""
        line = (f"CHAR {char.id} {char.name} "
                f"@{self._place_name(char.at)} {char.status}{agentic}")
        if char.status != "active":
            return line
        return f"{line}   wants: {char.agenda} | now: {char.aim}"

    def _actor_node_id(self, actor_id: str) -> str:
        """An welchem Knoten steht der Akteur GERADE - Spieler oder Figur."""
        return (self.player_at if actor_id == "player"
                else self.characters[actor_id].at)

    # ------------------------------------------------------------ Anwenden

    def apply_turn(self, actor_id: str, delta) -> list[str]:
        """Das Delta EINES Akteurzuges anwenden. Rueckgabe: was abgelehnt
        wurde.

        Eine Runde ruft das hier je aktivem Akteur einmal (Spieler, dann jede
        aktive agentische Figur in Zug-Reihenfolge). Hier - und nur hier -
        waechst die Welt. Die Rueckgabeliste ist kein Fehlerkanal, sondern
        eine Beobachtung fuers Debug-Log: sie zeigt, wo das Modell etwas
        wollte, was der Graph nicht hergibt.

        actor_id ist "player" oder eine CharId - der Akteur, dessen Delta
        das hier ist. scene_number wird HIER NICHT mehr erhoeht: der
        Aufrufer (Game.advance) erhoeht ihn genau einmal, am Rundenende.
        """
        rejected: list[str] = []
        actor_node = self._actor_node_id(actor_id)

        # --- Neuer Raum (ZUERST, siehe schema.resolve_model) ---
        # Ein neuer Raum kann in actor_move_to gar nicht als Ziel auftauchen
        # - seine Id existiert erst NACH diesem Aufruf, die Grammatik von
        # actor_move_to wurde aber VOR dem Aufruf aus den bestehenden
        # Ausgaengen gebaut. delta.new_room mit einem Namen IMPLIZIERT
        # deshalb selbst die Ankunft dort; actor_move_to wird in diesem Fall
        # ignoriert (resolve_player.txt/resolve_agentic.txt weisen das
        # Modell an, es dann auf "stay" zu setzen).
        if delta.new_room.name:
            if self.can_grow():
                new_id = self.add_node(
                    delta.new_room.name, delta.new_room.anchor,
                    from_node=actor_node, one_way=delta.new_room.one_way,
                    justification=delta.new_room.justification)
                if actor_id == "player":
                    self.player_at = new_id
                else:
                    self.characters[actor_id].at = new_id
            else:
                rejected.append(
                    f"new_room: max_nodes ({self.max_nodes}) reached, discarded")
        elif delta.actor_move_to != "stay":
            if delta.actor_move_to in self.exits_from(actor_node):
                if actor_id == "player":
                    self.player_at = delta.actor_move_to
                else:
                    self.characters[actor_id].at = delta.actor_move_to
            else:
                rejected.append(
                    f"{actor_id} {actor_node} -> {delta.actor_move_to}: no such exit")

        # --- Bewegungen ANDERER Figuren ---
        # actor_id selbst ist hier bewusst ausgeschlossen: die eigene
        # Bewegung des Akteurs laeuft ausschliesslich ueber actor_move_to.
        # Ein zweiter Kanal fuer dieselbe Figur waere ein Widerspruch im
        # selben Delta (zwei Ziele moeglich) - deshalb Ablehnung, nicht
        # stille Uebernahme.
        for move in delta.moves:
            if move.character == actor_id:
                rejected.append(
                    f"moves: cannot re-move the acting character "
                    f"{actor_id}, use actor_move_to")
                continue
            char = self.characters.get(move.character)
            if char is None:
                rejected.append(f"move {move.character}: unknown character")
            elif char.status != "active":
                rejected.append(f"move {char.id} -> {move.to}: {char.status}")
            elif move.to not in self.exits_from(char.at):
                rejected.append(f"move {char.id} {char.at} -> {move.to}: no such exit")
            else:
                char.at = move.to

        # --- Zustandswechsel ---
        # NACH den Bewegungen: so darf eine Figur im selben Zug noch
        # fliehen und danach zusammenbrechen. Umgekehrt waere die Flucht
        # abgelehnt worden, weil sie da schon nicht mehr aktiv gewesen waere.
        for change in delta.status_changes:
            # Manche Modelle schreiben "c1 Thomas" statt "c1" ins Feld (die
            # Grammatik SOLLTE nur die Id zulassen, aber ein Reparaturversuch
            # kann daneben liegen). Den fuehrenden Id-Token nehmen, damit ein
            # sonst gueltiger Zustandswechsel nicht an der Schreibweise
            # scheitert (Playtest "Schneesturm": Thomas' Tod ging so verloren).
            cid = change.character.split()[0] if change.character else ""
            char = self.characters.get(cid)
            if char is None:
                rejected.append(f"status {change.character}: unknown character")
            else:
                char.status = change.status
                # Ein toter NPC verlaesst den agentischen Pool: agentic_count()
                # sinkt, ein spaeter eingefuehrter NPC kann den Slot erben.
                # Nur bei "dead", nicht bei "disabled" (kann sich erholen).
                if change.status == "dead":
                    char.is_agentic = False

        # --- Spuren am Ort ---
        for mark in delta.marks_added:
            node = self.nodes.get(mark.node)
            if node is None:
                rejected.append(f"mark at {mark.node}: unknown node")
            elif mark.clause in node.marks:
                rejected.append(f"mark at {mark.node}: duplicate")
            else:
                node.marks.append(mark.clause)

        # --- Weltfakten ---
        for fact in delta.facts_added:
            if fact in self.facts:
                rejected.append(f"fact: duplicate")
            else:
                self.facts.append(fact)

        # --- Neue Figuren (nur mode="player" - das Feld existiert bei
        # mode="agentic" im Schema gar nicht, siehe schema.resolve_model) ---
        if hasattr(delta, "characters_introduced"):
            self._introduce_characters(delta.characters_introduced, rejected)

        # --- Ereignisse: Erinnerungen verteilen, Rundenprotokoll fuehren ---
        # Auf MAX_EVENTS gekappt - eine Obergrenze im Client, nicht im
        # Schema/Prompt (Brief 8.3). Ueberschuss wird verworfen und geloggt,
        # nicht teilweise verarbeitet.
        for event in delta.events[:MAX_EVENTS]:
            node = self.nodes.get(event.node)
            if node is None:
                rejected.append(f"event at {event.node}: unknown node")
                continue

            # Character ist ein gewoehnliches (nicht-frozen) dataclass und
            # damit unhashbar (Python setzt __hash__ automatisch auf None,
            # sobald eq=True und frozen=False) - deshalb Liste mit "not in"
            # statt set(). "not in" nutzt das generierte __eq__ und reicht
            # hier: die Liste ist pro Event nie groesser als eine Handvoll
            # Figuren.
            receivers = self.perceivers(event.node)
            # Die handelnde Figur bekommt ihr eigenes Event IMMER, auch wenn
            # sie nicht (mehr) an diesem Knoten steht - explizit hinzugefuegt
            # statt perceivers() selbst akteursbewusst zu machen. So bleibt
            # perceivers() ein reiner Positionsfilter, wiederverwendbar auch
            # anderswo.
            actor_char = self.characters.get(actor_id)   # None fuer "player"
            if actor_char is not None and actor_char not in receivers:
                receivers = receivers + [actor_char]

            for c in receivers:
                c.memory.append(event.clause)
                # Negativer Index zaehlt von hinten: die letzten N behalten.
                del c.memory[:-MEMORY_LIMIT]

            # player_was_at NACH der Bewegung dieses Akteurs gelesen (siehe
            # oben) - bewegt sich der Spieler in diesem Zug, gilt fuer seine
            # EIGENEN Events schon die neue Position, exakt "zum Zeitpunkt
            # des Events".
            self.round_log.append(
                RoundEntry(actor_id, event.node, event.clause, self.player_at))

        if len(delta.events) > MAX_EVENTS:
            rejected.append(
                f"events: {len(delta.events) - MAX_EVENTS} dropped, over MAX_EVENTS")

        # --- Gegenstaende (Inventar) ---
        # Ein Objekt liegt entweder in genau einer Inventarliste (Spieler oder
        # eine Figur) oder ist ungetrackt (auf einem Moebel, im Raum). Jeder
        # item_move nimmt es ueberall raus und legt es ans Ziel - "player",
        # eine Figuren-Id, eine Knoten-Id (abgelegt, nicht mehr getragen) oder
        # "gone" (zerstoert/verloren).
        for im in getattr(delta, "item_moves", []):
            item = im.item.strip()
            if not item:
                continue
            low = item.lower()
            self.player_carries[:] = [
                x for x in self.player_carries if low not in x.lower()]
            for c in self.characters.values():
                c.inventory[:] = [
                    x for x in c.inventory if low not in x.lower()]
            if im.to in ("gone",) or im.to in self.nodes:
                continue                       # nicht mehr getragen
            if im.to == "player":
                dest = self.player_carries
            elif im.to in self.characters:
                dest = self.characters[im.to].inventory
            else:
                rejected.append(f"item_move {item!r} -> {im.to}: no such holder")
                continue
            if len(dest) >= INVENTORY_LIMIT:
                rejected.append(
                    f"item_move {item!r} -> {im.to}: inventory full "
                    f"({INVENTORY_LIMIT})")
                continue
            dest.append(item)

        return rejected

    def _introduce_characters(self, new_chars, rejected: list[str]) -> None:
        """0-2 vorgeschlagene Figuren tatsaechlich anlegen.

        Die ersten MAX_AGENTIC Figuren im Spiel werden agentisch - egal ob
        aus INIT oder aus einem spaeteren Zug, in Einfuegereihenfolge (kein
        Modell-Urteil, kein Vorsortieren). Danach eingefuehrte Figuren sind
        nicht agentisch; ihr Verhalten faellt in NARRATE. Stirbt eine
        agentische Figur, gibt sie ihren Slot frei (apply_turn), und die
        naechste eingefuehrte kann nachruecken.

        Jede beförderte Figur bekommt ihr eigenes hidden_target aus ihrem
        agenda_target_hint - das pro-Figur verborgene Ziel, nur in ihrem
        eigenen DECIDE-Kontext sichtbar.
        """
        for nc in list(new_chars[:2]):   # 0-2 laut Schema, defensiv gekappt
            if nc.at not in self.nodes:
                rejected.append(f"character {nc.name}: unknown node {nc.at}")
                continue
            make_agentic = self.agentic_count() < self.max_agentic
            char_id = self.add_character(nc.name, nc.at, nc.agenda_draft,
                                         make_agentic)
            self.characters[char_id].inventory = list(
                getattr(nc, "carries", []) or [])[:INVENTORY_LIMIT]
            if make_agentic:
                # agenda_target_hint ist bereits die knappe Nominalphrase
                # (schema.py, NewCharacter) - direkt uebernommen.
                self.characters[char_id].hidden_target = nc.agenda_target_hint
