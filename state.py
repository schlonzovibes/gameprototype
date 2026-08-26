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
from dataclasses import dataclass, field
from typing import NamedTuple

# Wie viele Erinnerungen eine Figur behaelt. Begrenzt, weil der Zustand bei
# JEDEM Aufruf komplett mitgeschickt wird - eine unbegrenzte Liste liesse
# den Kontext mit jeder Szene wachsen.
MEMORY_LIMIT = 8

# Wie viele fruehere Narrationen im Zustand bleiben. Sie ersetzen die
# assistant-Turns des alten Chatverlaufs und dienen allein der stilistischen
# Verankerung - inhaltlich steht alles Wichtige im Graphen.
RECENT_LIMIT = 3

# Obergrenzen im CLIENT, nicht im Prompt (siehe World.apply_turn): ein
# Resolve-Aufruf darf hoechstens so viele Events melden, eine Runde
# hoechstens so viele aktive NPCs ziehen lassen. Ein Modell, das mehr
# liefert, wird gekappt statt abgewiesen - der Ueberschuss landet in der
# Ablehnungsliste fuers Debug-Log.
MAX_EVENTS = 4
MAX_ACTIVE_NPCS = 4


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
    memory: list[str] = field(default_factory=list)


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
    # Rundenzustand, KEINE Weltwahrheit - siehe copy(). Faengt bei jeder
    # Runde leer an und sammelt, was diese eine Runde an Events ausgeloest
    # hat, mitsamt der Spielerposition zum jeweiligen Zeitpunkt.
    round_log: list[RoundEntry] = field(default_factory=list)

    # ------------------------------------------------------------ Aufbau

    @classmethod
    def from_init(cls, init) -> World:
        """Aus dem validierten InitWorld-Objekt eine lebende Welt bauen.

        Listen werden zu dicts: im Spiel wird fast immer ueber die Id
        zugegriffen ("wo steht c1?"), nicht durchlaufen.
        """
        return cls(
            language=init.language,
            nodes={
                n.id: Node(
                    id=n.id, name=n.name, anchor=n.anchor,
                    exits=[Exit(to=e.to, one_way=e.one_way,
                                justification=e.justification)
                           for e in n.exits],
                )
                for n in init.nodes
            },
            characters={
                c.id: Character(id=c.id, name=c.name, at=c.at,
                                agenda=c.agenda, aim=c.aim)
                for c in init.characters
            },
            facts=list(init.facts),
            player_at=init.player_at,
        )

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

    def active_npcs_in_order(self) -> list[Character]:
        """Aktive Figuren in stabiler Reihenfolge - die Zugreihenfolge einer
        Runde.

        "Stabil" heisst: Einfuegereihenfolge des dicts, die exakt der
        Reihenfolge aus INIT entspricht (from_init baut characters aus
        init.characters, einer Liste, und Python-dicts erhalten die
        Einfuegereihenfolge). Nicht sortiert, nicht gemischt - wer zuerst
        zieht, hat einen echten Vorteil, und der muss vorhersagbar bleiben.

        MAX_ACTIVE_NPCS ist eine defensive Obergrenze (Brief 8.3); InitWorld
        begrenzt characters bereits auf 2-4, dieser Fall ist also heute
        unerreichbar - die Kappung bleibt trotzdem, falls sich das aendert.
        """
        npcs = [c for c in self.characters.values() if c.status == "active"]
        return npcs[:MAX_ACTIVE_NPCS]

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

        for char in self.characters.values():
            lines.append(self._char_line(char))
            lines.extend(f"  remembers: {m}" for m in char.memory)

        if self.facts:
            lines.append("FACTS: " + " | ".join(self.facts))
        if self.recent:
            lines.append("RECENTLY: " + " | ".join(self.recent))

        return "\n".join(lines)

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
        """
        node = self.nodes[char.at]

        lines = [
            f"YOU ARE {char.id} {char.name}",
            f"YOUR AGENDA: {char.agenda}",
            f"YOUR CURRENT AIM: {char.aim}",
            "",
            f'YOU ARE AT {node.id} "{node.name}"',
            f"  {node.anchor}",
        ]
        lines.extend(f"  {mark}" for mark in node.marks)
        lines.append(f"  exits: {self._exit_text(node)}")

        # Andere Anwesende - nur Name, sonst nichts. Was jemand will, sieht
        # man ihm nicht an.
        others = [c for c in self.characters.values()
                  if c.id != char.id and c.at == char.at and c.status == "active"]
        if others:
            lines.append("ALSO HERE: "
                         + ", ".join(f"{c.id} {c.name}" for c in others))
        if self.player_at == char.at:
            lines.append("THE PLAYER IS HERE")

        if char.memory:
            lines.append("YOU REMEMBER:")
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
        lines = [f'{node.id} "{node.name}"', f"  {node.anchor}"]
        lines.extend(f"  {mark}" for mark in node.marks)
        lines.append(f"  exits: {self._exit_text(node)}")
        present = [c for c in self.characters.values()
                  if c.status == "active" and c.at == self.player_at]
        if present:
            lines.append("PRESENT: " + ", ".join(c.name for c in present))
        return "\n".join(lines)

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

    def _char_line(self, char: Character) -> str:
        """"CHAR c1 Vogel @n3 active   wants: get the pump running | now: force the valve".

        agenda und aim stehen nur bei handlungsfaehigen Figuren - bei einer
        toten waeren sie bestenfalls verwirrend.
        """
        line = f"CHAR {char.id} {char.name} @{char.at} {char.status}"
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

        Ersetzt das fruehere apply(): eine Runde besteht jetzt aus mehreren
        Aufrufen hier (einmal fuer den Spieler, einmal je aktivem NPC), statt
        aus einem einzigen. Hier - und nur hier - waechst die Welt. Die
        Rueckgabeliste ist kein Fehlerkanal, sondern eine Beobachtung fuers
        Debug-Log: sie zeigt, wo das Modell etwas wollte, was der Graph nicht
        hergibt.

        actor_id ist "player" oder eine CharId - der Akteur, dessen Delta
        das hier ist. scene_number wird HIER NICHT mehr erhoeht: eine Runde
        besteht aus mehreren apply_turn()-Aufrufen, der Aufrufer (Game.advance)
        erhoeht ihn genau einmal, am Rundenende.
        """
        rejected: list[str] = []
        actor_node = self._actor_node_id(actor_id)

        # --- Bewegung des Akteurs ---
        if delta.actor_move_to != "stay":
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
            char = self.characters.get(change.character)
            if char is None:
                rejected.append(f"status {change.character}: unknown character")
            else:
                char.status = change.status

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

        return rejected
