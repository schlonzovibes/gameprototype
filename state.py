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
apply() verhindert Bewegungen, die es nicht geben KANN - beim Anwenden.

Beides sind harte Regeln, und beide stehen in deterministischem Code, nicht
in einer Bitte an das Modell. Was apply() ablehnt, wandert ins Debug-Log
statt still zu passieren.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import schema

# Wie viele Erinnerungen eine Figur behaelt. Begrenzt, weil der Zustand bei
# JEDEM Aufruf komplett mitgeschickt wird - eine unbegrenzte Liste liesse
# den Kontext mit jeder Szene wachsen.
MEMORY_LIMIT = 8

# Wie viele fruehere Narrationen im Zustand bleiben. Sie ersetzen die
# assistant-Turns des alten Chatverlaufs und dienen allein der stilistischen
# Verankerung - inhaltlich steht alles Wichtige im Graphen.
RECENT_LIMIT = 3


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
    goal: str
    status: str = "active"
    memory: list[str] = field(default_factory=list)


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
                c.id: Character(id=c.id, name=c.name, at=c.at, goal=c.goal)
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

    def companions(self) -> list[Character]:
        """Aktive Figuren am Spielerknoten - die Kandidaten fuer den
        INTENT-Aufruf. Wer nicht anwesend ist, hat in diesem Zug nichts
        wahrzunehmen."""
        return [c for c in self.characters.values()
                if c.status == "active" and c.at == self.player_at]

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
        """Der gefilterte Kontext fuer den INTENT-Aufruf.

        DAS IST DER KERN DER AGENTENIDEE: Das Nichtwissen dieser Figur ist
        eine Eigenschaft des Kontextfensters, keine Bitte im Prompt. Was
        hier nicht steht, kann das Modell nicht verwenden - es muss sich
        nicht daran erinnern, etwas nicht zu wissen.

        Ausdruecklich NICHT enthalten:
          - andere Knoten (die Figur kennt nur, wo sie steht)
          - die Ziele anderer Figuren (sonst spielt sie gegen ihr Wissen)
          - die globalen facts (die sieht man nirgends "an einem Ort")
          - die Spielerposition, wenn er woanders ist
        """
        node = self.nodes[char.at]

        lines = [
            f"YOU ARE {char.id} {char.name}",
            f"YOUR GOAL: {char.goal}",
            "",
            f'YOU ARE AT {node.id} "{node.name}"',
            f"  {node.anchor}",
        ]
        lines.extend(f"  {mark}" for mark in node.marks)
        lines.append(f"  exits: {self._exit_text(node)}")

        # Andere Anwesende - Name und Zustand, aber KEIN Ziel. Was jemand
        # will, sieht man ihm nicht an.
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
        """"CHAR c1 Vogel @n3 active   wants: get the pump running".

        Das Ziel steht nur bei handlungsfaehigen Figuren - bei einer toten
        waere es bestenfalls verwirrend.
        """
        line = f"CHAR {char.id} {char.name} @{char.at} {char.status}"
        return f"{line}   wants: {char.goal}" if char.status == "active" else line

    # ------------------------------------------------------------ Anwenden

    def apply(self, turn) -> list[str]:
        """Das Delta anwenden. Rueckgabe: was abgelehnt wurde.

        Hier - und nur hier - waechst die Welt. Die Rueckgabeliste ist kein
        Fehlerkanal, sondern eine Beobachtung fuers Debug-Log: sie zeigt,
        wo das Modell etwas wollte, was der Graph nicht hergibt. Haeufen
        sich dieselben Ablehnungen, stimmt etwas mit dem Prompt nicht - das
        sieht man nur, wenn man es aufschreibt.
        """
        rejected: list[str] = []

        # --- Spielerbewegung ---
        if turn.player_move_to != "stay":
            if turn.player_move_to in self.exits_from(self.player_at):
                self.player_at = turn.player_move_to
            else:
                rejected.append(
                    f"player {self.player_at} -> {turn.player_move_to}: no such exit")

        # --- Figurenbewegungen ---
        for move in turn.moves:
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
        for change in turn.status_changes:
            char = self.characters.get(change.character)
            if char is None:
                rejected.append(f"status {change.character}: unknown character")
            else:
                char.status = change.status

        # --- Spuren am Ort ---
        for mark in turn.marks_added:
            node = self.nodes.get(mark.node)
            if node is None:
                rejected.append(f"mark at {mark.node}: unknown node")
            elif mark.clause in node.marks:
                rejected.append(f"mark at {mark.node}: duplicate")
            else:
                node.marks.append(mark.clause)

        # --- Weltfakten ---
        for fact in turn.facts_added:
            if fact in self.facts:
                rejected.append(f"fact: duplicate")
            else:
                self.facts.append(fact)

        # --- Erinnerungen ---
        for memory in turn.memories_added:
            char = self.characters.get(memory.character)
            if char is None:
                rejected.append(f"memory {memory.character}: unknown character")
            else:
                char.memory.append(memory.clause)
                # Negativer Index zaehlt von hinten: die letzten N behalten.
                del char.memory[:-MEMORY_LIMIT]

        # --- Zaehler und Erzaehlgedaechtnis ---
        self.scene_number += 1
        # Der Zugriff laeuft ueber schema.scene_text() statt direkt ins
        # Turn-Objekt: die Feldnamen des Ausgabeformats stehen ausschliesslich
        # in schema.py, sonst waere der Contract wieder an zwei Orten.
        # [0] ist der Erzaehltext, [1] waere der Bildprompt.
        self.recent.append(schema.scene_text(turn)[0])
        del self.recent[:-RECENT_LIMIT]

        return rejected
