# Zustandsformat: Charaktere, Welt, Delta

Entwurf. Noch nichts davon ist implementiert - dieses Dokument ist die
Vorlage, gegen die implementiert wird (wie ein "Brief"). Offene Punkte am
Ende sind mit **[FRAGE]** markiert.

---

## 1. Ziel & Prinzipien

1. **Die ganze Wahrheit liegt in Python.** `state.py` haelt den vollen
   Zustand als Dataclasses. Das Sprachmodell hat kein Gedaechtnis und sieht
   bei jedem Aufruf nur eine gerenderte Sicht.
2. **Das Modell gibt nur ein Delta aus** - nie den vollen Zustand. Das Delta
   ist JSON, von einer Grammatik (schema.py) erzwungen und von Pydantic
   validiert. Python wendet es an (`apply_turn`).
3. **Constrained fields sind Enums**, keine freien Strings: Haltung, Status,
   Gegenstandszustand, Reichweite eines Ereignisses.
4. **Das Level ist persistent.** Raeume verschwinden nie. Beim Start
   existiert nur der Raum des Spielers; jeder weitere entsteht als
   Nebenprodukt einer Aktion und wird ins Level-Ledger eingetragen.
5. **Nichtwissen ist eine Eigenschaft des Kontextfensters.** Was ein NPC
   in DECIDE sieht, haengt von der Beziehung ab: sich selbst voll; die
   Karte des Levels voll; **Bekannte/Rivalen** (`bonds` mit `known=True`)
   inkl. deren `goal`/`aim`; **Fremde und den Spieler** nur ueber
   wahrgenommene Handlungen im geteilten Raum (siehe 5).
6. **Der Spieler ist eine Entitaet wie jeder NPC** - gleiche `Actor`-
   Struktur inkl. `traits`, nur `id == "player"` und keine
   `goal`/`stance`/`flaw`/`aim`.
7. **`name` ist ein Eigenname, `look` ist das Aussehen.** "Thomas", nicht
   "Sailor". Eine Rolle/Beschreibung im Namensfeld ist ein Fehler.

---

## 2. Der kanonische Zustand (Python)

### Enums

```python
class Posture(str, Enum):
    standing = "standing"
    kneeling = "kneeling"
    sitting  = "sitting"
    lying    = "lying"
    moving   = "moving"      # laeuft/rennt - momentan, faellt meist im
                             # naechsten Zug auf standing zurueck
    jumping  = "jumping"     # ebenso momentan

class Status(str, Enum):
    active   = "active"      # handelt normal
    disabled = "disabled"    # anwesend, aber handlungsunfaehig
    dead     = "dead"        # endgueltig weg

class Stance(str, Enum):
    ally      = "ally"       # stuetzend (aus Eigennutz, nicht aus Nettigkeit)
    adversary = "adversary"  # gegnerisch
    neutral   = "neutral"    # nur ZWISCHEN NPCs (bonds) - zum Spieler gibt
                             # es kein neutral, da haengt eine Figur immer
                             # mit an (6a)

class ItemState(str, Enum):
    intact = "intact"
    open   = "open"          # Behaelter/Tuer offen
    broken = "broken"
    lit    = "lit"           # brennt/leuchtet
    # bewusst klein gehalten; erweiterbar, aber jede neue Konstante muss
    # in describe() erklaerbar sein

class EventReach(str, Enum):
    here     = "here"        # nur im selben Raum wahrnehmbar
    adjacent = "adjacent"    # auch in direkt verbundenen Raeumen hoerbar

class Rating(str, Enum):
    low     = "low"          # deutlich unter der Norm - scheitert an dem,
                             # was ein Durchschnittsmensch gerade so schafft
    average = "average"
    high    = "high"         # schafft, woran die Norm scheitert
```

### Item

```python
@dataclass
class Item:
    id: str                 # "i1", "i2", ... fortlaufend, nie wiederverwendet
    name: str               # in Spielsprache
    look: str               # kurze optische Beschreibung, English, physisch
    state: ItemState = ItemState.intact
    # Ort ist IMPLIZIT: ein Item steht entweder in Room.items ODER in
    # Actor.inventory - nie in beidem, nie nirgends.
```

### Actor  (Spieler und NPCs)

```python
@dataclass
class Traits:               # jeder Actor, auch der Spieler
    # Fuenf breite Kategorien, angelehnt an D&D / Fallout SPECIAL, aber
    # zusammengefasst - je low / average / high. Bewusst allgemein: sie
    # sollen JEDE Aktion einordnen koennen, nicht ein Regelwerk sein.
    might:    Rating = Rating.average  # STR + CON / Fallout STR+END. Kraft
                            # UND Zaehigkeit: heben, brechen, eintreten,
                            # festhalten, einen Treffer wegstecken, nicht
                            # zusammenbrechen.
    finesse:  Rating = Rating.average  # DEX + AGI. Tempo, Balance, Praezision:
                            # springen, klettern, schleichen, ausweichen,
                            # zielen, Taschenspiel.
    wits:     Rating = Rating.average  # INT. Verstand, Wissen, Plaene, eine
                            # Luege durchschauen - und wie differenziert die
                            # Figur spricht (low = einfachstes Vokabular, in
                            # Rede UND Verstaendnis).
    notice:   Rating = Rating.average  # PER + WIS. Wahrnehmung: den Hinterhalt
                            # sehen, den Schritt hoeren, einen Raum lesen,
                            # spueren dass gleich gelogen wird.
    presence: Rating = Rating.average  # CHA. Ueberzeugen, einschuechtern,
                            # beruhigen, Aufmerksamkeit halten, glaubhaft
                            # luegen. Spielt gegen `notice`/`wits` des
                            # Gegenuebers.

@dataclass
class Bond:                 # wie EIN NPC zu einem ANDEREN NPC steht
    stance: Stance          # ally | adversary | neutral (zueinander, NICHT
                            # zum Spieler)
    known: bool = False     # kennen sie sich von VOR dem Spiel? known=True
                            # -> die Figur sieht das goal/aim des anderen in
                            # DECIDE (Kollegen koordinieren; alte Rivalen
                            # wissen, was der andere will). known=False ->
                            # nur ueber Wahrnehmung im geteilten Raum.

@dataclass
class Actor:
    id: str                 # "player" | "c1" | "c2" | ...
    name: str               # ein ECHTER Eigenname ("Thomas", "Mara"), NIE
                            # eine Rolle/Beschreibung ("Sailor", "the guard").
                            # Das Aussehen gehoert in look, nicht in name.
    at: str                 # RoomId - die Position lebt HIER, im Actor-
                            # Ledger, nicht im Raum (siehe 5, "Zwei Ledger")
    posture: Posture = Posture.standing
    status: Status = Status.active
    look: str = ""          # kurze BILDLICHE Beschreibung ("heavyset, oil-
                            # stained overalls"), English, physisch - hierher
                            # gehoert alles Aussehen. Aendert sich selten.
    traits: Traits = field(default_factory=Traits)
    inventory: list[Item] = field(default_factory=list)   # MAX 3
    memory: list[str] = field(default_factory=list)       # Wahrnehmungen,
                            # gekappt auf MEMORY_LIMIT (bestehend: 8)

    # --- nur NPCs (beim Spieler leer/irrelevant) ---
    goal: str = ""          # unveraenderlich, aus der Einfuehrung. Das eine
                            # Ziel, das diese Figur antreibt - und das sie
                            # ZWANGSLAEUFIG mit dem Spieler in Kontakt oder
                            # Konflikt bringt: etwas, das nur der Spieler
                            # geben, blockieren oder bedrohen kann. Kein
                            # Ziel "irgendwo da draussen". (siehe 6a)
    stance: Stance = Stance.adversary   # NUR zum Spieler: ally | adversary
                            # (kein neutral - siehe 6a).
    bonds: dict[str, Bond] = field(default_factory=dict)   # ActorId ->
                            # Bond, zu ANDEREN NPCs. Kein Eintrag = Fremde:
                            # neutral, nie zuvor getroffen. Beidseitig -
                            # Client traegt jeden Bond bei beiden ein.
    flaw: str = ""          # ein kurzer, konkreter Schwachpunkt, der ihr
                            # Verhalten faerbt und sabotiert - "she lies
                            # when cornered", "he will not enter an unlit
                            # room", "he cannot resist a dare". Wirkt in
                            # DECIDE (Wahl) und RESOLVE (Ausgang). (siehe 6a)
    aim: str = ""           # aktueller Schritt Richtung goal; die Figur
                            # ersetzt ihn in jedem DECIDE-Aufruf selbst
    agentic: bool = False   # hat eigene DECIDE/RESOLVE-Zuege. HOECHSTENS 4
                            # Figuren gleichzeitig true (siehe 7).
```

**Inventar-Obergrenze (3):** in `apply_turn` durchgesetzt. Ein `take`, das
das vierte Item braechte, wird abgelehnt und ins Debug-Log geschrieben -
wie `MAX_EVENTS` heute.

### Room

```python
@dataclass
class Room:
    id: str                 # "n1", "n2", ... fortlaufend, nie wiederverwendet
    name: str               # in Spielsprache
    anchor: str             # PERMANENTE physische Beschreibung, English -
                            # was in 20 Szenen noch stimmt. (bestehend)
    look: str               # kurze aktuelle optische Beschreibung - darf
                            # sich mit Ereignissen verschieben (Licht,
                            # Unordnung). Getrennt von anchor.
    exits: list[Exit]       # (bestehend: to, one_way, justification)
    items: list[Item] = field(default_factory=list)
    marks: list[str] = field(default_factory=list)
                            # LASTENDE Manipulationen: "the door is kicked in",
                            # "the chest stands open". Append-only. KEINE
                            # transienten Actor-Zustaende mehr (die stehen
                            # jetzt in Actor.posture / Actor.at).
```

**Marks vs. Item-State vs. Posture** - die Regel gegen veraltete Marks:

| Was | Wo es hin gehoert |
|---|---|
| "player kauert hinter der Bank" | `Actor.posture = kneeling` + `Actor.at` - **nie** ein Mark |
| "Kiste ist offen" (Kiste ist ein getracktes Item) | `Item.state = open` |
| "Tuer eingetreten" (Tuer ist blosse Kulisse) | `Room.marks += ["the door is kicked in"]` |
| "Blut auf dem Boden" | `Room.marks` |

Marks bleiben append-only und beschreiben nur, **was tatsaechlich bleibt**.
Transientes hat dort nichts verloren - damit loest sich das
"world state has no concept of my actions"-Problem: die aktuelle Haltung
und Position jedes Akteurs ist ein Feld, das jeden Zug ueberschrieben wird.

### Traits - Aktionsaufloesung

Fuenf Werte, je `low` / `average` / `high`. Sie sind **kein Wuerfelsystem** -
sie geben dem RESOLVER eine harte Vorgabe, wo eine Aktion ueber der Norm
liegt:

- `might low` + "hechtsprung aus dem fenster" / "die verklemmte Tuer
  eintreten" -> scheitert; `might high` -> schafft, woran `average`
  scheitert, und steckt einen Schlag weg, der einen anderen umwirft.
- `finesse low` + "am Sims entlangbalancieren" -> Absturz. `finesse high`
  -> schleicht unbemerkt an `notice average` vorbei.
- `wits low` -> in DECIDE nur einfachste Plaene, `utterance` nur einfachstes
  Vokabular, durchschaut keine Luege. `wits high` -> durchschaut den
  `flaw: "she lies when cornered"` einer Figur, sobald er sie luegen sieht.
- `notice low` -> geht dem Hinterhalt ins offene Messer, hoert den
  herannahenden Schritt nicht. `notice high` -> sieht `presence`-Bluff und
  versteckte Gegenstaende.
- `presence low` + "talk the guard into standing down" -> geht nicht, egal
  wie gut die Zeile klingt. `presence high` -> haelt einen Raum in Schach,
  luegt glaubhaft (solange `notice`/`wits` des Gegenuebers es nicht
  aufdeckt).

Der RESOLVE-Prompt bekommt die `traits` des HANDELNDEN Akteurs und die
Anweisung: eine Aktion klar ueber seinen Werten **gelingt nicht** - und
misslingt mit FOLGEN (siehe "Scheitern & Tod" unten), nicht mit einem
hoeflichen "das klappt nicht". Der DECIDE-Prompt bekommt die `traits` der
entscheidenden Figur (Plan-Komplexitaet, Sprache, sozialer Zugriff).

Woher die Werte kommen: NPCs bei der Einfuehrung (Modell). Der Spieler bei
INIT aus dem Start-Prompt / der Figur ("Prinzessin mit Minigun" ->
might/presence hoch; "mueder alter Wachmann" -> finesse niedrig).
Default ueberall `average`.

### Scheitern & Tod

Aktionen KOENNEN misslingen - fuer den Spieler wie fuer NPCs - und im
schlimmsten Fall toedlich enden. Das ist kein Randfall, sondern die
Grundspannung des Spiels.

- RESOLVE bestimmt fuer den handelnden Akteur ein **`actor_status`**
  (`active` | `disabled` | `dead`, Default `active`) - der Ausgang seiner
  eigenen Aktion fuer ihn selbst. `apply_turn` schreibt es auf den Akteur.
- **Absurde oder unmoegliche Handlungen werden bestraft, nicht abgewiesen.**
  "Ich springe vom Glockenturm und fange den Vogel" -> Sturz, `disabled`
  oder `dead`. "Ich schlucke das Gift, um ihn zu taeuschen" -> es wirkt.
  Der Resolver hat die Autoritaet, hart zu sein.
- Stirbt der Spieler (`player.status == "dead"`), **endet das Spiel sofort**
  - egal in welcher Szene (siehe 8). Er muss Szene 15 nicht erreichen.
- `disabled` beim Spieler: das Spiel laeuft weiter, aber seine Aktionen
  sind eingeschraenkt und werden als Muehen erzaehlt; NPCs handeln um ihn
  herum. **[FRAGE]** oder soll `disabled` bei genug Eskalation ebenfalls
  zum Ende fuehren?

### Wie Traits dem Spieler erscheinen - nie als Zahl

Der Spieler bekommt seine `traits` **nie direkt genannt** - kein Wert, kein
Etikett ("du bist schwach", "deine Staerke: niedrig"). Er soll sie aus
seinem Start-Prompt und dem Verlauf der Erzaehlung ERSCHLIESSEN.

NARRATE bekommt `player.traits` nur als Faerbe-Hinweis, mit harter Regel:

- **Nie benennen.** Kein "low/high", kein "geschickt/ungeschickt" als
  Urteil.
- **Nur als koerperliche Reibung zeigen, und nur wenn sie gerade greift.**
  Gut: "du willst zur Tuer - dein schwerer Koerper kommt nur langsam in
  Gang". Gut: "der Riegel gibt sofort nach, als haettest du ihn nur
  angetippt". Schlecht: "dank deiner hohen `might` gelingt es dir".
- **Nicht vorne draufladen.** Die Eroeffnung deutet die Figur an (Statur,
  Bewegung, wie sie spricht), der Rest ergibt sich Szene fuer Szene, wenn
  eine Aktion an einen Wert stoesst.

Gleiche Disziplin wie bei `direction` / `flaw`: der Wert lebt in Python,
die Erzaehlung spiegelt ihn nur schraeg.

### World

```python
@dataclass
class World:
    language: str
    scene: int
    player: Actor
    rooms: dict[str, Room]
    actors: dict[str, Actor]          # nur NPCs (Spieler in .player)
    facts: list[str]
    recent: list[str]                 # letzte 3 Narrationen, Stilanker
    direction: StoryDirection         # siehe 6
    round_log: list[RoundEntry]       # transient, pro Runde (bestehend)

    max_rooms: int = 15
    max_agentic: int = 4
    last_scene: int = 15              # das Spiel endet SPAETESTENS hier -
                                     # frueher nur durch Spielertod (siehe 8)
```

`shared_target` (heute: verborgenes gemeinsames Ziel von Spieler +
agentischem NPC) geht in `StoryDirection.pull` auf.

### Zwei Ledger - und wo die Position lebt

- **Level-Ledger** = `World.rooms`. Persistent: Geometrie, `anchor`,
  aktueller `look`, `items`, `marks`. Weiss NICHT, wer gerade drin steht.
- **Actor-Ledger** = `World.actors` + `World.player`. Die vollen
  Figuren-Datensaetze - inklusive **`at` (Position)**.

Die Position lebt also im **Actor-Ledger**, an genau einer Stelle. "Wer ist
in n1" ist eine abgeleitete Abfrage (`[a for a in actors if a.at == "n1"]`),
kein gespeichertes Feld. Grund: Position wechselt jede Runde, `Room` ist
das Persistente - zwei Speicherorte fuer dieselbe Wahrheit waeren die
naechste Quelle fuer Desync. `render()` rechnet die Raum-Belegung beim
Rendern aus.

---

## 3. Das Delta (LLM-Ausgabe)

Drei Aufruftypen, drei Delta-Schemata. Die Grammatik waechst **nie** mit
der Weltgroesse: sie beschreibt die Form eines Deltas, nie den Umfang der
Welt. Alle Ids kommen als `Literal` aus den tatsaechlich existierenden
Entitaeten des aktuellen Zuges.

### 3a. RESOLVE  (Spieler ODER ein agentischer NPC)

Feldreihenfolge = Generierungsreihenfolge. Bewegung/Haltung des Akteurs
zuerst, dann Wirkung auf die Welt, dann Ereignisse.

```jsonc
{
  // --- der Akteur selbst ---
  "new_room": {                 // leerer name = kein neuer Raum
    "name": "", "anchor": "", "look": "",
    "one_way": false, "justification": ""
  },
  "move_to": "stay",            // Exit des aktuellen Raums | "stay"
  "posture": "unchanged",       // Posture | "unchanged"
  "actor_status": "active",     // active | disabled | dead - der Ausgang
                                // DIESER Aktion fuer den Akteur SELBST.
                                // "dead"/"disabled" bei einem Fehlschlag mit
                                // Folgen (Sturz, Falle, absurde Handlung -
                                // siehe "Scheitern & Tod" in 2).
  "actor_action": "prying the chest open with a crowbar",
                                // KURZE bildliche Klausel, Praesens, was der
                                // handelnde Akteur diese Runde SICHTBAR tut.
                                // Fuettert die Bild-Pipeline (4a). "" wenn er
                                // sichtbar nichts tut (nur zuschaut/wartet).

  // --- Wirkung auf andere Akteure (Seiteneffekt dieser Aktion) ---
  "moves":           [ { "actor": "c2", "to": "n3" } ],
  "posture_changes": [ { "actor": "c2", "posture": "lying" } ],
  "status_changes":  [ { "actor": "c2", "status": "disabled" } ],

  // --- Gegenstaende ---
  "item_ops": [
    { "op": "take",      "item": "i4" },                       // Raum -> Akteur-Inventar
    { "op": "drop",      "item": "i1", "to_room": "n1" },      // Inventar -> Raum
    { "op": "transfer",  "item": "i2", "to_actor": "c2" },     // Akteur -> anderer Akteur
    { "op": "manipulate","item": "i7", "new_state": "open" },  // Zustandswechsel
    { "op": "create",    "name": "brick", "look": "loose red brick", "in_room": "n1" },
    { "op": "destroy",   "item": "i3" }
  ],

  // --- der Raum ---
  "marks_added": [ { "room": "n1", "clause": "the chest stands forced open" } ],
  "facts_added": [ "the power is out across the whole building" ],

  // --- nur mode=player: 0-2 neue Figuren ---
  "characters_introduced": [
    {
      "name": "Thomas",              // ECHTER Eigenname, keine Rolle
      "at": "n1",
      "look": "heavyset, salt-stiff oilskin jacket",   // hierher das Aussehen
      "posture": "standing",
      "traits": { "might": "high", "finesse": "low", "wits": "average",
                  "notice": "average", "presence": "average" },
      "goal": "he needs the player to hand over the engine-room key so he can seal the lower deck",
      "stance": "adversary",         // zum SPIELER: ally | adversary
      "flaw": "he will not step into an unlit space",
      "bonds": [                     // zu EXISTIERENDEN NPCs
        { "actor": "c1", "stance": "adversary", "known": true }
      ]
      // inventory & agentic entscheidet der Client, nicht das Modell
      // goal MUSS durch den Spieler laufen (geben/blockieren/bedrohen) -
      // ein Ziel, das der Spieler nicht beruehrt, wird vom Client abgelehnt
      // bzw. in resolve_player.txt ausdruecklich verboten
      // bonds werden vom Client beidseitig eingetragen
    }
  ],

  // --- was diese Runde geschah (Perzeptions-Feed) ---
  "events": [
    { "room": "n1", "reach": "here",     "clause": "the chest lid bangs open" },
    { "room": "n1", "reach": "adjacent", "clause": "a shout carries down the corridor" }
  ]
}
```

`apply_turn`-Regeln (deterministisch, Ablehnungen ins Log):

- `new_room` mit Namen -> Raum wird angelegt, an `actor_node` gehaengt,
  ins Ledger eingetragen, Akteur zieht dort ein; `move_to` wird ignoriert.
- `move_to` nur entlang eines echten Exits, sonst verworfen.
- `moves` / `posture_changes` / `status_changes` fuer den **handelnden**
  Akteur sind ein Widerspruch -> abgelehnt (der nutzt `move_to` / `posture`
  / `actor_status`).
- `actor_status` wird auf den handelnden Akteur geschrieben. `player` ->
  `dead`: Spiel endet nach dieser Runde (noch mit NARRATE, dann Ende).
- `item_ops`: `take` scheitert bei vollem Inventar (max 3) oder wenn das
  Item nicht im Raum liegt; `manipulate` nur auf existierende Items.
- `events` auf `MAX_EVENTS` (4) gekappt.

### 3b. DECIDE  (ein agentischer NPC, aus seiner Sicht allein)

```jsonc
{
  "aim":       "get the player alone in a lit room and demand the key",  // ersetzt den alten
  "intent":    "step between the player and the corridor door",          // physisch, pruefbar
  "utterance": "That key. Now.",                                         // was er laut sagt
  "posture":   "standing",                                               // Posture | "unchanged"
  "move_to":   "stay"                                                    // Exit | "stay"
}
```

Kein Zustandsdelta, kein Erzaehltext. Der NPC will nur etwas; ob es klappt,
entscheidet der anschliessende RESOLVE (mode=agentic), der den ganzen
Zustand kennt.

`render_for` gibt dieser Figur `goal`, `stance`, `flaw`, `aim`, ihre
`traits` und ihr Wissen ueber andere (siehe 5) mit. Der decide-Prompt
weist an:

- **jeder Schritt zielt auf `goal` und damit auf den Spieler** (Kontakt
  oder Konflikt, je nach `stance`);
- **der `flaw` faerbt oder sabotiert die Wahl** - "will not step into an
  unlit space" -> nicht den dunklen Weg, auch wenn er kuerzer waere;
- **`wits` steuert `aim`/`intent`/`utterance`**: `low` = einfachster Plan,
  einfachstes Vokabular ("That key. Now."); `high` = mehrstufiger Plan,
  koordiniert mit Bekannten ("Mara is heading for the relay room - I cut
  the player off first").

### 3c. NARRATE

```jsonc
{
  "narrator_text": "...",   // 60-120 Woerter, 2. Person, Spielsprache
  "image_prompt": "..."     // detaillierter Diffusion-Prompt, English
}
```

**Kein `can_end` mehr.** Das Ende entscheidet allein der Client: Szene 15
erreicht ODER Spieler tot (siehe 8). Der Erzaehler urteilt nicht darueber.
In der LETZTEN Szene bekommt er einen Client-Hinweis "this is the final
scene - close it" und schreibt einen Abschluss.

Zwei getrennte Sichten fuer die zwei Ausgaben:

- **fuer `narrator_text`:** wie heute eng - der Ort des Spielers + die fuer
  ihn SICHTBAREN Events (siehe 5). Er darf nur ueber Wahrgenommenes
  schreiben.
- **fuer `image_prompt`:** der volle **scene brief** (4a) - `Room.visual`
  und JEDE anwesende Figur mit `look`, `posture` und ihrer `actor_action`
  dieser Runde. Das ist alles, was der Spieler an seinem Ort ohnehin
  SIEHT, also kein Leck - nur gebuendelt, damit der Prompt so detailliert
  wie moeglich wird.

Bekommt **nie** `direction.pull` / `direction.pressure`, keine `goal`/
`aim`/`flaw`, keine anderen Raeume - **und keine Ids**: `render_player_place()`
nennt den Raum nur beim Namen und die Ausgaenge nur als Anzahl (+
Einbahn-Begruendungen), nie `n1`/`n2`. Ein Erzaehler hatte im Test "n1"
in die Prosa geschrieben; die Id war schlicht nicht mehr in seinem
Kontext, seit sie hier raus ist. `narrate.txt` verbietet Ids zusaetzlich.

Python stellt den scene brief zusammen (nichts kann fehlen), der Erzaehler
formt ihn zu einem detaillierten Prompt (liest gut, setzt Fokus/Kamera).
Beides zusammen - siehe 4a.

### 3d. INIT  (einmalig, Weltaufbau)

Baut den Startraum, die Spielerfigur UND **0-2 NPCs, die schon darin
stehen** - keine weiteren Raeume (die entstehen im Spiel). Mit NPCs zu
starten ist der Normalfall, nicht die Ausnahme: das Spiel dreht sich um den
Umgang mit den Agenten. Allein startet nur ein Prompt, der ausdruecklich
von Einsamkeit handelt.

```jsonc
{
  "language": "de",
  "start_room_name": "Kirche",
  "start_room_anchor": "stone vaulted ceiling, rows of wooden pews, a stone altar",
  "start_room_look": "dim, dust hanging in the window light",

  "player_look": "grey overcoat torn at the sleeve, a crowbar in one hand",
  "player_traits": { "might": "average", "finesse": "high",
                     "wits": "average", "notice": "average", "presence": "low" },

  "starting_characters": [        // 0-2, alle im Startraum
    { "name": "Thomas",
      "agenda_draft": "he wants the player to hand over the engine-room key",
      "agenda_target_hint": "the engine-room key" }
    // volle NPC-Werte (traits/stance/flaw/bonds) setzt der Client bzw. der
    // erste RESOLVE - INIT liefert nur name + Ziel. Kein at-Feld: es gibt
    // nur den einen Raum, der Client setzt "n1" ein.
  ],

  "opening_narration": "...",   // 60-120 Woerter, 2. Person, Spielsprache
  "opening_image_prompt": "..." // detaillierter Diffusion-Prompt aus
                                // start_room_anchor + start_room_look +
                                // player_look + jede Figur in starting_characters
}
```

- `start_room_anchor` + `start_room_look` landen zusammen in `Room` und
  werden ueberall als eine Einheit verwendet - `Room.visual` (siehe 4a).
- **`starting_characters`** (0-2): jede muss ein `agenda_draft` haben, das
  DURCH DEN SPIELER laeuft (6a), und einen echten Eigennamen. Sie laufen
  durch dieselbe Client-Maschinerie wie eine spaeter eingefuehrte Figur
  (`_introduce_characters`): Ids, agentic-Wahl (die ERSTE Figur wird
  agentic, siehe 7), `direction.pull`/`shared_target`.
- `opening_narration` MUSS die Anwesenden einfuehren - mit Namen, und zeigen
  was sie TUN, nicht was sie wollen.
- `player_look` / `player_traits` werden aus dem Start-Prompt / der Figur
  abgeleitet, die der Spieler spielt.
- **`opening_narration`** macht weiter, was es bisher macht: es EROEFFNET
  die Geschichte und gibt dem Spieler Kontext aus seinem Start-Prompt.
  Zusaetzlich MUSS es enthalten: eine Beschreibung des ersten Raums UND
  einen Eindruck der Spielerfigur - Statur, Bewegung, Auftreten, wie sie
  spricht. Aber **nicht ihre Werte im Klartext**: die Traits werden
  angedeutet (der "schwere Koerper", die "ruhige Hand"), nie benannt
  (siehe "Wie Traits dem Spieler erscheinen" in 2). Der Spieler soll aus
  Prompt + Eroeffnung auf seine Faehigkeiten schliessen, nicht sie
  vorgelesen bekommen. Noch keine Aktion ist passiert, also reagiert
  nichts darauf.
  **[FRAGE]** passt das alles in 60-120 Woerter, oder braucht die
  Eroeffnung mehr Laenge als eine spaetere `narrator_text`?
- `opening_image_prompt`: Szene 1 hat keine Aktionen, das Modell baut ihn
  aus `Room.visual` + `player.look` (siehe 4a).
- **[FRAGE]** `direction.pull`/`pressure` hier schon als Feld mitgeben
  (Modell setzt sie), oder Client-seitig? (siehe 6)

---

## 4. Rendern (Zustand -> Modell)

Der kanonische Speicher ist JSON/Dataclasses. Was das Modell **sieht**,
ist davon getrennt und austauschbar. Empfehlung fuers Erste:
**strukturierter Klartext beibehalten** (dokumentierte Gruende: ~30-40 %
weniger Tokens; JSON-Eingabe verleitet zum Spiegeln der Struktur). Neu ist
nur, was im Klartext auftaucht:

```
NODE n1 "Kirche"  [player here]
  anchor: stone vaulted ceiling, rows of wooden pews, a stone altar
  look:   dim, dust hanging in the window light
  marks:  the chest stands forced open
  items:  i4 brass candlestick (intact) | i7 wooden chest (open)
  exits:  n2

PLAYER  @n1  kneeling  active
  look:      grey overcoat, torn at the sleeve
  traits:    body average | agility high | wits average | presence low
  carries:   i1 crowbar (intact)
  remembers: the chest lid bangs open

CHAR c1 "Thomas"  @n1  standing  active   agentic  adversary
  goal: get the engine-room key from the player | now: block the corridor door
  flaw: will not step into an unlit space
  look:      heavyset, salt-stiff oilskin jacket
  traits:    body high | agility low | wits average | presence average
  carries:   (nothing)
  remembers: a shout carries down the corridor

FACTS: the power is out across the whole building
DIRECTION (phase: escalate): everyone is being pushed toward the upper deck;
          the water is rising one level per scene    <-- nur RESOLVE/DECIDE
RECENTLY: ...
```

**[FRAGE]** Alternative: kompaktes JSON als Eingabe, und wir messen an
echten Laeufen, ob die Ausgabequalitaet leidet. Der Klartext-Renderer und
ein JSON-Renderer koennen parallel existieren (env-Schalter), dann ist das
eine Messung statt einer Meinung.

---

## 4a. Bild-Pipeline (Zustand -> Erzaehler -> Diffusion)

Zwei Schritte, damit das Bild WEDER etwas verliert NOCH nach Liste klingt:

1. **Python baut den `scene_brief`** - das vollstaendige Rohmaterial der
   Runde aus dem Zustand. Nichts kann fehlen.
2. **Der Erzaehler (NARRATE) formt daraus `image_prompt`** - einen
   detaillierten, fluessigen Diffusion-Prompt, setzt Fokus und Kamera.
   Er BEKOMMT den `scene_brief` als Kontext (siehe 3c) und die Anweisung,
   jede genannte Figur und jede `actor_action` aufzugreifen.

Als Sicherung haengt Python nach dem Erzaehler jede anwesende Figur, deren
`look` NICHT im `image_prompt` vorkommt, noch hinten an - so faellt
niemand raus, auch wenn der Erzaehler patzt.

### `Room.visual`

`start_room_anchor` + `start_room_look` (und spaetere Aenderungen) leben in
`Room` als zwei Felder - Zustandslogik braucht "permanent" vs. "jetzt".
Ein Accessor zieht sie zu EINER Beschreibung zusammen (Render UND
`scene_brief`):

```python
Room.visual  ->  f"{anchor}. {look}. {'. '.join(marks)}"
                 # "stone vaulted ceiling, rows of wooden pews, a stone
                 #  altar. dim, dust hanging in the window light. the
                 #  chest stands forced open."
```

**[FRAGE]** oder `anchor`/`look` doch zu EINEM Feld verschmelzen und
Permanenz nur ueber `marks` fuehren?

### `World.scene_brief()` - der Kontext, den NARRATE fuer das Bild bekommt

Deterministisch aus dem Zustand der gerade abgeschlossenen Runde:

```
LOCATION: {Room.visual des Spielerraums}

PRESENT (each: appearance, posture, and what they are visibly doing now):
  {fuer JEDE aktive Figur im Spielerraum, Spieler zuerst:}
  - {name}: {look}; {posture}{; actor_action wenn diese Runde gesetzt}

HAPPENING: {bis zu 2 markante sichtbare Events der Runde}
```

Beispiel (die Kirchenszene):

```
LOCATION: a stone church interior, high vaulted ceiling, rows of wooden
pews, a stone altar, dim light through tall windows

PRESENT:
  - Thomas: a tall man in a black cassock with a long grey beard;
    standing; holding out a coin
  - Marta: a woman in a worn red dress; kneeling; reaching up with an
    open hand

HAPPENING: a coin passes from the priest's hand to the beggar woman's
```

Daraus macht der Erzaehler z.B.:

> *"Interior of a dim stone church, high vaulted ceiling and rows of wooden
> pews, a stone altar in the background, shafts of light through tall
> windows. In the foreground a tall bearded priest in a black cassock
> leans down and places a coin into the open hand of a kneeling woman in a
> worn red dress. Close, warm, candlelit."*

- `diffusion.render()` haengt `STYLE`/`NEGATIVE` weiterhin selbst an.
- Eroeffnungsbild (Szene 1, keine Aktionen): `scene_brief` = nur LOCATION
  + der Spieler unter PRESENT.
- **[FRAGE]** braucht das Bild einen eigenen Aufruf statt des Feldes in
  NARRATE? (Ein Feld ist billiger; ein eigener IMAGE-Aufruf koennte auch
  einen Kamerawinkel / Stil je Szene variieren.)

---

## 5. Perzeption & persistentes Level

### Level-Ledger

- `World.rooms` IST das Ledger. Start: `{"n1": <Spielerraum>}`.
- Neuer Raum nur, wenn eine Aktion ihn erzwingt (Tuer, die nichts sonst
  erklaert; Aufstieg; Ziel, das noch nicht existiert). Kein Raum als
  Belohnung fuers Erkunden. (Regel aus `resolve_player.txt` bleibt.)
- Bei `max_rooms` (12): kein Wachstum mehr, Aktion im Bestand aufloesen.
- Verbindung standardmaessig beidseitig; `one_way` nur mit benennbarem
  physischen Grund.

### Was ein NPC in DECIDE weiss

Kein globales "Ensemble". Was `render_for` einer Figur zeigt, haengt davon
ab, wie sie zur anderen Figur steht:

**1. Sich selbst - voll.**
`goal`, `stance`, `flaw`, `aim`, die eigenen `traits`.

**2. Die Karte - voll.**
Alle Raeume + Exits. Das ist "Zugriff auf das persistente Level" (§4-Doku
gelockert: NPCs kennen die Geografie des Gebaeudes).

**3. Bekannte / alte Rivalen (`bonds[x].known == True`) - Innensicht inkl.**
Figuren, die diese NPC schon vor dem Spiel kennt - Kollegen ODER alte
Gegner. Fuer jede: `name`, `at`, `posture`, `look`, **ihr `goal` und
`aim`**, sowie die `stance` des Bonds (ally/adversary/neutral). So koennen
Verbuendete koordinieren UND Rivalen einander zuvorkommen. `bonds` ist
beidseitig.

**4. Fremde + der Spieler (kein Bond) - nur was wahrgenommen wurde.**
Kein `goal`, kein `aim`. Nur, was in der eigenen `memory` steht (Ereignisse
aus geteilten Raeumen) plus - fuer jeden, der GERADE im selben Raum steht -
`name`, `look`, `posture`, sichtbare Handlung dieser Runde. Wer nie im
selben Raum war, taucht gar nicht auf.

**Nicht enthalten:** globale `facts`, `direction`.

```
YOU ARE c1 Thomas   might high | finesse low | wits average | notice average | presence high
YOUR GOAL: get the engine-room key from the player
YOUR FLAW: you will not step into an unlit space
YOUR AIM:  block the corridor door

PEOPLE YOU KNOW:
  c2 Mara  @n2  moving   - small, hood up   [you: adversary]
           wants: get to the crypt stair first
           now:   reach the sacristy before the player moves the hatch

IN THE ROOM WITH YOU:
  the player  - grey overcoat, crowbar in hand, kneeling by the chest
               just now: forced the chest open
```

- **Aktionen/Perzeption:** Ein `Event` mit `reach: here` erreicht alle
  aktiven Akteure im selben Raum. `reach: adjacent` zusaetzlich alle in
  Raeumen mit einem Exit zum Ereignisraum (Schall durch die Tuer). Jeder
  wahrnehmende Akteur haengt die `clause` an seine `memory`. Der handelnde
  Akteur bekommt sein eigenes Event immer (bestehend).
- `visible(round_log)` fuer NARRATE bleibt: nur was der Spieler an seiner
  damaligen Position mitbekam.

**[FRAGE]** Reicht "im selben Raum + eigene memory" fuer Fremde, oder soll
eine Figur auch mitbekommen, wer im NACHBARRAUM ist (Schritte, Stimmen)?

---

## 6. Story-Richtung & Spannungsbogen  *(pull/pressure umgesetzt)*

Zwei flache Felder auf `World` (nicht das volle `StoryDirection`-Objekt -
das kommt mit dem Rest der Migration):

```python
pull: str = ""       # worauf alle zugezogen werden - ein Objekt, eine
                     # Person, ein Ort, ein Ausgang. Abstrakt bei INIT
                     # gesetzt; wird durch die Handlungen im Spielverlauf
                     # konkret. Jedes NPC-goal ist eine spielerbezogene
                     # Auspraegung davon (6a).
pressure: str = ""   # was es dringend macht - eine Kraft, die auf den Ort
                     # und die Leute darin wirkt. Ebenfalls abstrakt.
# phase (spaeter): reine Arithmetik auf scene - setup 1-4 / commit 5-9 /
# escalate 10-15. Noch nicht implementiert.
```

- **`direction` ist ein Feld im INIT-Schema** (`schema.init_model()`,
  nested `Direction{pull, pressure}`) - das Modell leitet beide aus dem
  Start-Prompt ab, **abstrakt** (nur die Form, nie das konkrete Ding).
  Die [FRAGE] "Modell oder Client" ist damit zugunsten Modell entschieden.
- `from_init` traegt sie nach `World.pull` / `World.pressure`.
- Gehen als `PULL`/`PRESSURE`-Zeilen an **RESOLVE** (`World.render()`) und
  **DECIDE** (`World.render_for()` - JEDE Figur, nicht nur die agentische),
  **nie** an NARRATE (`render_player_place()` ruft `_direction_lines()`
  nicht auf). Prompts (`resolve_*.txt`, `decide.txt`): "bend toward the
  pull, feel the pressure, never voice it".
- `World.hidden_target_leaked()` prueft jetzt `shared_target` **und**
  `pull`/`pressure` woertlich gegen `narrator_text`.
- Der phasenspezifische Regieblock (setup/commit/escalate) ist noch
  Entwurf.

Umgesetzt in: `schema.py` (`Direction`-Modell + `direction`-Feld),
`game_prompts/default/init.txt` (Abschnitt 4 "THE DIRECTION"),
`state.World` (`pull`/`pressure`, `_direction_lines()`, `render`/
`render_for`/`hidden_target_leaked`), `resolve_*.txt` + `decide.txt`.

---

## 6a. NPC-Interaktion ist die Kern-Spielmechanik

Ausdrueckliche Design-Prioritaet: Was der Spieler tut, misst sich an den
NPCs - nicht an Raetseln, nicht am Erkunden. Konkret heisst das:

### Jedes NPC-`goal` laeuft durch den Spieler

Ein `goal` ist nur gueltig, wenn der Spieler darin **vorkommt** - als
einziger, der es geben, blockieren, bedrohen oder ausloesen kann:

    gut:   "he needs the player to hand over the key"
    gut:   "she wants the player gone from this floor before her people arrive"
    gut:   "he thinks the player saw what he did and wants to know for sure"
    schlecht: "he wants to fix the generator"          (Spieler kommt nicht vor)
    schlecht: "she wants to find her sister"           (Spieler nur zufaellig dabei)

`resolve_player.txt` (Abschnitt "NEW CHARACTERS") verbietet das
spielerlose Ziel ausdruecklich; der Client kann es nicht hart pruefen,
aber ein schwaches Ziel faellt im Playtest sofort auf.

### `stance` bestimmt die RICHTUNG des Drucks, nicht die Freundlichkeit

- `ally`: das goal geht nur auf, wenn der Spieler etwas tut/erreicht -
  die Figur draengt ihn also vorwaerts, hilft, oeffnet Wege. Aus
  Eigennutz, nicht aus Sympathie: ein ally kann fordernd, nervig,
  feige sein.
- `adversary`: das goal kollidiert mit dem, was der Spieler will -
  die Figur blockiert, nimmt, droht, verfolgt.

`stance` ist bei Einfuehrung gesetzt und aendert sich normalerweise nicht.
Ein Kippen (ally wird adversary) ist ein Ereignis, kein stiller Wechsel -
und braucht einen sichtbaren Grund im `round_log`.

Zum SPIELER gibt es nur `ally` / `adversary` - eine Figur, die ihn gar
nicht beruehrt, gehoert nicht ins Spiel (6a, "goal laeuft durch den
Spieler").

### `bonds` - wie NPCs zueinander stehen

Jeder NPC hat zu jedem anderen NPC, den er kennt oder trifft, einen `Bond`
(`stance` = ally | adversary | **neutral**, plus `known`). Das ist die
zweite Konfliktachse neben Spieler-vs-NPC:

- zwei Wachen derselben Schicht: `Bond(ally, known=True)` - sie decken
  sich, teilen Wissen, handeln abgestimmt.
- der Priester und die Diebin, die frueher zusammen gearbeitet haben und
  jetzt beide die Krypta wollen: `Bond(adversary, known=True)` - sie
  wissen genau, was der andere vorhat, und kommen ihm zuvor.
- zwei Fremde, die sich im selben Raum begegnen: kein Eintrag -> `neutral`,
  bis eine Handlung daraus etwas macht.

`bonds` wirkt in DECIDE: ein NPC waehlt seinen Schritt auch gegen (oder
mit) die anderen NPCs, nicht nur gegen den Spieler. Der Client kann daraus
Allianzen und offene Fehden zwischen NPCs entstehen lassen, die der Spieler
ausnutzen (oder in die er geraten) kann.

**[FRAGE]** darf sich ein `bond` im Spiel aendern (Verrat unter NPCs), oder
ist er wie `stance` zum Spieler fest, ausser bei einem sichtbaren Ereignis?

### `flaw` - jede Figur hat einen Schwachpunkt

Ein kurzer, **konkreter, physisch oder verhaltensmaessig pruefbarer**
Defekt. Kein Persoenlichkeitsprofil.

    "she lies when cornered"
    "he will not step into an unlit room"
    "he cannot resist a dare"
    "she always overplays her hand"
    "he freezes at the sight of blood"

Wo der `flaw` mechanisch beisst:

- **DECIDE:** `render_for` gibt der Figur ihren `flaw`; der decide-Prompt
  weist an, ihn die Wahl faerben oder verderben zu lassen (der Feigling
  nimmt den Umweg, der Luegner sagt etwas Falsches).
- **RESOLVE(agentic):** der Resolver darf eine Aktion am `flaw` scheitern
  oder nach hinten losgehen lassen - "he will not step into an unlit room"
  heisst, sein `aim` bricht an der dunklen Tuer ab, auch wenn er sonst
  durchgekommen waere.
- **Fuer den Spieler:** der `flaw` ist der Hebel. Er sieht ihn nicht als
  Feld, aber die Narration zeigt ihn (Zittern, Ausweichen, ein zu
  glatter Satz), und er kann ihn ausnutzen.

### Folgen fuers System

- NARRATE stellt anwesende NPCs und ihre Handlung in den Vordergrund;
  ein leerer Raum ohne Figur ist die Ausnahme, nicht die Norm.
- Das Spiel laeuft ohnehin bis Szene 15 (8), es gibt also kein "zu frueh
  vorbei". Die escalate-Phase (10-15) soll offene NPC-Konflikte
  ZUSPITZEN, nicht abbinden - der Abschluss kommt erst in Szene 15.
- Neue Raeume bleiben selten (Regel aus 5) - die Zuege gehoeren den
  Figuren, nicht der Geografie.

---

## 7. Agentic-NPCs (bis zu 4)

- Der agentische Loop laeuft, **sobald der erste NPC existiert**.
- Beim Einfuehren wird ein NPC `agentic = True`, solange
  `count(agentic) < max_agentic` (4). Danach eingefuehrte NPCs sind
  `agentic = False`.
- `advance()` verarbeitet pro Runde: **1x Spieler-RESOLVE**, dann
  **je agentischem NPC** (in Spawn-Reihenfolge, nur aktive) **DECIDE ->
  RESOLVE(agentic)**, dann **1x NARRATE**.
- Nicht-agentische NPCs (5+): kein eigener Zug. Sie sind anwesend,
  wahrnehmbar und wahrnehmend; ihr Verhalten faellt in NARRATE.
- Fehlerisolierung wie heute: scheitert DECIDE/RESOLVE eines NPC, entfaellt
  nur dessen Zug.

**Kosten:** 4 agentische NPCs = bis zu `1 + 4*2 + 1 = 10` LLM-Aufrufe pro
Runde. **[FRAGE]** ist das akzeptabel, oder soll `max_agentic` kleiner
sein (2?) bzw. nur die N naechsten NPCs am Spieler agentisch handeln?

`shared_target` heute nur zwischen Spieler + 1 agentischem NPC. Neu: jeder
agentische NPC hat sein eigenes, spielerbezogenes `goal` (6a), das
mittelbar an `direction.pull` zieht - kein geteiltes verborgenes Feld. Das
entfaellt damit; nur `direction` bleibt (und nur fuer RESOLVE/DECIDE).

---

## 8. Charakter-Quote  *(umgesetzt gegen das aktuelle Schema)*

War: 3 Figuren bis Zug 5. Jetzt: **eine einzige Figur**, und die steht
meist schon von Anfang an da, weil INIT 0-2 NPCs in den Startraum setzt
(3d). Der Hinweis greift also nur noch, wenn WIRKLICH niemand da ist.

```python
def character_quota_status(self, turn_number: int) -> str:
    if self.characters:      # irgendeine Figur -> erfuellt
        return ""
    if turn_number > 5:
        return "MANDATORY: this call MUST introduce a character."
    left = max(0, 5 - turn_number)
    return (f"STORY DIRECTION: a character must appear by turn 5. "
            f"{left} turn(s) remain. Introduce one where the scene gives "
            f"a natural occasion.")
```

- Notbremse (`spawn_fallback_character`) + Retry-Schleife in `advance()`
  bleiben, greifen jetzt bei 0 Figuren nach Zug 5.
- Der **erste** NPC im Spiel wird `agentic` (`_introduce_characters`,
  siehe 7) - egal ob aus INIT oder aus einem spaeteren Zug. Nicht mehr
  "der dritte".

Umgesetzt in: `schema.init_model()` (`starting_characters`-Feld),
`game_prompts/default/init.txt` (Abschnitt 3 "WHO IS IN THE ROOM"),
`state.World.from_init` (Figuren anlegen via `_introduce_characters`),
`state.World.character_quota_status` (3 -> 1),
`state.World._introduce_characters` (erste Figur agentic).
Der Rest von §2-7 (Actor/traits/bonds/`direction`) bleibt Entwurf.

### Spielende - Szene 15 oder Tod

Genau zwei Enden, beide im Client entschieden (NARRATE urteilt nicht, kein
`can_end` mehr):

```python
def is_over(world) -> bool:
    return world.player.status == "dead" or world.scene >= world.last_scene
```

- **`scene >= 15`** (`last_scene`): das regulaere Ende. Die 15. Szene
  bekommt einen Client-Hinweis "final scene - bring it to a close", der
  Erzaehler schreibt einen Abschluss. Kein frueher Ausstieg, wenn "die
  Lage sich beruhigt hat" - `MIN_SCENES`/`can_end` sind gestrichen.
- **`player.status == "dead"`**: sofortiges Ende in JEDER Szene. Die Runde
  wird noch fertig erzaehlt (NARRATE mit Hinweis "the player has died -
  narrate the end"), dann Schluss. Der Spieler muss Szene 15 nicht
  erreichen.
- **`disabled`** beendet das Spiel NICHT (siehe "Scheitern & Tod" in 2) -
  offene [FRAGE], ob es das ab einer bestimmten Eskalation soll.

`MAX_SCENES = 15` bleibt als Konstante, `MIN_SCENES` entfaellt.

---

## 9. Migrationsschritte (nach Freigabe des Formats)

1. `state.py`: `Actor` einfuehren (inkl. `traits` mit 5 Feldern, `bonds`),
   `player` als `Actor`, `Character` -> in `Actor` aufgehen lassen.
   `Item`, `Posture`, `ItemState`, `EventReach`, `Rating`, `Traits`,
   `Stance`(+`neutral`), `Bond`, `StoryDirection` ergaenzen. `Node` ->
   `Room` (+ `look`, `items`). Position bleibt `Actor.at` (nicht in `Room`
   doppeln, siehe 2). Marks von transienten Zustaenden bereinigen.
   `max_rooms` 12 -> 15, `last_scene` = 15, `MIN_SCENES`/`can_end` raus.
2. `schema.py`: RESOLVE-Delta um `actor_status`, `actor_action`, `posture`,
   `posture_changes`, `item_ops` erweitern; `NewCharacter` von
   `agenda_draft`/`agenda_target_hint` auf `name`(Eigenname)/`look`/
   `posture`/`traits`(5)/`goal`/`stance`/`flaw`/`bonds` umstellen; `Event`
   um `reach`. DECIDE um `posture`. NARRATE: `can_end` RAUS, `image_prompt`
   bleibt (aus `scene_brief` gespeist). INIT auf `start_room_look`/
   `player_look`/`player_traits`(5) erweitern (+ evtl. `direction`);
   `opening_narration` behaelt seine Rolle (Story eroeffnen, Kontext aus
   dem Start-Prompt), MUSS aber zusaetzlich Figur + ersten Raum nennen.
   **`starting_characters` (0-2) ist bereits umgesetzt** (siehe 8) - hier
   nur noch von `name`/`agenda_*` auf die vollen NPC-Felder erweitern.
3. `state.apply_turn`: `item_ops`, Inventar-Cap, `posture`-Anwendung,
   `actor_status` auf den handelnden Akteur, `actor_action` an der
   Figur/Runde ablegen, `bonds` beidseitig eintragen, `reach`-basierte
   Perzeption.
4. `story.advance`: agentischer Loop ueber bis zu 4 NPCs; `is_over()` =
   Spielertod ODER Szene 15; letzte Szene + Tod-Szene bekommen einen
   Abschluss-Hinweis an NARRATE.
5. `state.render` / `render_for`: neue Felder; Karte + beziehungsabhaengige
   Sicht (bonds known -> Innensicht, Fremde -> nur Wahrnehmung) +
   eigene `traits` in `render_for`.
6. `state.py`: `Room.visual` + `World.scene_brief()`. `story._narrate`
   uebergibt `scene_brief()` als Bild-Kontext an NARRATE; danach haengt
   Python fehlende anwesende Figuren an `image_prompt` an, bevor es an
   `diffusion.render()` geht.
7. `StoryDirection`: INIT-Anbindung + phasenspezifischer Regieblock.
8. Prompts (`init.txt`, `resolve_*.txt`, `decide.txt`, `narrate.txt`):
   neue Felder, Eigenname-vs-look-Regel, `actor_action`-Regel (kurze
   Praesens-Klausel), `actor_status`/Scheitern-mit-Folgen + Bestrafung
   absurder Handlungen, Marks-vs-Item-vs-Posture-Regel,
   Traits-Aktionsaufloesung (5 Werte), `wits`->Sprachregister,
   `bonds` (NPC-vs-NPC) in DECIDE, Phasen-Regie, "goal muss durch den
   Spieler laufen"-Regel, `flaw` in DECIDE/RESOLVE(agentic). `init.txt` +
   `narrate.txt`: Traits der Spielerfigur NIE benennen, nur als
   koerperliche Reibung zeigen und nicht vorne draufladen (siehe 2).
   `narrate.txt`: `image_prompt` aus dem `scene_brief` bauen, jede Figur +
   `actor_action` aufgreifen; Abschluss-Hinweis in letzter/Tod-Szene.
9. Tests: Quote (1 statt 3), Inventar-Cap, `reach`-Perzeption, Posture im
   Delta, `traits`-Aktionsaufloesung (`might low` -> Hechtsprung
   scheitert), `actor_status: dead` beim Spieler -> `is_over()` True vor
   Szene 15, Szene 15 -> `is_over()` True, absurde Aktion -> Bestrafung
   statt Abweisung, agentischer Multi-NPC-Loop, `render_for`-Sicht (Bond
   `known` sieht goal/aim, Fremder nicht), `bonds` beidseitig, `neutral`
   als Default zwischen Fremden, `scene_brief()` nennt Raum + jede
   anwesende Figur + `actor_action`, Python-Sicherung haengt fehlende
   Figuren an, Leak-Check fuer `direction` UND fuer Trait-Etiketten in
   `narrator_text`, `stance`/`flaw`/`traits`(5)/`bonds`/Eigenname im
   eingefuehrten Charakter, INIT-Narration eroeffnet die Story UND nennt
   Raum + deutet die Figur an.

---

## Offene Fragen (Sammlung)

Geklaert: Spieler-`traits` kommen bei INIT aus dem Start-Prompt; Traits
sind 5 breite Kategorien (`might`/`finesse`/`wits`/`notice`/`presence`,
D&D/Fallout-angelehnt); Position lebt im Actor-Ledger; kein globales
Ensemble - Wissen ueber andere haengt an `bonds` bzw. Wahrnehmung; NPCs
haben `bonds` (stance + known) auch untereinander, nicht nur zum Spieler;
`opening_narration` eroeffnet weiter die Story UND muss Figur + Raum
enthalten; Bild: Python baut `scene_brief`, NARRATE formt `image_prompt`;
`max_rooms` = 15; Spiel endet IMMER Szene 15 oder frueher durch Spielertod
(`can_end`/`MIN_SCENES` gestrichen); Aktionen koennen scheitern und toeten,
absurde Handlungen werden bestraft.

Offen:

- **[4]** Klartext-Render behalten oder JSON-Eingabe gegen Klartext messen
  (beide Renderer parallel, env-Schalter)?
- **[4a]** `Room`: `anchor`/`look` zwei Felder lassen (permanent vs. jetzt)
  oder zu einem verschmelzen?
- **[4a]** `image_prompt` als Feld in NARRATE (billig) oder eigener
  IMAGE-Aufruf (kann Kamera/Stil je Szene variieren, +1 Aufruf/Runde)?
- **[5]** Fremde/Spieler: reicht "im selben Raum + eigene memory", oder
  soll eine Figur auch Praesenz im NACHBARRAUM mitbekommen (Schritte,
  Stimmen)?
- **[6]** `pull`/`pressure` bei INIT vom Modell (eigenes Feld) oder vom
  Client aus dem Start-Prompt?
- **[6a]** `stance` (zum Spieler) vom Modell gesetzt - oder Client (z.B.
  abwechselnd ally/adversary)?
- **[6a]** duerfen sich `bonds` im Spiel aendern (Verrat unter NPCs), oder
  nur bei sichtbarem Ereignis wie `stance` zum Spieler?
- **[7]** `max_agentic = 4` -> bis zu 10 LLM-Aufrufe/Runde. Akzeptabel,
  oder kleiner / nur die N naechsten am Spieler?
- **[8]** beendet `disabled` beim Spieler das Spiel ab einer bestimmten
  Eskalation, oder nie?
- **Posture-Enum:** `standing/kneeling/sitting/lying/moving/jumping` -
  `crouching` ergaenzen ("kauern" != "knien")?
- **`flaw`:** freier Text oder Enum aus ~12 Archetypen (`liar`, `coward`,
  `greedy`, `reckless`, ...)?
- **ItemState-Enum:** `intact/open/broken/lit` - reicht der Satz?
- **`wits`-Sprachregister:** nur NPC-`utterance`, oder auch die Narration
  der Spielerfigur (dumme Spielerfigur -> einfachere Erzaehlsprache)?
- **`traits`:** reichen 5, oder braucht "Toughness/einen Treffer wegstecken"
  einen eigenen Wert neben `might`?

---

## Anhang A: Vollstaendiger Zustand als JSON

Ein Mittelspiel-Zustand (Szene 7), der jedes Feld einmal zeigt. `phase`
ist kein Feld - sie wird aus `scene` gerechnet (7 -> `commit`). Kommentare
`//` sind nur Erlaeuterung, echtes JSON hat sie nicht.

```jsonc
{
  "language": "de",
  "scene": 7,                   // Spiel endet Szene 15 oder bei Spielertod
  "max_rooms": 15,
  "max_agentic": 4,
  "last_scene": 15,

  // --- Story-Richtung: NUR fuer RESOLVE/DECIDE, nie im Wortlaut an NARRATE ---
  "direction": {
    "pull": "the locked crypt beneath the sacristy floor",
    "pressure": "the bishop's car is minutes from the gate; once it arrives the church is sealed for the night"
  },

  "facts": [
    "the power to the church is out",
    "heavy rain is coming down outside"
  ],
  "recent": [
    "Regen trommelt auf die hohen Fenster. Du drueckst dich in die Sakristei, die Tuer faellt hinter dir ins Schloss.",
    "Das Schloss des Reliquiars gibt unter dem Brecheisen nach. In der Kassette liegt nur ein kleiner Messingschluessel.",
    "Aus dem Kirchenschiff ruft eine schwere Stimme deinen Namen. Schritte, langsam, kommen naeher."
  ],
  "round_log": [],   // transient - faengt jede Runde leer an

  // =========================================================== SPIELER
  "player": {
    "id": "player",
    "name": "",                 // 2. Person; der Start-Prompt definiert die Figur
    "at": "n2",
    "posture": "kneeling",
    "status": "active",
    "look": "lean, a soaked grey overcoat torn at one sleeve",
    "traits": { "might": "average", "finesse": "high", "wits": "average", "notice": "average", "presence": "low" },
    "inventory": [
      { "id": "i1", "name": "Brecheisen",     "look": "scarred steel pry bar", "state": "intact" },
      { "id": "i2", "name": "Messingschluessel", "look": "small, worn, no head stamp", "state": "intact" }
    ],
    "memory": [
      "the reliquary lid splinters loose",
      "Thomas calls your name from the nave",
      "the bell rope creaks somewhere above"
    ]
    // keine goal/stance/flaw/aim/bonds/agentic beim Spieler
  },

  // ====================================================== NPC-LEDGER
  "actors": {
    "c1": {
      "id": "c1",
      "name": "Thomas",
      "at": "n1",
      "posture": "standing",
      "status": "active",
      "look": "heavy-set, a black cassock, a long grey beard",
      "traits": { "might": "high", "finesse": "low", "wits": "average", "notice": "high", "presence": "high" },
      "inventory": [],
      "memory": [
        "the north door bangs open",
        "footsteps cross the sacristy",
        "Mara's voice up in the bell tower"
      ],
      "goal": "he wants the player out of the church before the bishop's car reaches the gate",
      "stance": "adversary",           // zum Spieler
      "bonds": {
        "c2": { "stance": "adversary", "known": true },   // alter Kompagnon, jetzt Rivale um die Krypta
        "c3": { "stance": "ally", "known": true }         // sein Kuester
      },
      "flaw": "he will not step into an unlit room",
      "aim": "block the sacristy doorway and talk the player back outside",
      "agentic": true                  // erster NPC -> agentic
    },
    "c2": {
      "id": "c2",
      "name": "Mara",
      "at": "n3",
      "posture": "standing",
      "status": "active",
      "look": "small, quick, a dripping hooded jacket",
      "traits": { "might": "average", "finesse": "high", "wits": "high", "notice": "high", "presence": "average" },
      "inventory": [
        { "id": "i7", "name": "Dietrich-Set", "look": "rolled leather case of picks", "state": "intact" }
      ],
      "memory": [
        "the reliquary is already forced open",
        "someone else is down in the sacristy",
        "the tower door is jammed shut"
      ],
      "goal": "she needs the player to force the crypt stair open - she cannot shift it alone",
      "stance": "ally",                 // zum Spieler: stuetzend aus Eigennutz - kann kippen
      "bonds": {
        "c1": { "stance": "adversary", "known": true }
      },
      "flaw": "she cannot resist bragging about a job",
      "aim": "get down to the sacristy and talk the player into helping with the stair",
      "agentic": true
    },
    "c3": {
      "id": "c3",
      "name": "Piet",
      "at": "n1",
      "posture": "sitting",
      "status": "active",
      "look": "old, stooped, one sleeve pinned up over a missing arm",
      "traits": { "might": "low", "finesse": "low", "wits": "average", "notice": "low", "presence": "average" },
      "inventory": [
        { "id": "i8", "name": "Schluesselring", "look": "a heavy iron ring of church keys", "state": "intact" }
      ],
      "memory": [
        "the priest is shouting a name",
        "cold air pouring from the open north door"
      ],
      "goal": "he wants the player blamed for the break-in so the bishop leaves his lodge alone",
      "stance": "adversary",           // zum Spieler
      "bonds": {
        "c1": { "stance": "ally", "known": true }
      },
      "flaw": "he is half-deaf and mishears instructions",
      "aim": "stay by the door and make sure it is the player who is seen leaving",
      "agentic": true
    }
    // ein 4. NPC waere noch agentic (max_agentic 4), ein 5. nicht mehr
  },

  // ==================================================== LEVEL-LEDGER
  "rooms": {
    "n1": {
      "id": "n1",
      "name": "Kirche",
      "anchor": "a long stone nave, rows of dark oak pews, a raised stone altar, tall pointed windows",
      "look": "unlit but for one guttering altar candle; rain streaking the glass",
      "exits": [
        { "to": "n2", "one_way": false, "justification": "" },
        { "to": "n3", "one_way": false, "justification": "" }
      ],
      "items": [
        { "id": "i3", "name": "Altarleuchter", "look": "tall brass, a single candle burning", "state": "lit" }
      ],
      "marks": [
        "the north door hangs open",
        "a pew is shoved out of its row"
      ]
    },
    "n2": {
      "id": "n2",
      "name": "Sakristei",
      "anchor": "a small vestry, a vestment cupboard, a low wooden bench, an iron hatch set in the floor",
      "look": "papers pulled from the cupboard and strewn across the bench",
      "exits": [
        { "to": "n1", "one_way": false, "justification": "" }
        // die Kryptatreppe existiert noch NICHT - sie ist der `pull`,
        // ein kuenftiger new_room, wenn jemand die Luke aufbricht
      ],
      "items": [
        { "id": "i4", "name": "Reliquienkasten", "look": "carved wood, the lid forced", "state": "open" },
        { "id": "i6", "name": "Bodenluke", "look": "heavy, ring-handled, iron, set flush in the floor", "state": "intact" }
      ],
      "marks": [
        "the reliquary lid lies on the floor",
        "the vestment cupboard stands emptied"
      ]
    },
    "n3": {
      "id": "n3",
      "name": "Glockenturm",
      "anchor": "a narrow stone shaft, a wooden ladder to the bell platform, one bronze bell on a rotted headstock",
      "look": "wind driving rain in through the louvres",
      "exits": [
        { "to": "n1", "one_way": false, "justification": "" }
      ],
      "items": [
        { "id": "i5", "name": "Glockenseil", "look": "frayed, hanging from the headstock", "state": "intact" }
      ],
      "marks": []
    }
  }
}
```

Abgeleitet (nicht gespeichert):

- **`phase`** = `commit` (aus `scene` 7; setup 1-4, commit 5-9, escalate 10-15).
- **`is_over()`** = False (Spieler `active`, Szene 7 < 15).
- **Wer ist in n1** = `[c1, c3]` (Abfrage ueber `actor.at`).
- **`Room.visual` von n1** = `"a long stone nave, rows of dark oak pews, a
  raised stone altar, tall pointed windows. unlit but for one guttering
  altar candle; rain streaking the glass. the north door hangs open. a pew
  is shoved out of its row."`
- **`scene_brief()`** (Spieler in n2, allein): LOCATION = `Room.visual` von
  n2; PRESENT = nur der Spieler; HAPPENING = aus dem `round_log` der Runde.
