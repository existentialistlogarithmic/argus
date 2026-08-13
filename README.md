# Argus

**An ontology-driven intelligence platform.** Messy records from many source
systems become resolved real-world entities; resolved entities become a typed
knowledge graph; the graph becomes findings an analyst can act on — with
provenance preserved end to end.

Pure Python standard library. No dependencies are required to ingest, resolve,
build the graph, run the analytics, or query it.

```bash
git clone <this repo> && cd ai-research
python3 -m argus demo
```

---

## The problem this solves

A national registry knows *Robert J. Whitlock, b. 1974-03-08, passport
K37186034*. An HR export knows *Bob Whitlock, bob.w@corp.com*. A payment file
knows *account 6655106336*. Comms metadata knows only *+44 712 345 6789*.

Four systems, four fragments, one person — and **not one of them contains the
network he sits in**. The network is in the *joins*, and the joins do not exist
until something decides those four fragments are the same human being.

That decision is the hard part, and everything downstream is worthless if it is
wrong. Two failure modes, both fatal:

- **Under-merging** leaves one person as five entities. The network never
  appears, because its edges are scattered across fragments.
- **Over-merging** fuses unrelated people into one entity. That *invents*
  relationships — far worse, because it manufactures evidence against innocent
  parties.

## What it does

```
 sources          ingest            resolve             graph            analyse
┌────────┐      ┌────────┐      ┌───────────┐      ┌──────────┐      ┌──────────┐
│ 7 sys- │─────▶│ typed  │─────▶│ Fellegi-  │─────▶│  typed   │─────▶│typologies│
│ tems,  │      │ records│      │ Sunter    │      │ entities │      │ + risk   │
│ 7 sche-│      │ + link │      │ + conflict│      │ + rewired│      │ + cases  │
│ mas    │      │ obs.   │      │ clustering│      │   edges  │      │          │
└────────┘      └────────┘      └───────────┘      └──────────┘      └──────────┘
                                       │                                   │
                                  ground truth ──▶ precision / recall      ▼
                                                                    query + NL
```

Everything — object types, their properties, how each property is normalized
and compared, how much evidence it carries, and how objects relate — is
declared in [`ontology/intel.json`](ontology/intel.json). Swapping domains
means swapping the JSON, not editing the engine.

---

## Results

Run against a synthetic corpus with **known ground truth** (300 people, 80
companies, 516 accounts, ~9,000 source records across 7 systems, each with its
own schema and its own characteristic corruptions):

```
type            records    true   found    prec  recall      f1   exact
--------------------------------------------------------------------------
account           1,015     516     516  1.0000  1.0000  1.0000 100.00%
organization        256      80      84  1.0000  0.9691  0.9843  95.00%
person            7,338     300     386  1.0000  0.9140  0.9551  76.00%
--------------------------------------------------------------------------
OVERALL           8,609                  1.0000  0.9146  0.9554
```

**Precision 1.0 — zero false merges.** No two distinct real-world objects were
ever fused. Recall 0.91 is the deliberate other side of that trade: the system
would rather leave a record unattributed than invent a link.

Reproduce it yourself — the corpus, the resolution, and the metrics are all
deterministic:

```bash
python3 -m argus eval
```

### The honest part

The unresolved 9% is mostly **comms selectors that never co-occur with an
identifying attribute** — an email address that appears in no other system has
no evidence to link it to anyone, and the correct answer is to leave it as an
unattributed selector rather than guess. `eval` prints concrete examples of
every fragmentation and every false merge, because an aggregate F1 tells you
the system is imperfect without telling you *how*.

---

## Finding the network

The corpus has a network planted inside it: six principals, five shell
companies sharing one registered address, incorporated within a 90-day window,
moving money in a closed loop in amounts sitting just under a 10,000 reporting
threshold. **Only one of the six is on the watchlist.** No single source reveals
the shape.

From that one seed, all six principals rank 1–6 by risk:

```
rank  risk   ring?  name
   1  0.944  RING   Michael E Underwood      <- the only watchlisted one
   2  0.930  RING   Chen Kowalski
   3  0.908  RING   Susan E Nakamura
   4  0.904  RING   Elizabeth A Whitllock
   5  0.901  RING   Victoria E Ingram
   6  0.875  RING   Peter A Jarvis
   7  0.851         Christopher J Tanaka     <- shell-company officers:
   8  0.822         Isabella Castellanos        genuine leads, correctly
   9  0.816         Ekaterina Duarte            ranked below the principals
```

Four independent typologies fire and correlate into a single case:

| Typology | What it looks for |
|---|---|
| `circular_flow` | Value that leaves an account and returns to it — no commercial purpose |
| `structuring` | Payments clustered just under a reporting threshold, never above it |
| `shared_registration` | One address hosting many companies with overlapping officers |
| `rapid_incorporation` | Companies sharing officers *and* incorporated in a tight window |

Findings fire on whatever object carries the behaviour — usually a bare account
number. An account number is not a suspect, so each finding is walked back
along ownership and directorship links to the people and companies behind it.
That attribution is also what lets findings on *different object types* be
recognised as describing one network.

---

## Design decisions worth arguing about

**Probabilistic linkage, not rules.** Each property contributes a
log-likelihood ratio: `log2(m/u)`, where `m` is how often it agrees for true
matches and `u` is how often it collides by chance. A national ID outweighs a
first name by orders of magnitude *without anyone tuning a weight* — the ratio
falls out of the value space. Missing data contributes zero: a source that
never collects a field must not be punished for its silence.

**Clustering is conflict-aware, not transitive closure.** A~B and B~C does not
make A~C. Plain union-find over pairwise matches will happily chain a whole
population into one blob. Merges are applied strongest-evidence-first and
*vetoed* when two clusters assert different values for an exclusive identifier.

**Links are rewired, not pre-joined.** Ingest emits relationships between
*source-record* keys, because entities do not exist yet. The graph builder
remaps them after resolution. This is where the payoff lands: a transaction
observed between two bare account numbers becomes an edge between two accounts
whose owners are known.

**The LLM writes queries; it never touches the graph.** A model that answers
from the graph can hallucinate an entity, an edge, or a number, and the analyst
cannot tell. A model that only writes a query can be wrong — but *visibly*, in
one line of text the analyst can read, correct, and re-run. Every generated
query is parsed before it executes and printed alongside the result.

**Louvain, not label propagation.** Label propagation was the cheap obvious
choice and it fails badly here: on a graph with a dense giant component every
node adopts its neighbours' label and the population collapses into a single
community with modularity ≈ 0 — worse than useless, because it looks like an
answer. Louvain's hierarchical contraction gets modularity 0.44 and keeps small
tight groups intact inside large loose ones.

**Risk accumulates as noisy-OR, and seeds are not pinned to 1.0.** Summing
severities pins everything a case touches to the maximum within three findings,
destroying the ranking. And pinning known subjects to 1.0 ties them at the top
— erasing exactly the comparison the ranking exists to support: *which
not-yet-known entities look worst.*

---

## Using it

```bash
python3 -m argus demo                    # full walkthrough, start to finish
python3 -m argus generate --out data     # regenerate the corpus (deterministic)
python3 -m argus ontology                # print the loaded ontology
python3 -m argus run --out artifacts     # full pipeline, write graph + findings
python3 -m argus eval                    # score resolution against ground truth
python3 -m argus cases                   # ranked correlated cases
python3 -m argus risk person             # highest-risk entities
python3 -m argus dossier PER-000023      # full profile with provenance
python3 -m argus connect PER-000023 ORG-000052
```

### Show your work

Every identity decision can be interrogated:

```console
$ python3 -m argus explain natreg:NR00023 watch:WL0000

  natreg:NR00023   {"full_name": "Michael E UNDERWOOD", "dob": "1995-12-01", ...}
  watch:WL0000     {"full_name": "Michael Underwood", "dob": "30/11/1995", ...}

natreg:NR00023  <->  watch:WL0000   score=+18.61  [match]
    + dob            sim=0.900  weight=+10.10
    + full_name      sim=0.950  weight=+4.94
    + nationality    sim=1.000  weight=+3.57

    match threshold  +9.00   review threshold +4.50
```

Note the watchlist entry's date of birth is *wrong* by one day. Partial
agreement keeps most of its weight, because a transposed digit is far more
common than a different person.

### Querying

```bash
python3 -m argus query 'find person where risk > 0.8 and degree > 10 order by risk limit 5'
python3 -m argus query 'find organization linked to PER-000023 via officer_of within 2'
python3 -m argus query 'path PER-000023 to ORG-000052 within 4'
```

```
find <type> [where <expr>] [linked to <ref> [via <link,...>] [within <n>]]
            [order by <field> [asc|desc]] [limit <n>]
path <ref> to <ref> [within <n>] [limit <n>]
show <ref>

  <op>     =  !=  >  <  >=  <=  ~   (~ is "contains")
  <field>  any ontology property, or: risk, degree, type, id, label,
           sources, records, community, flags.<name>
```

### Natural language

```bash
python3 -m argus ask "who are the riskiest people?"
python3 -m argus ask "how is Michael E Underwood connected to Chen Kowalski?"
```

Works with no API key — a deterministic pattern-based translator handles the
common question shapes offline. Set `ANTHROPIC_API_KEY` (and
`pip install anthropic`) to route translation through Claude instead; the SDK is
imported lazily, so the engine and the whole test suite still run on a machine
that has never installed it. Override the model with `ARGUS_MODEL`.

---

## Layout

| Module | Responsibility |
|---|---|
| `argus/ontology.py` | Typed entity/link schema, per-property match parameters, blocking rules |
| `argus/normalize.py` | Value normalizers — names, orgs, dates, phones, emails, addresses |
| `argus/similarity.py` | Jaro-Winkler, edit distance, token alignment, date proximity |
| `argus/corpus.py` | Synthetic multi-source generator with ground truth and a planted network |
| `argus/ingest.py` | One adapter per source system; emits records and link observations |
| `argus/resolve.py` | Blocking, Fellegi-Sunter scoring, conflict-aware clustering, evaluation |
| `argus/graph.py` | Knowledge graph, link rewiring, traversal, path enumeration |
| `argus/analytics.py` | Centrality, Louvain, personalized PageRank, typologies, risk |
| `argus/query.py` | Structured query language — tokenizer, parser, executor |
| `argus/nl.py` | Pluggable natural-language → query translation |
| `argus/pipeline.py` | End-to-end orchestration and the analyst-facing surfaces |

```bash
python3 -m unittest discover -s tests      # 112 tests, ~0.4s
```

## Extending it

Add an entity type or property to `ontology/intel.json` and it is immediately
normalized, blocked, compared, queryable, and describable in natural language —
the NL schema prompt is generated from the live ontology, so nothing else
changes. To point the platform at a real domain, write one adapter per source
in `ingest.py` and supply an ontology; the resolver, graph, analytics, and
query layers are domain-agnostic.
