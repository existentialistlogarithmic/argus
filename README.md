# Argus

Entity resolution and link analysis over messy multi-source data.

Records from several source systems get resolved into real-world entities,
those entities become a typed knowledge graph, and the graph gets analysed for
patterns worth investigating. Provenance is kept at every step, so any
conclusion traces back to the rows it came from.

Python 3.10+, standard library only. Nothing to install.

```bash
git clone https://github.com/existentialistlogarithmic/argus
cd argus
python3 -m argus demo
```

## The problem

A national registry has *Robert J. Whitlock, b. 1974-03-08, passport K37186034*.
An HR export has *Bob Whitlock, bob.w@corp.com*. A payment file has *account
6655106336*. Comms metadata has *+44 712 345 6789* and nothing else.

Four systems, four fragments, one person. None of them contains the network he
sits in, because the network lives in the joins, and the joins don't exist
until something works out that those four fragments are the same human.

Getting that wrong ruins everything downstream, in two directions:

- Under-merge and one person stays five entities. The network never shows up,
  because its edges are spread across the fragments.
- Over-merge and unrelated people become one entity. That invents
  relationships. Much worse, since it manufactures evidence against people who
  had nothing to do with each other.

## How it works

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

Object types, their properties, how each property is normalized and compared,
how much evidence it carries, and how objects relate are all declared in
[`ontology/intel.json`](ontology/intel.json). The engine has no idea what a
"person" is. Changing domain means changing that file.

## Results

Measured against a synthetic corpus with known ground truth: 300 people, 80
companies, 516 accounts, roughly 9,000 records across 7 systems, each with its
own schema and its own corruptions.

```
type            records    true   found    prec  recall      f1   exact
--------------------------------------------------------------------------
account           1,015     516     516  1.0000  1.0000  1.0000 100.00%
organization        256      80      84  1.0000  0.9691  0.9843  95.00%
person            7,338     300     386  1.0000  0.9140  0.9551  76.00%
--------------------------------------------------------------------------
OVERALL           8,609                  1.0000  0.9146  0.9554
```

Precision 1.0, so no false merges: no two distinct real-world objects were
fused. Recall 0.91 is the other side of that trade. The system leaves a record
unattributed rather than guess.

```bash
python3 -m argus eval
```

The unresolved 9% is mostly comms selectors that never co-occur with an
identifying attribute. An email address appearing in no other system has no
evidence linking it to anyone, and leaving it as an unattributed selector is
the correct answer. `eval` prints concrete examples of every fragmentation and
every false merge, since an aggregate F1 says the system is imperfect without
saying how.

## Finding the network

There's a network planted in the corpus: six principals, five shell companies
sharing a registered address, incorporated inside a 90-day window, moving money
in a closed loop in amounts just under a 10,000 reporting threshold. One of the
six is on the watchlist. No single source shows the shape.

Starting from that one seed, all six rank 1-6 by risk:

```
rank  risk   ring?  name
   1  0.952  RING   Chen R Yarrow            <- the only watchlisted one
   2  0.940  RING   Ekaterina C Zielinski
   3  0.937  RING   Leila J Kowalski
   4  0.929  RING   Julian Novak
   5  0.921  RING   Anthony Petrov
   6  0.907  RING   Isabella R Grimaldi
   7  0.803         Anthony Quintero         <- officers of the shells:
   8  0.794         Yuki Jarvis                 real leads, ranked lower
```

Four typologies fire and correlate into one case:

| Typology | Looks for |
|---|---|
| `circular_flow` | Value leaving an account and returning to it |
| `structuring` | Payments clustered under a reporting threshold, never above |
| `shared_registration` | One address hosting many companies with overlapping officers |
| `rapid_incorporation` | Companies sharing officers and incorporated close together |

Typologies fire on whatever object carries the behaviour, which for financial
patterns is an account number. An account number isn't a suspect, so each
finding gets walked back along ownership and directorship links to the people
and companies behind it. That attribution is also how findings on different
object types get recognised as one network.

## Notes on the approach

**Probabilistic linkage rather than rules.** Each property contributes a
log-likelihood ratio, `log2(m/u)`, where `m` is how often it agrees for true
matches and `u` how often it collides by chance. A national ID outweighs a
first name by orders of magnitude and there's no weight to tune, because the
ratio falls out of the value space. Missing data contributes zero. A source
that doesn't collect a field shouldn't be penalised for it.

**Conflict-aware clustering, not transitive closure.** A~B and B~C doesn't make
A~C. Plain union-find over pairwise matches will chain a whole population into
one blob. Merges apply strongest-evidence-first and get vetoed when two
clusters hold different values for an exclusive identifier.

**Links are rewired, not pre-joined.** Ingest emits relationships between
source-record keys, since entities don't exist yet at that point. The graph
builder remaps them after resolution. This is where the payoff lands: a
transaction between two bare account numbers becomes an edge between two
accounts whose owners are known.

**The LLM writes queries, it doesn't touch the graph.** A model answering from
the graph can hallucinate an entity or a number and there's nothing to check it
against. A model that only writes a query can still be wrong, but wrong in one
visible line you can read and re-run. Every generated query gets parsed before
it executes and printed next to the result.

**Louvain instead of label propagation.** Label propagation was the cheap first
attempt and it collapsed the entire graph into one community with modularity
near zero, which looks like an answer but isn't. Louvain gets 0.44 and keeps
small tight groups intact inside large loose ones.

**Risk uses noisy-OR, and seeds aren't pinned to 1.0.** Summing severities pins
everything a case touches to the maximum within three findings and destroys the
ranking. Pinning known subjects to 1.0 ties them at the top, which erases the
comparison the ranking exists for: which not-yet-known entities look worst.

## Commands

```bash
python3 -m argus demo                    # walkthrough
python3 -m argus generate --out data     # regenerate the corpus (deterministic)
python3 -m argus ontology                # print the loaded ontology
python3 -m argus run --out artifacts     # full pipeline, write graph + findings
python3 -m argus eval                    # score resolution against ground truth
python3 -m argus cases                   # ranked correlated cases
python3 -m argus risk person             # highest-risk entities
python3 -m argus dossier PER-000023      # profile with provenance
python3 -m argus connect PER-000023 ORG-000052
```

### Explaining a decision

Any identity decision can be interrogated:

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

The watchlist date of birth is wrong by a day. Partial agreement keeps most of
its weight, because a transposed digit is far more common than a different
person.

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
python3 -m argus ask "how is Chen R Yarrow connected to Julian Novak?"
```

Runs with no API key. A pattern-based translator handles the common question
shapes offline. Set `ANTHROPIC_API_KEY` and `pip install anthropic` to route
translation through an LLM instead. The SDK is imported lazily, so the engine
and the test suite still run on a machine that's never installed it. Override
the model with `ARGUS_MODEL`.

## Layout

| Module | Does |
|---|---|
| `argus/ontology.py` | Typed schema, per-property match parameters, blocking rules |
| `argus/normalize.py` | Value normalizers for names, orgs, dates, phones, emails, addresses |
| `argus/similarity.py` | Jaro-Winkler, edit distance, token alignment, date proximity |
| `argus/corpus.py` | Synthetic generator with ground truth and a planted network |
| `argus/ingest.py` | One adapter per source system |
| `argus/resolve.py` | Blocking, scoring, clustering, evaluation |
| `argus/graph.py` | Knowledge graph, link rewiring, traversal, path enumeration |
| `argus/analytics.py` | Centrality, Louvain, PageRank, typologies, risk |
| `argus/query.py` | Query language: tokenizer, parser, executor |
| `argus/nl.py` | Pluggable natural language to query translation |
| `argus/pipeline.py` | Orchestration and the analyst-facing surfaces |

```bash
python3 -m unittest discover -s tests     # 112 tests, under a second
```

## Extending

Add an entity type or property to `ontology/intel.json` and it's immediately
normalized, blocked, compared, queryable and describable in natural language.
The NL schema prompt is generated from the live ontology, so nothing else needs
touching.

For a real domain, write one adapter per source in `ingest.py` and supply an
ontology. The resolver, graph, analytics and query layers don't know or care
what domain they're in.
