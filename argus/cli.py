"""Command line interface.

    python -m argus demo                     # full walkthrough, start to finish
    python -m argus generate --out data
    python -m argus run --data data
    python -m argus eval
    python -m argus query 'find person where risk > 0.8 limit 10'
    python -m argus ask "who is connected to the highest risk person?"
    python -m argus dossier PER-000023
    python -m argus connect PER-000023 ORG-000052
    python -m argus explain natreg:NR00001 bank:BK00042H
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import corpus
from .analytics import Finding
from .ontology import Ontology, load_default
from .pipeline import Investigation, Pipeline, save_artifacts
from .query import GRAMMAR_HELP, QueryError, execute, parse
from .resolve import Resolver, find_errors

BANNER = r"""
   _____ ____   ______ __  __ _____
  /  _  \  _ \ / ___  |  |/  / ___/     ontology-driven intelligence platform
 /  /_\  | |_) | |  _ |     \\__ \      resolve -> connect -> find
/  |  |  |  _ <| |_| ||  |\  |__) |
\__|  |__|_| \_\\____/|__| \_____/
"""


def load_ontology(path: str | None) -> Ontology:
    return Ontology.load(path) if path else load_default()


def run_pipeline(args, verbose: bool = True):
    ontology = load_ontology(getattr(args, "ontology", None))
    data_dir = Path(getattr(args, "data", "data"))
    if not data_dir.exists():
        print(f"No corpus at '{data_dir}'. Generating one first...\n")
        corpus.generate(data_dir)
    return Pipeline(ontology, verbose=verbose).run(data_dir)


def context_for(result) -> dict:
    return {"risk": result.risk, "communities": result.communities}


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_generate(args):
    meta = corpus.generate(args.out, seed=args.seed, n_people=args.people,
                           n_orgs=args.orgs, noise=args.noise)
    print(f"Corpus written to {args.out}/")
    print(f"  {meta['people']} people, {meta['organizations']} organizations, "
          f"{meta['accounts']} accounts")
    print(f"  {meta['transactions']:,} transactions, {meta['communications']:,} communications")
    for name, count in meta["source_records"].items():
        print(f"  {name:<10} {count:>7,} records")
    print(f"\n  ground truth -> {args.out}/truth.json  ({len(meta['ring_people'])} "
          f"planted principals, {len(meta['ring_orgs'])} shell companies)")
    return 0


def cmd_ontology(args):
    print(load_ontology(args.ontology).summary())
    return 0


def cmd_run(args):
    print(BANNER)
    result = run_pipeline(args)
    if args.out:
        save_artifacts(result, args.out)
        print(f"\n  artifacts written to {args.out}/")
    investigation = Investigation(result)
    print("\n" + investigation.case_report(limit=args.cases))
    return 0


def cmd_eval(args):
    ontology = load_ontology(args.ontology)
    result = run_pipeline(args)
    if not result.evaluation:
        print("No ground truth found. Generate a corpus first.")
        return 1

    print("\nRESOLUTION QUALITY (pairwise, against ground truth)")
    print(f"{'type':<14} {'records':>8} {'true':>7} {'found':>7} "
          f"{'prec':>7} {'recall':>7} {'f1':>7} {'exact':>7}")
    print("-" * 74)
    for name, stats in result.evaluation.items():
        if name == "overall":
            continue
        print(f"{name:<14} {stats['labelled_records']:>8,} {stats['true_entities']:>7,} "
              f"{stats['predicted_entities']:>7,} {stats['precision']:>7.4f} "
              f"{stats['recall']:>7.4f} {stats['f1']:>7.4f} "
              f"{stats['exact_cluster_rate']:>7.2%}")
    overall = result.evaluation["overall"]
    print("-" * 74)
    print(f"{'OVERALL':<14} {overall['labelled_records']:>8,} {'':>7} {'':>7} "
          f"{overall['precision']:>7.4f} {overall['recall']:>7.4f} {overall['f1']:>7.4f}")

    errors = find_errors(result.ingestor.by_type(), result.resolution, limit=args.examples)
    print(f"\nFALSE MERGES ({len(errors['false_merges'])} shown)")
    if not errors["false_merges"]:
        print("  none. No two distinct real-world objects were fused.")
    for merge in errors["false_merges"]:
        print(f"  {merge['entity']} [{merge['type']}] fuses {len(merge['truths'])} objects")
        for record in merge["records"][:5]:
            print(f"      {record['key']:<22} {record['truth']:<10} {record['name']}")

    print(f"\nFRAGMENTED ENTITIES ({len(errors['fragmented'])} shown)")
    for miss in errors["fragmented"]:
        print(f"  {miss['truth']} [{miss['type']}] split across {miss['fragments']} entities")
        for record in miss["records"][:5]:
            print(f"      {record['key']:<22} {record['entity']:<14} {record['name']}")
    return 0


def cmd_query(args):
    result = run_pipeline(args, verbose=not args.quiet)
    context = context_for(result)
    try:
        parsed = parse(args.query)
        output = parsed.run(result.graph, context)
    except QueryError as error:
        print(f"query error: {error}", file=sys.stderr)
        return 2
    print()
    _render(result, output)
    return 0


def cmd_ask(args):
    from .nl import NaturalLanguage

    result = run_pipeline(args, verbose=not args.quiet)
    interface = NaturalLanguage(result.graph)
    backend = (interface.provider.name if interface.using_model
               else "rule-based translator (no API key configured)")
    print(f"\ntranslating with {backend}")
    try:
        translation = interface.translate(args.question)
    except QueryError as error:
        print(f"\n{error}", file=sys.stderr)
        return 2
    print(f"  question  {args.question}")
    print(f"  query     {translation.query}")
    if translation.reasoning:
        print(f"  why       {translation.reasoning}")
    print()
    _render(result, parse(translation.query).run(result.graph, context_for(result)))
    return 0


def cmd_dossier(args):
    result = run_pipeline(args, verbose=not args.quiet)
    print()
    print(Investigation(result).dossier(args.reference))
    return 0


def cmd_connect(args):
    result = run_pipeline(args, verbose=not args.quiet)
    print()
    print(Investigation(result).connect(args.left, args.right,
                                        max_hops=args.hops, limit=args.limit))
    return 0


def cmd_cases(args):
    result = run_pipeline(args, verbose=not args.quiet)
    print()
    print(Investigation(result).case_report(limit=args.limit))
    return 0


def cmd_risk(args):
    result = run_pipeline(args, verbose=not args.quiet)
    print()
    print(Investigation(result).top_risk(args.type, limit=args.limit))
    return 0


def cmd_explain(args):
    """Show the evidence behind a single match decision."""
    ontology = load_ontology(args.ontology)
    from .ingest import Ingestor

    ingestor = Ingestor(ontology)
    ingestor.ingest_dir(args.data)
    left = ingestor.records.get(args.left)
    right = ingestor.records.get(args.right)
    if left is None or right is None:
        missing = args.left if left is None else args.right
        print(f"no such record: {missing}", file=sys.stderr)
        return 2
    if left.entity_type != right.entity_type:
        print(f"cannot compare a {left.entity_type} with a {right.entity_type}", file=sys.stderr)
        return 2

    decision = Resolver(ontology).score(left, right)
    entity_type = ontology.entity(left.entity_type)
    print()
    print(f"  {left.key}   {json.dumps(left.raw, default=str)[:120]}")
    print(f"  {right.key}   {json.dumps(right.raw, default=str)[:120]}")
    print()
    print(decision.explain())
    print(f"\n    match threshold  {entity_type.match_threshold:+.2f}"
          f"   review threshold {entity_type.review_threshold:+.2f}")
    return 0


def cmd_demo(args):
    print(BANNER)
    data_dir = Path(args.data)
    if not (data_dir / "truth.json").exists():
        print("=== 1. GENERATE a multi-source corpus with planted ground truth ===\n")
        corpus.generate(data_dir)
    print("=== 1. RESOLVE, BUILD, ANALYZE ===\n")
    result = run_pipeline(args)
    investigation = Investigation(result)

    if result.evaluation:
        overall = result.evaluation["overall"]
        print("\n=== 2. HOW GOOD IS THE RESOLUTION? (scored against ground truth) ===")
        print(f"\n  pairwise precision {overall['precision']:.4f}   "
              f"recall {overall['recall']:.4f}   f1 {overall['f1']:.4f}")
        for name, stats in result.evaluation.items():
            if name != "overall":
                print(f"    {name:<14} {stats['true_entities']:>4} true -> "
                      f"{stats['predicted_entities']:>4} found   f1 {stats['f1']:.4f}")

    print("\n=== 3. WHAT DID IT FIND? ===\n")
    print(investigation.case_report(limit=1))

    print("\n=== 4. WHO IS MOST EXPOSED? ===\n")
    print(investigation.top_risk("person", limit=8))

    ranked = sorted(
        (e for e in result.graph.entities.values() if e.entity_type == "person"),
        key=lambda e: -result.risk.get(e.id, 0.0),
    )
    if ranked:
        print(f"\n=== 5. DOSSIER ON THE TOP-RANKED SUBJECT ===\n")
        print(investigation.dossier(ranked[0].id))
    if len(ranked) > 1:
        print(f"\n=== 6. HOW ARE THE TOP TWO SUBJECTS CONNECTED? ===\n")
        print(investigation.connect(ranked[0].id, ranked[1].id, max_hops=4, limit=2))
    return 0


# --------------------------------------------------------------------------

def _render(result, output):
    graph = result.graph
    if isinstance(output, dict) and "paths" in output:
        source, target, paths = output["source"], output["target"], output["paths"]
        if not paths:
            print(f"no connection found between {source.label()} and {target.label()}")
            return
        print(f"{len(paths)} connection(s): {source.label()} -> {target.label()}")
        for index, path in enumerate(paths, start=1):
            print(f"\n  path {index} ({len(path)} hops)")
            for line in graph.format_path(path, source.id).splitlines():
                print("    " + line)
        return

    if hasattr(output, "entity_type"):
        print(graph.describe(output.id))
        return

    if not output:
        print("no matches")
        return

    print(f"{len(output)} result(s)\n")
    print(f"{'id':<15} {'type':<13} {'risk':>6} {'deg':>4}  label")
    print("-" * 78)
    for entity in output:
        print(f"{entity.id:<15} {entity.entity_type:<13} "
              f"{result.risk.get(entity.id, 0.0):>6.3f} "
              f"{graph.degree(entity.id):>4}  {entity.label()[:34]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Ontology-driven entity resolution, link analysis, and investigation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=GRAMMAR_HELP,
    )
    parser.add_argument("--ontology", help="path to an ontology JSON document")
    parser.add_argument("--data", default="data", help="corpus directory (default: data)")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress pipeline progress")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a synthetic multi-source corpus")
    generate.add_argument("--out", default="data")
    generate.add_argument("--seed", type=int, default=20260813)
    generate.add_argument("--people", type=int, default=300)
    generate.add_argument("--orgs", type=int, default=80)
    generate.add_argument("--noise", type=float, default=0.28)
    generate.set_defaults(func=cmd_generate)

    subparsers.add_parser("ontology", help="print the loaded ontology").set_defaults(func=cmd_ontology)

    run = subparsers.add_parser("run", help="run the full pipeline")
    run.add_argument("--out", help="directory to write artifacts to")
    run.add_argument("--cases", type=int, default=2)
    run.set_defaults(func=cmd_run)

    evaluate = subparsers.add_parser("eval", help="score resolution against ground truth")
    evaluate.add_argument("--examples", type=int, default=3)
    evaluate.set_defaults(func=cmd_eval)

    query = subparsers.add_parser("query", help="run a structured query")
    query.add_argument("query")
    query.set_defaults(func=cmd_query)

    ask = subparsers.add_parser("ask", help="ask a question in natural language")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)

    dossier = subparsers.add_parser("dossier", help="full profile of one entity")
    dossier.add_argument("reference")
    dossier.set_defaults(func=cmd_dossier)

    connect = subparsers.add_parser("connect", help="find paths between two entities")
    connect.add_argument("left")
    connect.add_argument("right")
    connect.add_argument("--hops", type=int, default=4)
    connect.add_argument("--limit", type=int, default=5)
    connect.set_defaults(func=cmd_connect)

    cases = subparsers.add_parser("cases", help="ranked correlated cases")
    cases.add_argument("--limit", type=int, default=3)
    cases.set_defaults(func=cmd_cases)

    risk = subparsers.add_parser("risk", help="highest-risk entities of a type")
    risk.add_argument("type", nargs="?", default="person")
    risk.add_argument("--limit", type=int, default=15)
    risk.set_defaults(func=cmd_risk)

    explain = subparsers.add_parser("explain", help="explain one match decision")
    explain.add_argument("left")
    explain.add_argument("right")
    explain.set_defaults(func=cmd_explain)

    subparsers.add_parser("demo", help="end-to-end walkthrough").set_defaults(func=cmd_demo)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except QueryError as error:
        print(f"query error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
