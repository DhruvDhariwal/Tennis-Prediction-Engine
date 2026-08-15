"""Command line interface: ``python -m tennis_engine <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ALL_SURFACES, Config, ModelConfig, Paths, SplitConfig, display_path


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity < 0 else (
        logging.INFO if verbosity == 0 else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _config_from_args(args: argparse.Namespace) -> Config:
    paths = Paths(
        base_dir=Path(args.data_dir).resolve(),
        artifacts_dir=Path(args.artifacts_dir).resolve(),
    )
    split = SplitConfig()
    if getattr(args, "train_start", None):
        split = SplitConfig(
            warmup_years=tuple(range(1968, args.train_start)),
            train_years=tuple(range(args.train_start, args.val_year)),
            val_years=(args.val_year,),
            test_years=(args.test_year,),
        )
    model = ModelConfig()
    if getattr(args, "rounds", None):
        model = ModelConfig(num_boost_round=args.rounds)
    return Config(
        paths=paths,
        split=split,
        model=model,
        include_futures=not getattr(args, "no_futures", False),
        include_challengers=not getattr(args, "no_challengers", False),
    )


def cmd_train(args: argparse.Namespace) -> int:
    from . import pipeline

    cfg = _config_from_args(args)
    metrics = pipeline.run(cfg, n_trials=args.tune)

    test = metrics["test"]
    base = metrics["test_baselines"]["elo_surface_blended"]
    print("\n" + "=" * 62)
    print(f"  TEST SEASON {metrics['split']['test_years']}  "
          f"({metrics['dataset']['test_matches']:,} matches)")
    print("=" * 62)
    print(f"  Accuracy   {test['accuracy']:.4f}   (Elo baseline {base['accuracy']:.4f})")
    print(f"  ROC AUC    {test['auc']:.4f}   (Elo baseline {base['auc']:.4f})")
    print(f"  Log loss   {test['log_loss']:.4f}   (Elo baseline {base['log_loss']:.4f})")
    print(f"  Brier      {test['brier']:.4f}   (Elo baseline {base['brier']:.4f})")
    print(f"  ECE        {test['ece']:.4f}")
    if "atp_main_tour" in metrics["test_slices"]:
        tour = metrics["test_slices"]["atp_main_tour"]
        print(f"\n  ATP main tour only: acc {tour['accuracy']:.4f} | "
              f"auc {tour['auc']:.4f} | n={tour['n']:,}")
    print(f"\n  Order-invariance max error: "
          f"{metrics['symmetry']['max_abs_error']:.2e}")
    print(f"  Metrics written to {display_path(cfg.paths.metrics_path)}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import PlayerNotFound, PredictionEngine

    cfg = _config_from_args(args)
    engine = PredictionEngine.load(cfg)
    try:
        result = engine.predict_match(
            args.player1, args.player2,
            surface=args.surface, match_date=args.date,
            best_of=args.best_of,
        )
    except PlayerNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result)
        if args.explain:
            print("\n  Pre-match feature diffs (player1 - player2):")
            for name, value in result.features.items():
                if abs(value) > 1e-9 or value != value:
                    print(f"    {name:<26s} {value:>10.4f}")
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    from .predict import PlayerNotFound, PredictionEngine

    engine = PredictionEngine.load(_config_from_args(args))
    try:
        card = engine.player_card(args.player, surface=args.surface)
    except PlayerNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    width = max(len(k) for k in card)
    for key, value in card.items():
        print(f"  {key:<{width}s}  {value}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from .predict import PredictionEngine

    engine = PredictionEngine.load(_config_from_args(args))
    frame = engine.elo_leaderboard(
        top_n=args.top, surface=args.surface, min_matches=args.min_matches
    )
    if frame.empty:
        print("No players matched the filters.")
        return 1
    print(frame.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tennis_engine",
        description="Leak-proof ATP match prediction engine.",
    )
    parser.add_argument("--data-dir", default=".", help="directory holding the ATP CSVs")
    parser.add_argument("--artifacts-dir", default="artifacts", help="model/state output dir")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="build features, train, evaluate and save")
    train.add_argument("--train-start", type=int, default=None,
                       help="first training season (earlier seasons warm up ratings only)")
    train.add_argument("--val-year", type=int, default=2023)
    train.add_argument("--test-year", type=int, default=2024)
    train.add_argument("--rounds", type=int, default=None, help="max boosting rounds")
    train.add_argument("--tune", type=int, default=0, metavar="N",
                       help="run N random-search trials scored on the validation season")
    train.add_argument("--no-futures", action="store_true")
    train.add_argument("--no-challengers", action="store_true")
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="score a match-up")
    predict.add_argument("player1")
    predict.add_argument("player2")
    predict.add_argument("--surface", default="Hard", choices=list(ALL_SURFACES))
    predict.add_argument("--date", default="today")
    predict.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    predict.add_argument("--explain", action="store_true", help="show feature diffs")
    predict.add_argument("--json", action="store_true")
    predict.set_defaults(func=cmd_predict)

    card = sub.add_parser("card", help="show a player's current ratings and form")
    card.add_argument("player")
    card.add_argument("--surface", default="Hard", choices=list(ALL_SURFACES))
    card.set_defaults(func=cmd_card)

    board = sub.add_parser("leaderboard", help="top players by Elo")
    board.add_argument("--top", type=int, default=20)
    board.add_argument("--surface", default=None, choices=list(ALL_SURFACES))
    board.add_argument("--min-matches", type=int, default=20)
    board.set_defaults(func=cmd_leaderboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(-1 if args.quiet else args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
