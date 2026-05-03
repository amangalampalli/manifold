"""Command line entrypoint for manifold experiments."""

from manifold.cli.commands import run_main as main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
