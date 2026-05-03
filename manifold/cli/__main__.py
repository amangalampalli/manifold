"""Allow `python -m manifold.cli` execution."""

from manifold.cli.main import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
