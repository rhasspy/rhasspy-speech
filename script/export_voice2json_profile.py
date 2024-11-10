#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", help="Directory with voice2json profile")
    parser.add_argument("output_dir", help="Directory to export rhasspy speech model")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------

def export_lexicon_db(output_dir: Path) -> None:
    """Export base dictionary to sqlite3 database."""

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
