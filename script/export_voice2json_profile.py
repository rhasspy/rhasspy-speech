#!/usr/bin/env python3
import argparse
import sqlite3
import gzip
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, TextIO, Set


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", help="Directory with voice2json profile")
    parser.add_argument("output_dir", help="Directory to export rhasspy speech model")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dict_path = profile_dir / "base_dictionary.txt.gz"
    assert dict_path.exists()
    g2p_corpus_path = profile_dir / "g2p.corpus.gz"
    assert g2p_corpus_path
    g2p_fst_path = profile_dir / "g2p.fst.gz"
    assert g2p_fst_path.exists()

    # lexicon.db
    with sqlite3.Connection(output_dir / "lexicon.db") as db_conn:
        print("Exporting dictionary")
        with gzip.open(dict_path, "rt", encoding="utf-8") as dict_file:
            export_dictionary(dict_file, db_conn)

        print("Exporting alignments")
        with gzip.open(g2p_corpus_path, "rt", encoding="utf-8") as corpus_file:
            export_alignments(corpus_file, db_conn)

    # g2p.fst
    print("Extracting phonetisaurus model")
    with gzip.open(g2p_fst_path, "rb") as gz_fst_file, open(
        output_dir / "g2p.fst", "wb"
    ) as fst_file:
        shutil.copyfileobj(gz_fst_file, fst_file)

    # model
    print("Copying acoustic model")
    output_model_dir = output_dir / "model"
    output_model_dir.mkdir(parents=True, exist_ok=True)

    for model_sub_dir in (profile_dir / "acoustic_model").iterdir():
        if (not model_sub_dir.is_dir()) or (model_sub_dir.name == "base_graph"):
            continue

        output_model_sub_dir = output_model_dir / model_sub_dir.name
        if output_model_sub_dir.exists():
            shutil.rmtree(output_model_sub_dir)

        # cp src/model/dir dst/model/dir
        shutil.copytree(model_sub_dir, output_model_sub_dir)

    print("Done")


# -----------------------------------------------------------------------------


def export_dictionary(dict_file: TextIO, db_conn: sqlite3.Connection) -> None:
    """Export base dictionary to sqlite3 database."""
    db_conn.execute("DROP TABLE IF EXISTS word_phonemes")
    db_conn.execute(
        "CREATE TABLE word_phonemes (word TEXT, phonemes TEXT, pron_order INTEGER)"
    )
    db_conn.execute("CREATE INDEX idx_word_phonemes ON word_phonemes (word)")

    pron_counts: Dict[str, int] = Counter()
    for line in dict_file:
        line = line.strip()
        if not line:
            continue

        line_parts = line.split(maxsplit=1)
        if len(line_parts) != 2:
            continue

        word, phonemes = line_parts
        if word[0] in ("<", "!"):
            continue

        db_conn.execute(
            "INSERT INTO word_phonemes (word, phonemes, pron_order) VALUES (?, ?, ?)",
            (word, phonemes, pron_counts[word]),
        )
        pron_counts[word] += 1


def export_alignments(corpus_file: TextIO, db_conn: sqlite3.Connection) -> None:
    """Export phonetisaurus alignments to sqlite3 database."""
    db_conn.execute("DROP TABLE IF EXISTS g2p_alignments")
    db_conn.execute("CREATE TABLE g2p_alignments (word TEXT, alignment TEXT)")
    db_conn.execute("CREATE INDEX idx_g2p_alignments ON word_phonemes (word)")

    used_words: Set[str] = set()

    for line in corpus_file:
        alignment = line.strip()
        if not alignment:
            continue

        word = ""
        in_phonemes = False
        for c in alignment:
            if c == "|":
                # Grapheme separator
                continue

            if c == "}":
                in_phonemes = True
            elif c == " ":
                in_phonemes = False
            elif not in_phonemes:
                word += c

        if (not word) or (word in used_words):
            continue

        db_conn.execute(
            "INSERT INTO g2p_alignments (word, alignment) VALUES (?, ?)",
            (word, alignment),
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
