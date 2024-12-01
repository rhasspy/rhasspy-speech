"""Convert sentences to FST."""

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple, Union

from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import extractOne
from unicode_rbnf import RbnfEngine

from .const import EPS, UNK, WordCasing
from .g2p import LexiconDatabase, split_words
from .sentences import generate_sentences

_LOGGER = logging.getLogger(__name__)


@dataclass
class IntentsToFstContext:
    fst_file: TextIO
    lexicon: LexiconDatabase
    number_engine: RbnfEngine
    current_state: Optional[int] = None
    eps: str = EPS
    vocab: Set[str] = field(default_factory=set)
    meta_labels: Set[str] = field(default_factory=set)
    word_casing: WordCasing = WordCasing.LOWER

    def next_state(self) -> int:
        if self.current_state is None:
            self.current_state = 0
        else:
            self.current_state += 1

        return self.current_state

    def next_edge(
        self,
        from_state: int,
        from_label: Optional[str] = None,
        to_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> int:
        to_state = self.next_state()
        self.add_edge(from_state, to_state, from_label, to_label, log_prob)
        return to_state

    def add_edge(
        self,
        from_state: int,
        to_state: int,
        from_label: Optional[str] = None,
        to_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> None:
        if from_label is None:
            from_label = self.eps

        if to_label is None:
            to_label = from_label

        if (" " in from_label) or (" " in to_label):
            raise ValueError(
                f"Cannot have white space in labels: from={from_label}, to={to_label}"
            )

        if (not from_label) or (not to_label):
            raise ValueError(
                f"Labels cannot be empty: from={from_label}, to={to_label}"
            )

        if log_prob is None:
            print(from_state, to_state, from_label, to_label, file=self.fst_file)
        else:
            print(
                from_state, to_state, from_label, to_label, log_prob, file=self.fst_file
            )

    def accept(self, state: int) -> None:
        print(state, file=self.fst_file)


def intents_to_fst(
    train_dir: Union[str, Path],
    sentence_yaml: Dict[str, Any],
    fst_file: TextIO,
    lexicon: LexiconDatabase,
    number_engine: RbnfEngine,
    word_casing: WordCasing = WordCasing.LOWER,
    eps: str = EPS,
    unk: str = UNK,
) -> IntentsToFstContext:
    """Convert YAML sentence files to an FST for Kaldi."""
    os.makedirs(train_dir, exist_ok=True)

    context = IntentsToFstContext(
        fst_file=fst_file, lexicon=lexicon, number_engine=number_engine, eps=eps
    )
    root_state = context.next_state()
    final_state = context.next_state()

    num_sentences = 0
    sentences_db_path = os.path.join(train_dir, "sentences.db")
    used_sentences: Set[str] = set()
    casing_func = WordCasing.get_function(word_casing)
    start_time = time.monotonic()
    with sqlite3.Connection(sentences_db_path) as sentences_db:
        sentences_db.execute("DROP TABLE IF EXISTS sentences")
        sentences_db.execute("CREATE TABLE sentences (input TEXT, output TEXT)")
        sentences_db.execute("CREATE INDEX idx_input ON sentences (input)")

        for input_text, output_text in generate_sentences(sentence_yaml, number_engine):
            input_words = [
                casing_func(w) for w in split_words(input_text, lexicon, number_engine)
            ]
            input_text = " ".join(input_words).strip()
            output_text = output_text.strip()

            if input_text in used_sentences:
                continue

            used_sentences.add(input_text)

            sentences_db.execute(
                "INSERT INTO sentences (input, output) VALUES (?, ?)",
                (input_text, output_text),
            )

            state = root_state
            context.vocab.update(input_words)

            for word in input_words:
                state = context.next_edge(state, word, word)

            context.add_edge(state, final_state)
            num_sentences += 1

    # unk_order = 5
    # unk_log_prob = -3 #-math.log(num_sentences / 1000)
    # for num_unk in range(1, unk_order + 1):
    #     state = root_state
    #     for _ in range(num_unk):
    #         state = context.next_edge(state, unk, log_prob=unk_log_prob)

    #     context.add_edge(state, final_state)

    context.accept(final_state)
    context.fst_file.seek(0)

    _LOGGER.debug(
        "Generated %s sentence(s) in %s second(s)",
        num_sentences,
        time.monotonic() - start_time,
    )

    return context


def get_matching_scores(
    texts: List[str],
    sentences_db_path: Union[str, Path],
    norm_distance_threshold: Optional[float] = None,
    weights: Tuple[int, int, int] = (1, 1, 3),
) -> Optional[Tuple[str, float]]:
    best_text = None
    best_score = None

    with sqlite3.connect(str(sentences_db_path)) as db_conn:
        for text in texts:
            score_cutoff: Optional[float] = 0
            if norm_distance_threshold is not None:
                score_cutoff = int(len(text) * norm_distance_threshold)

            cursor = db_conn.execute("SELECT input, output from sentences")
            result = extractOne(
                [text],
                cursor,
                processor=lambda s: s[0],
                scorer=Levenshtein.distance,
                score_cutoff=score_cutoff,
                scorer_kwargs={"weights": weights},
            )

            if result is None:
                # Didn't make the score cutoff
                continue

            fixed_row, score = result[0], result[1]
            if score == 0:
                # Can't do any better
                return (fixed_row[1], score)

            if (best_score is None) or (score < best_score):
                best_text = fixed_row[1]
                best_score = score

    if (best_text is None) or (best_score is None):
        return None

    return (best_text, best_score)
