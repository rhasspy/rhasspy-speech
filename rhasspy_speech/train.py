import io
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Union, Optional

from hassil.util import merge_dict
from yaml import safe_load

from .kaldi import KaldiTrainer, intents_to_fst


def train_model(
    language: str,
    sentence_files: Iterable[Union[str, Path]],
    kaldi_dir: Union[str, Path],
    model_dir: Union[str, Path],
    train_dir: Union[str, Path],
    phonetisaurus_bin: Union[str, Path],
    # frequent_words_path: Optional[Union[str, Path]] = None,
    # max_unknown_words: int = 50,
    # min_unknown_length: int = 2,
    # max_unknown_length: int = 4,
    # unk_prob: float = 1e-7,
):
    """Train a model on YAML sentences."""
    sentence_yaml: Dict[str, Any] = {}

    for sentence_path in sentence_files:
        with open(sentence_path, "r", encoding="utf-8") as sentence_file:
            merge_dict(sentence_yaml, safe_load(sentence_file))

    with io.StringIO() as fst_file:
        fst_context = intents_to_fst(
            train_dir=train_dir,
            sentence_yaml=sentence_yaml,
            fst_file=fst_file,
            language=language,
            lexicon_db_path=os.path.join(model_dir, "lexicon.db"),
            # frequent_words_path=frequent_words_path,
            # max_unknown_words=max_unknown_words,
            # min_unknown_length=min_unknown_length,
            # max_unknown_length=max_unknown_length,
            # unk_prob=unk_prob,
        )
        trainer = KaldiTrainer(
            kaldi_dir,
            os.path.join(model_dir, "model"),
            os.path.join(model_dir, "lexicon.db"),
            phonetisaurus_bin,
        )
        trainer.train(fst_context, train_dir)
