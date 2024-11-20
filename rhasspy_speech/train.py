"""Methods to train a custom Kaldi model."""

import io
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Optional, Union

from hassil.util import merge_dict
from unicode_rbnf import RbnfEngine
from yaml import safe_load

from .g2p import LexiconDatabase, get_sounds_like
from .kaldi import KaldiTrainer, WordCasing, intents_to_fst


def train_model(
    language: str,
    sentence_files: Iterable[Union[str, Path]],
    kaldi_dir: Union[str, Path],
    model_dir: Union[str, Path],
    train_dir: Union[str, Path],
    phonetisaurus_bin: Union[str, Path],
    openfst_dir: Optional[Union[str, Path]] = None,
    opengrm_dir: Optional[Union[str, Path]] = None,
):
    """Train a model on YAML sentences."""
    model_config: Dict[str, Any] = {}
    model_config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as model_config_file:
            model_config = json.load(model_config_file)

    word_casing = WordCasing(model_config.get("word_casing", "lower"))
    sentence_yaml: Dict[str, Any] = {}

    for sentence_path in sentence_files:
        with open(sentence_path, "r", encoding="utf-8") as sentence_file:
            merge_dict(sentence_yaml, safe_load(sentence_file))

    lexicon = LexiconDatabase(os.path.join(model_dir, "lexicon.db"))
    number_engine = RbnfEngine.for_language(language)

    # User lexicon
    words = sentence_yaml.get("words", {})
    for word, word_prons in words.items():
        if isinstance(word_prons, str):
            word_prons = [word_prons]

        for word_pron in word_prons:
            lexicon.add(word, get_sounds_like(word_pron.split(), lexicon))

    with io.StringIO() as fst_file:
        fst_context = intents_to_fst(
            train_dir=train_dir,
            sentence_yaml=sentence_yaml,
            fst_file=fst_file,
            lexicon=lexicon,
            number_engine=number_engine,
            word_casing=word_casing,
        )
        trainer = KaldiTrainer(
            kaldi_dir=kaldi_dir,
            model_dir=os.path.join(model_dir, "model"),
            phonetisaurus_bin=phonetisaurus_bin,
            opengrm_dir=opengrm_dir,
            openfst_dir=openfst_dir,
        )

        train_kwargs = {}
        if "spn_phone" in model_config:
            train_kwargs["spn_phone"] = model_config["spn_phone"]

        trainer.train(fst_context, train_dir, **train_kwargs)
