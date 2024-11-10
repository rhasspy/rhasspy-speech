import argparse
import csv
import asyncio
import math
import asyncio.subprocess
import itertools
import logging
import io
import json
import base64
import os
import subprocess
import sys
import re
import sqlite3
import shutil
import shlex
import tempfile
import collections
from dataclasses import dataclass, field
from collections.abc import Collection, Iterable, AsyncIterable
from functools import partial
from typing import Any, Dict, Optional, List, Set, TextIO, Union
from pathlib import Path

from hassil.intents import Intents, IntentData, TextSlotList, RangeSlotList, SlotList
from hassil.sample import sample_expression
from hassil.expression import (
    Expression,
    TextChunk,
    Sentence,
    Sequence,
    SequenceType,
    ListReference,
    RuleReference,
)
from unicode_rbnf import RbnfEngine

from .sentences import generate_sentences

_LOGGER = logging.getLogger(__name__)

EPS = "<eps>"
UNK = "<unk>"
SPACE = "▁"

META_BEGIN = "__begin_"
META_END = "__end_"
META_VALUE = "__value_"
META_INTENT = "__intent_"


@dataclass
class IntentsToFstContext:
    fst_file: TextIO
    engine: RbnfEngine
    current_state: Optional[int] = None
    eps: str = EPS
    vocab: Set[str] = field(default_factory=set)
    meta_labels: Set[str] = field(default_factory=set)

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


@dataclass
class TrainingContext:
    train_dir: Path
    kaldi_dir: Path
    model_dir: Path
    vocab: Collection[str]
    eps: str = EPS
    unk: str = UNK
    # unk_nonterm: str = "#nonterm:unk"
    # unk_prob: float = 1e-5
    spn_phone: str = "SPN"
    sil_phone: str = "SIL"
    # unknown_length: int = 0
    # unk_vocab: Optional[Collection[str]] = None

    _extended_env: Optional[Dict[str, Any]] = None

    # @property
    # def unknown_words_enabled(self):
    #     return self.unknown_length > 0

    @property
    def egs_utils_dir(self):
        return self.kaldi_dir / "utils"

    @property
    def egs_steps_dir(self):
        return self.kaldi_dir / "steps"

    @property
    def conf_dir(self):
        return self.train_dir / "conf"

    @property
    def data_dir(self):
        return self.train_dir / "data"

    @property
    def data_local_dir(self):
        return self.data_dir / "local"

    @property
    def lang_dir(self):
        return self.data_dir / "lang"

    @property
    def graph_dir(self):
        return self.train_dir / "graph"

    @property
    def extended_env(self):
        if self._extended_env is None:
            kaldi_bin_dir = self.kaldi_dir / "bin"
            self._extended_env = os.environ.copy()
            self._extended_env["PATH"] = (
                str(kaldi_bin_dir)
                + ":"
                + str(self.egs_utils_dir)
                + ":"
                + self._extended_env["PATH"]
            )

        return self._extended_env

    def run(self, *args, **kwargs):
        if "cwd" not in kwargs:
            kwargs["cwd"] = self.train_dir

        if "env" not in kwargs:
            kwargs["env"] = self.extended_env

        return subprocess.check_call(*args, **kwargs)


def intents_to_fst(
    train_dir: Union[str, Path],
    sentence_yaml: Dict[str, Any],
    fst_file: TextIO,
    language: str,
    lexicon_db_path: Union[str, Path],
    # frequent_words_path: Optional[Union[str, Path]] = None,
    # max_unknown_words: int = 50,
    # min_unknown_length: int = 2,
    # max_unknown_length: int = 6,
    # unk_prob: float = 1e-8,
    eps: str = "<eps>",
) -> IntentsToFstContext:
    context = IntentsToFstContext(
        fst_file=fst_file, engine=RbnfEngine.for_language(language), eps=eps
    )
    root_state = context.next_state()
    final_state = context.next_state()

    num_sentences = 0
    # sentences = list(generate_sentences(sentence_yaml, language))
    # if frequent_words_path:
    #     sentence_prob = (1.0 - unk_prob) / len(sentences)
    # else:
    #     sentence_prob = 1.0 / len(sentences)

    # sentence_log_prob = -math.log(sentence_prob)

    # for input_text, output_text in sentences:
    sentences_path = os.path.join(train_dir, "sentences.csv")
    used_sentences: Set[str] = set()
    with open(sentences_path, "w", encoding="utf-8") as sentences_file:
        writer = csv.writer(sentences_file, delimiter="|")
        for sen_idx, (input_text, output_text) in enumerate(
            generate_sentences(sentence_yaml, language)
        ):
            original_input_text = input_text

            # TODO: Expose as setting
            # TODO: Replace numbers, etc.
            input_text = input_text.lower()
            input_text = re.sub(r"[_\-\.]", " ", input_text)
            input_text = re.sub(r"[^a-z ]", "", input_text)

            if input_text in used_sentences:
                continue

            input_words = input_text.split()
            input_text = " ".join(input_words)
            used_sentences.add(input_text)

            writer.writerow((original_input_text, output_text, input_text))

            # state = context.next_edge(root_state, eps, eps, log_prob=sentence_log_prob)
            state = root_state
            is_output_different = output_text != input_text
            # is_output_different = False

            context.vocab.update(input_words)

            output_text = f"__{sen_idx}"
            context.meta_labels.add(output_text)
            for word_idx, word in enumerate(input_words):
                if word_idx == 0:
                    state = context.next_edge(state, word, output_text)
                else:
                    state = context.next_edge(state, word, eps)

            # for word in input_words:
            #     state = context.next_edge(
            #         state, word, word, #eps if is_output_different else word
            #     )

            # if is_output_different:
            #     # Emit output text
            #     output_text = output_text.replace(" ", SPACE)
            #     output_text = f"__{sen_idx}"
            #     context.meta_labels.add(output_text)
            #     state = context.next_edge(state, eps, output_text)

            context.add_edge(state, final_state)
            num_sentences += 1

    # if frequent_words_path:
    # Add a path in the FST to catch out-of-vocabulary (OOV) sentences.
    #
    # This is done by collecting a list of frequently used words in the
    # language, removing words used in input sentences, and then creating a
    # sequence of states like:
    # [start] -> [unknown] -> [unknown] -> ... -> [end]
    #
    # Each [unknown] state accepts all of the "unknown" words from the
    # frequent list. Additionally, each [unknown] state contains an epsilon
    # transition to [end] so the path can be 1..max_unknown_length in
    # length.
    #
    # Finally, the initial [start] transition is given a low probability
    # (unk_prob) to ensure it's only taken if nothing else fits.
    #
    # A balance must be struck between:
    # * unk_prob - probability of entering the OOV/unknown path
    # * max_unknown_words - number of frequent words used at each path step
    # * max_unknown_length - number of steps in the path
    #
    # Increasing max_unknown_words and max_unknown_length will catch more
    # OOV sentences, but slow down decoding.
    #
    # Increasing unk_prob will catch more OOV sentences, but increase the
    # number of false negatives (valid sentences detected as OOV).
    # unknown_words: Set[str] = set()
    # unk_prob = max(0.0, min(1.0 - sys.float_info.epsilon, unk_prob))
    # unk_log_prob = -math.log(unk_prob)

    # # Expecting a text file with a list of frequently used words in the
    # # language, most frequent first.
    # with open(
    #     frequent_words_path, "r", encoding="utf-8"
    # ) as freq_file, sqlite3.connect(str(lexicon_db_path)) as lexicon_db:
    #     for word in freq_file:
    #         word = word.strip()
    #         if word and (word not in context.vocab):
    #             cursor = lexicon_db.execute(
    #                 "SELECT COUNT(*) from word_phonemes WHERE word = ?",
    #                 (word,),
    #             )
    #             count = next(iter(cursor), [0])[0]
    #             if count < 1:
    #                 # Not in lexicon
    #                 continue

    #             unknown_words.add(word)
    #             context.vocab.add(word)
    #             if len(unknown_words) >= max_unknown_words:
    #                 break

    # _LOGGER.debug("Adding OOV path using %s word(s)", len(unknown_words))

    # # [start] -> [unknown] -> [unknown] -> ... -> [end]
    # state = context.next_edge(root_state, log_prob=unk_log_prob)
    # for i in range(max_unknown_length):
    #     unknown_start = context.next_edge(state)
    #     unknown_end = context.next_state()
    #     for word in unknown_words:
    #         context.add_edge(unknown_start, unknown_end, word, UNK)

    #     if i >= min_unknown_length:
    #         context.add_edge(unknown_end, final_state)

    #     state = unknown_end

    context.accept(final_state)
    context.fst_file.seek(0)

    _LOGGER.debug("Generated %s sentence(s)", num_sentences)

    return context


# -----------------------------------------------------------------------------


class KaldiTrainer:
    def __init__(
        self,
        kaldi_dir: Union[str, Path],
        model_dir: Union[str, Path],
        lexicon_db_path: Union[str, Path],
        phonetisaurus_bin: Union[str, Path],
    ):
        self.kaldi_dir = Path(kaldi_dir).absolute()
        self.model_dir = Path(model_dir).absolute()
        self.lexicon_db = sqlite3.connect(lexicon_db_path)
        self.phonetisaurus_bin = Path(phonetisaurus_bin)

    def train(
        self,
        fst_context: IntentsToFstContext,
        train_dir: Union[str, Path],
        eps: str = EPS,
        unk: str = UNK,
        unk_nonterm: str = "#nonterm:unk",
        spn_phone: str = "SPN",
        sil_phone: str = "SIL",
        # unknown_length: int = 0,
        # max_unknown_words: int = 100,
        # possible_unknown_words: Optional[Collection[str]] = None,
    ):
        ctx = TrainingContext(
            train_dir=Path(train_dir).absolute(),
            kaldi_dir=self.kaldi_dir,
            model_dir=self.model_dir,
            vocab=fst_context.vocab,
            eps=eps,
            unk=unk,
            spn_phone=spn_phone,
            sil_phone=sil_phone,
            # unknown_length=unknown_length,
        )

        # if ctx.unknown_words_enabled:
        #     # Exclude vocab
        #     ctx.unk_vocab = [w for w in possible_unknown_words if w not in vocab][
        #         :max_unknown_words
        #     ]
        #     assert ctx.unk_vocab, "No unknown words remain"

        # ---------------------------------------------------------------------

        # Extend PATH
        ctx.train_dir.mkdir(parents=True, exist_ok=True)

        # Copy conf
        if ctx.conf_dir.exists():
            shutil.rmtree(ctx.conf_dir)

        shutil.copytree(self.model_dir / "conf", ctx.conf_dir)

        # Delete existing data/graph
        if ctx.data_dir.exists():
            shutil.rmtree(ctx.data_dir)

        ctx.lang_dir.mkdir(parents=True, exist_ok=True)

        if ctx.graph_dir.exists():
            shutil.rmtree(ctx.graph_dir)

        # ---------------------------------------------------------------------
        # Kaldi Training
        # ---------------------------------------------------------
        # 1. prepare_lang.sh
        # 2. format_lm.sh (or fstcompile)
        # 3. mkgraph.sh
        # 4. prepare_online_decoding.sh
        # ---------------------------------------------------------

        # Create empty path.sh
        path_sh = ctx.train_dir / "path.sh"
        if not path_sh.is_file():
            path_sh.write_text("")

        # Write pronunciation dictionary
        self._create_lexicon(ctx, fst_context.meta_labels)

        # Create utils link
        model_utils_link = ctx.train_dir / "utils"

        try:
            # Can't use missing_ok in 3.6
            model_utils_link.unlink()
        except Exception:
            pass

        model_utils_link.symlink_to(ctx.egs_utils_dir, target_is_directory=True)

        # 1. prepare_lang.sh
        self._prepare_lang(ctx)

        # 2. Generate G.fst from skill graph
        self._create_fst(ctx, fst_context.fst_file)

        # 3. mkgraph.sh
        self._mkgraph(ctx)

        # 4. prepare_online_decoding.sh
        self._prepare_online_decoding(ctx)

    # -------------------------------------------------------------------------

    def _create_lexicon(self, ctx: TrainingContext, meta_labels: Iterable[str]):
        _LOGGER.debug("Generating lexicon")
        dict_local_dir = ctx.data_local_dir / "dict"
        dict_local_dir.mkdir(parents=True, exist_ok=True)

        # Copy phones
        phones_dir = self.model_dir / "phones"
        for phone_file in phones_dir.glob("*.txt"):
            shutil.copy(phone_file, dict_local_dir / phone_file.name)

        # Create dictionary
        dictionary_path = dict_local_dir / "lexicon.txt"
        with open(dictionary_path, "w", encoding="utf-8") as dictionary_file:
            missing_words = set()
            for word in sorted(ctx.vocab):
                cursor = self.lexicon_db.execute(
                    "SELECT phonemes from word_phonemes WHERE word = ? ORDER BY pron_order",
                    (word,),
                )

                word_found = False
                for row in cursor:
                    phonemes_str = row[0]
                    print(word, phonemes_str, file=dictionary_file)
                    word_found = True

                if not word_found:
                    missing_words.add(word)

            if missing_words:
                g2p_model_path = self.model_dir.parent / "g2p.fst"
                missing_words_path = ctx.train_dir / "missing_words_dictionary.txt"
                with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".txt", encoding="utf-8"
                ) as missing_words_file, open(
                    missing_words_path, "w", encoding="utf-8"
                ) as missing_dictionary_file:
                    for word in sorted(missing_words):
                        _LOGGER.warning("Guessing pronunciation for %s", word)
                        print(word, file=missing_words_file)

                    missing_words_file.seek(0)
                    phonetisaurus_output = (
                        subprocess.check_output(
                            [
                                str(self.phonetisaurus_bin),
                                f"--model={g2p_model_path}",
                                f"--wordlist={missing_words_file.name}",
                            ]
                        )
                        .decode()
                        .splitlines()
                    )
                    for line in phonetisaurus_output:
                        line = line.strip()
                        if line:
                            line_parts = line.split()
                            if len(line_parts) < 3:
                                continue

                            word = line_parts[0]
                            phonemes = " ".join(line_parts[2:])

                            print(
                                word,
                                phonemes,
                                file=missing_dictionary_file,
                            )
                            print(word, phonemes, file=dictionary_file)

            if ctx.unk not in ctx.vocab:
                # Add <unk>
                print(ctx.unk, ctx.spn_phone, file=dictionary_file)

            for label in meta_labels:
                print(label, ctx.sil_phone, file=dictionary_file)

    def _prepare_lang(self, ctx: TrainingContext):
        dict_local_dir = ctx.data_local_dir / "dict"
        lang_local_dir = ctx.data_local_dir / "lang"
        prepare_lang = [
            "bash",
            str(ctx.egs_utils_dir / "prepare_lang.sh"),
            str(dict_local_dir),
            ctx.unk,
            str(lang_local_dir),
            str(ctx.lang_dir),
        ]

        _LOGGER.debug(prepare_lang)
        subprocess.check_call(prepare_lang, cwd=ctx.train_dir, env=ctx.extended_env)

    def _create_fst(self, ctx: TrainingContext, fst_file: TextIO):
        fst_path = ctx.lang_dir / "G.fst"
        unsorted_fst_path = fst_path.with_suffix(".fst.unsorted")
        text_fst_path = fst_path.with_suffix(".fst.txt")

        with open(text_fst_path, "w", encoding="utf-8") as text_fst_file:
            shutil.copyfileobj(fst_file, text_fst_file)

        compile_grammar = [
            "fstcompile",
            shlex.quote(f"--isymbols={ctx.lang_dir}/words.txt"),
            shlex.quote(f"--osymbols={ctx.lang_dir}/words.txt"),
            "--keep_isymbols=false",
            "--keep_osymbols=false",
            "--keep_state_numbering=true",
            shlex.quote(str(text_fst_path)),
            shlex.quote(str(unsorted_fst_path)),
        ]

        _LOGGER.debug(compile_grammar)
        ctx.run(compile_grammar)

        # determinize/minimize
        determinized_fst_path = fst_path.with_suffix(".fst.determinized")
        determinize = [
            "fstdeterminize",
            shlex.quote(str(unsorted_fst_path)),
            shlex.quote(str(determinized_fst_path)),
        ]

        _LOGGER.debug(determinize)
        ctx.run(determinize)

        minimized_fst_path = fst_path.with_suffix(".fst.minimized")
        minimize = [
            "fstminimize",
            shlex.quote(str(determinized_fst_path)),
            shlex.quote(str(minimized_fst_path)),
        ]

        _LOGGER.debug(minimize)
        ctx.run(minimize)

        arcsort = [
            "fstarcsort",
            "--sort_type=ilabel",
            shlex.quote(str(minimized_fst_path)),
            shlex.quote(str(fst_path)),
        ]

        _LOGGER.debug(arcsort)
        ctx.run(arcsort)
        unsorted_fst_path.unlink()

    def _mkgraph(self, ctx: TrainingContext):
        mkgraph = [
            "bash",
            str(ctx.egs_utils_dir / "mkgraph.sh"),
            "--self-loop-scale",
            "1.0",
            str(ctx.lang_dir),
            str(ctx.model_dir / "model"),
            str(ctx.graph_dir),
        ]
        _LOGGER.debug(mkgraph)
        ctx.run(mkgraph)

    def _prepare_online_decoding(self, ctx: TrainingContext):
        extractor_dir = ctx.model_dir / "extractor"
        if extractor_dir.is_dir():
            # Generate online.conf
            mfcc_conf = ctx.model_dir / "conf" / "mfcc_hires.conf"
            prepare_online_decoding = [
                "bash",
                str(
                    ctx.egs_steps_dir
                    / "online"
                    / "nnet3"
                    / "prepare_online_decoding.sh"
                ),
                "--mfcc-config",
                str(mfcc_conf),
                str(ctx.lang_dir),
                str(extractor_dir),
                str(ctx.model_dir / "model"),
                str(ctx.model_dir / "online"),
            ]

            _LOGGER.debug(prepare_online_decoding)
            subprocess.run(
                prepare_online_decoding,
                cwd=ctx.train_dir,
                env=ctx.extended_env,
                stderr=subprocess.STDOUT,
                check=True,
            )


# -----------------------------------------------------------------------------


class KaldiTranscriber:
    """Speech to text with external Kaldi scripts."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        graph_dir: Union[str, Path],
        kaldi_bin_dir: Union[str, Path],
        max_active: int = 7000,
        lattice_beam: float = 8.0,
        acoustic_scale: float = 1.0,
        beam: float = 24.0,
    ):
        self.model_dir = Path(model_dir)
        self.graph_dir = Path(graph_dir)
        self.kaldi_bin_dir = Path(kaldi_bin_dir)

        self.decode_proc: Optional[subprocess.Popen] = None
        self.decode_proc_async: Optional[asyncio.subprocess.Process] = None

        # Additional arguments passed to Kaldi process
        self.kaldi_args = [
            f"--max-active={max_active}",
            f"--lattice-beam={lattice_beam}",
            f"--acoustic-scale={acoustic_scale}",
            f"--beam={beam}",
        ]

        self.temp_dir = None
        self.chunk_fifo_path = None
        self.chunk_fifo_file = None

        _LOGGER.debug("Using kaldi at %s", str(self.kaldi_bin_dir))

    def transcribe_wav(self, wav_path: Union[str, Path]) -> Optional[str]:
        """Speech to text from WAV data."""
        text = self._transcribe_wav_nnet3(str(wav_path))

        if text:
            if UNK in text:
                # Unknown words path
                text = ""

            # Success
            return _fix_text(text).strip()

        # Failure
        return None

    def _transcribe_wav_nnet3(self, wav_path: str) -> str:
        words_txt = self.graph_dir / "words.txt"
        online_conf = self.model_dir / "online" / "conf" / "online.conf"
        kaldi_cmd = (
            [
                str(self.kaldi_bin_dir / "online2-wav-nnet3-latgen-faster"),
                "--online=false",
                "--do-endpointing=false",
                f"--word-symbol-table={words_txt}",
                f"--config={online_conf}",
            ]
            + self.kaldi_args
            + [
                str(self.model_dir / "model" / "final.mdl"),
                str(self.graph_dir / "HCLG.fst"),
                "ark:echo utt1 utt1|",
                f"scp:echo utt1 {wav_path}|",
                "ark:/dev/null",
            ]
        )

        _LOGGER.debug(kaldi_cmd)

        try:
            lines = subprocess.check_output(
                kaldi_cmd, stderr=subprocess.STDOUT, universal_newlines=True
            ).splitlines()
        except subprocess.CalledProcessError as e:
            _LOGGER.exception("_transcribe_wav_nnet3")
            _LOGGER.error(e.output)
            lines = []

        text = ""
        for line in lines:
            if line.startswith("utt1 "):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    text = parts[1]
                break

        return _fix_text(text)

    async def transcribe_wav_async(self, wav_path: Union[str, Path]) -> Optional[str]:
        """Speech to text from WAV data."""
        text = await self._transcribe_wav_nnet3_async(str(wav_path))

        if text:
            if UNK in text:
                # Unknown words path
                text = ""

            # Success
            return _fix_text(text).strip()

        # Failure
        return None

    async def _transcribe_wav_nnet3_async(self, wav_path: str) -> str:
        words_txt = self.graph_dir / "words.txt"
        online_conf = self.model_dir / "online" / "conf" / "online.conf"
        kaldi_cmd = (
            [
                str(self.kaldi_bin_dir / "online2-wav-nnet3-latgen-faster"),
                "--online=false",
                "--do-endpointing=false",
                f"--word-symbol-table={words_txt}",
                f"--config={online_conf}",
            ]
            + self.kaldi_args
            + [
                str(self.model_dir / "model" / "final.mdl"),
                str(self.graph_dir / "HCLG.fst"),
                "ark:echo utt1 utt1|",
                f"scp:echo utt1 {wav_path}|",
                "ark:/dev/null",
            ]
        )

        _LOGGER.debug(kaldi_cmd)

        try:
            proc = await asyncio.create_subprocess_exec(
                kaldi_cmd[0],
                *kaldi_cmd[1:],
                stderr=asyncio.subprocess.STDOUT,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _stderr = await proc.communicate()
            lines = stdout.decode().splitlines()
        except subprocess.CalledProcessError as e:
            _LOGGER.exception("_transcribe_wav_nnet3")
            _LOGGER.error(e.output)
            lines = []

        text = ""
        for line in lines:
            if line.startswith("utt1 "):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    text = parts[1]
                break

        return _fix_text(text)

    # -------------------------------------------------------------------------

    def transcribe_stream(
        self,
        audio_stream: Iterable[bytes],
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> Optional[str]:
        """Speech to text from an audio stream."""
        if not self.decode_proc:
            self.start()

        assert self.decode_proc, "No decode process"

        # start_time = time.perf_counter()
        num_frames = 0
        for chunk in audio_stream:
            if chunk:
                num_samples = len(chunk) // sample_width

                # Write sample count to process stdin
                print(num_samples, file=self.decode_proc.stdin)
                self.decode_proc.stdin.flush()

                # Write chunk to FIFO.
                # Make sure that we write exactly the right number of bytes.
                self.chunk_fifo_file.write(chunk[: num_samples * sample_width])
                self.chunk_fifo_file.flush()
                num_frames += num_samples

        # Finish utterance
        print("0", file=self.decode_proc.stdin)
        self.decode_proc.stdin.flush()

        _LOGGER.debug("Finished stream. Getting transcription.")

        for line in self.decode_proc.stdout:
            line = line.strip()
            if line.lower() == "ready":
                continue

            confidence_and_text = line
            break

        _LOGGER.debug(confidence_and_text)

        if confidence_and_text:
            # Success
            # end_time = time.perf_counter()

            # <mbr_wer> <word> <word_confidence> <word_start_time> <word_end_time> ...
            _wer_str, *words = confidence_and_text.split()
            # confidence = 0.0

            # try:
            #     # Try to parse minimum bayes risk (MBR) word error rate (WER)
            #     confidence = max(0, 1 - float(wer_str))
            # except ValueError:
            #     _LOGGER.exception(wer_str)

            tokens = []
            for word, _word_confidence, _word_start_time, _word_end_time in grouper(
                words, n=4
            ):
                if word == UNK:
                    # Unknown words path
                    return ""

                tokens.append(word)

            return _fix_text(" ".join(t for t in tokens))

        # Failure
        return None

    async def transcribe_stream_async(
        self,
        audio_stream: AsyncIterable[bytes],
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> Optional[str]:
        """Speech to text from an audio stream."""
        if not self.decode_proc_async:
            await self.start_async()

        assert self.decode_proc_async, "No decode process"

        # start_time = time.perf_counter()
        num_frames = 0
        async for chunk in audio_stream:
            if chunk:
                num_samples = len(chunk) // sample_width

                # Write sample count to process stdin
                self.decode_proc_async.stdin.write(f"{num_samples}\n".encode())
                await self.decode_proc_async.stdin.drain()

                # Write chunk to FIFO.
                # Make sure that we write exactly the right number of bytes.
                self.chunk_fifo_file.write(chunk[: num_samples * sample_width])
                self.chunk_fifo_file.flush()
                num_frames += num_samples

        # Finish utterance
        self.decode_proc_async.stdin.write("0\n".encode())
        await self.decode_proc_async.stdin.drain()

        _LOGGER.debug("Finished stream. Getting transcription.")

        async def next_line():
            return (await self.decode_proc_async.stdout.readline()).decode()

        line = await next_line()
        while line:
            line = line.strip()
            if line.lower() == "ready":
                line = await next_line()
                continue

            confidence_and_text = line
            break

        _LOGGER.debug(confidence_and_text)

        if confidence_and_text:
            # Success
            # end_time = time.perf_counter()

            # <mbr_wer> <word> <word_confidence> <word_start_time> <word_end_time> ...
            _wer_str, *words = confidence_and_text.split()
            # confidence = 0.0

            # try:
            #     # Try to parse minimum bayes risk (MBR) word error rate (WER)
            #     confidence = max(0, 1 - float(wer_str))
            # except ValueError:
            #     _LOGGER.exception(wer_str)

            tokens = []
            for word, _word_confidence, _word_start_time, _word_end_time in grouper(
                words, n=4
            ):
                if word == UNK:
                    # Unknown words path
                    return ""

                tokens.append(word)

            return _fix_text(" ".join(t for t in tokens))

        # Failure
        return None

    def stop(self):
        """Stop the transcriber."""
        if self.decode_proc:
            self.decode_proc.terminate()
            self.decode_proc.wait()
            self.decode_proc = None

        if self.decode_proc_async:
            self.decode_proc_async.terminate()
            self.decode_proc_async = None

        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None

        if self.chunk_fifo_file:
            self.chunk_fifo_file.close()
            self.chunk_fifo_file = None

        self.chunk_fifo_path = None

    def start(self):
        """Starts online2-tcp-nnet3-decode-faster process."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory()

        if self.chunk_fifo_path is None:
            self.chunk_fifo_path = os.path.join(self.temp_dir.name, "chunks.fifo")
            _LOGGER.debug("Creating FIFO at %s", self.chunk_fifo_path)
            os.mkfifo(self.chunk_fifo_path)

        online_conf = self.model_dir / "online" / "conf" / "online.conf"

        kaldi_cmd = (
            [
                str(self.kaldi_bin_dir / "online2-cli-nnet3-decode-faster-confidence"),
                f"--config={online_conf}",
            ]
            + self.kaldi_args
            + [
                str(self.model_dir / "model" / "final.mdl"),
                str(self.graph_dir / "HCLG.fst"),
                str(self.graph_dir / "words.txt"),
                str(self.chunk_fifo_path),
            ]
        )

        _LOGGER.debug(kaldi_cmd)

        self.decode_proc = subprocess.Popen(
            kaldi_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        # NOTE: The placement of this open is absolutely critical
        #
        # At this point, the decode process will block waiting for the other
        # side of the pipe.
        #
        # We won't reach the "ready" stage if we open this earlier or later.
        if self.chunk_fifo_file is None:
            self.chunk_fifo_file = open(self.chunk_fifo_path, mode="wb")

        # Read until started
        line = self.decode_proc.stdout.readline().lower().strip()
        if line:
            _LOGGER.debug(line)

        while "ready" not in line:
            line = self.decode_proc.stdout.readline().lower().strip()
            if line:
                _LOGGER.debug(line)

        _LOGGER.debug("Decoder started")

    async def start_async(self):
        """Starts online2-tcp-nnet3-decode-faster process."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory()

        if self.chunk_fifo_path is None:
            self.chunk_fifo_path = os.path.join(self.temp_dir.name, "chunks.fifo")
            _LOGGER.debug("Creating FIFO at %s", self.chunk_fifo_path)
            os.mkfifo(self.chunk_fifo_path)

        online_conf = self.model_dir / "online" / "conf" / "online.conf"

        kaldi_cmd = (
            [
                str(self.kaldi_bin_dir / "online2-cli-nnet3-decode-faster-confidence"),
                f"--config={online_conf}",
            ]
            + self.kaldi_args
            + [
                str(self.model_dir / "model" / "final.mdl"),
                str(self.graph_dir / "HCLG.fst"),
                str(self.graph_dir / "words.txt"),
                str(self.chunk_fifo_path),
            ]
        )

        _LOGGER.debug(kaldi_cmd)

        self.decode_proc_async = await asyncio.create_subprocess_exec(
            kaldi_cmd[0],
            *kaldi_cmd[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )

        # NOTE: The placement of this open is absolutely critical
        #
        # At this point, the decode process will block waiting for the other
        # side of the pipe.
        #
        # We won't reach the "ready" stage if we open this earlier or later.
        if self.chunk_fifo_file is None:
            self.chunk_fifo_file = open(self.chunk_fifo_path, mode="wb")

        # Read until started
        async def next_line():
            return (
                (await self.decode_proc_async.stdout.readline())
                .decode()
                .lower()
                .strip()
            )

        line = await next_line()
        if line:
            _LOGGER.debug(line)

        while "ready" not in line:
            line = await next_line()
            if line:
                _LOGGER.debug(line)

        _LOGGER.debug("Decoder started")


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def _fix_text(text: str) -> str:
    return text.replace(SPACE, " ")
