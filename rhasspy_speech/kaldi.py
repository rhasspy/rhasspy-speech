import asyncio
import asyncio.subprocess
import gzip
import itertools
import logging
import os
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import AsyncIterable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Union

from unicode_rbnf import RbnfEngine

from .g2p import LexiconDatabase, split_words
from .sentences import generate_sentences

_LOGGER = logging.getLogger(__name__)

EPS = "<eps>"
UNK = "<unk>"
SPACE = "▁"

META_BEGIN = "__begin_"
META_END = "__end_"
META_VALUE = "__value_"
META_INTENT = "__intent_"

ARPA = "arpa"
ARPA_RESCORE = "arpa_rescore"
GRAMMAR = "grammar"


class ModelType(str, Enum):
    NNET3 = "nnet3"
    GMM = "gmm"


class WordCasing(str, Enum):
    KEEP = "keep"
    LOWER = "lower"
    UPPER = "upper"

    @staticmethod
    def get_function(casing: "WordCasing") -> Callable[[str], str]:
        if casing == WordCasing.LOWER:
            return str.lower

        if casing == WordCasing.UPPER:
            return str.upper

        return lambda s: s


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


@dataclass
class TrainingContext:
    train_dir: Path
    kaldi_dir: Path
    model_dir: Path
    vocab: Collection[str]
    fst_context: IntentsToFstContext
    opengrm_dir: Optional[Path] = None
    openfst_dir: Optional[Path] = None
    eps: str = EPS
    unk: str = UNK
    spn_phone: str = "SPN"
    sil_phone: str = "SIL"

    _extended_env: Optional[Dict[str, Any]] = None

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

    def lang_dir(self, suffix: str):
        return self.data_dir / f"lang_{suffix}"

    @property
    def dict_local_dir(self):
        return self.data_local_dir / "dict"

    def lang_local_dir(self, suffix: str):
        return self.data_local_dir / f"lang_{suffix}"

    def graph_dir(self, suffix: str):
        return self.train_dir / f"graph_{suffix}"

    @property
    def extended_env(self) -> Dict[str, str]:
        if self._extended_env is None:
            self._extended_env = os.environ.copy()
            bin_dirs: List[str] = [str(self.kaldi_dir / "bin"), str(self.egs_utils_dir)]
            lib_dirs: List[str] = [str(self.kaldi_dir / "lib")]

            if self.opengrm_dir:
                bin_dirs.append(str(self.opengrm_dir / "bin"))
                lib_dirs.append(str(self.opengrm_dir / "lib"))

            if self.openfst_dir:
                bin_dirs.append(str(self.openfst_dir / "bin"))
                lib_dirs.append(str(self.openfst_dir / "lib"))

            current_path = self._extended_env.get("PATH")
            if current_path:
                bin_dirs.append(current_path)

            current_lib_path = self._extended_env.get("LD_LIBRARY_PATH")
            if current_lib_path:
                lib_dirs.append(current_lib_path)

            self._extended_env["PATH"] = os.pathsep.join(bin_dirs)
            self._extended_env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs)

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
    lexicon: LexiconDatabase,
    number_engine: RbnfEngine,
    word_casing: WordCasing = WordCasing.LOWER,
    eps: str = "<eps>",
) -> IntentsToFstContext:
    context = IntentsToFstContext(
        fst_file=fst_file, lexicon=lexicon, number_engine=number_engine, eps=eps
    )
    root_state = context.next_state()
    final_state = context.next_state()

    num_sentences = 0
    sentences_db_path = os.path.join(train_dir, "sentences.db")
    used_sentences: Set[str] = set()
    casing_func = WordCasing.get_function(word_casing)
    with sqlite3.Connection(sentences_db_path) as sentences_db:
        sentences_db.execute("DROP TABLE IF EXISTS sentences")
        sentences_db.execute("CREATE TABLE sentences (input TEXT, output TEXT)")
        sentences_db.execute("CREATE INDEX idx_input ON sentences (input)")

        for input_text, output_text in generate_sentences(sentence_yaml, number_engine):
            input_words = [
                casing_func(w) for w in split_words(input_text, lexicon, number_engine)
            ]
            input_text = " ".join(input_words)

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
        phonetisaurus_bin: Union[str, Path],
        opengrm_dir: Optional[Union[str, Path]] = None,
        openfst_dir: Optional[Union[str, Path]] = None,
    ):
        self.kaldi_dir = Path(kaldi_dir).absolute()
        self.model_dir = Path(model_dir).absolute()
        self.phonetisaurus_bin = Path(phonetisaurus_bin)

        self.opengrm_dir: Optional[Path] = None
        if opengrm_dir:
            self.opengrm_dir = Path(opengrm_dir).absolute()

        self.openfst_dir: Optional[Path] = None
        if openfst_dir:
            self.openfst_dir = Path(openfst_dir).absolute()

    def train(
        self,
        fst_context: IntentsToFstContext,
        train_dir: Union[str, Path],
        eps: str = EPS,
        unk: str = UNK,
        spn_phone: str = "SPN",
        sil_phone: str = "SIL",
        rescore_order: Optional[int] = None,
    ):
        ctx = TrainingContext(
            train_dir=Path(train_dir).absolute(),
            kaldi_dir=self.kaldi_dir,
            model_dir=self.model_dir,
            opengrm_dir=self.opengrm_dir,
            openfst_dir=self.openfst_dir,
            vocab=fst_context.vocab,
            fst_context=fst_context,
            eps=eps,
            unk=unk,
            spn_phone=spn_phone,
            sil_phone=sil_phone,
        )

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

        # We will train two models:
        # * arpa - uses a 3-gram language model
        # * arpa_rescore - uses a 5-gram language model for rescoring
        # * grammar - uses a fixed grammar
        for lang_type in (ARPA, GRAMMAR):
            ctx.lang_dir(lang_type).mkdir(parents=True, exist_ok=True)
            if ctx.graph_dir(lang_type).exists():
                shutil.rmtree(ctx.graph_dir(lang_type))

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
        self._prepare_lang(ctx, ARPA, GRAMMAR)

        # 2. Generate G.fst from skill graph
        self._create_grammar(ctx, fst_context.fst_file)
        self._create_arpa(ctx, fst_context.fst_file)

        if rescore_order is not None:
            self._prepare_lang(ctx, ARPA_RESCORE)
            self._create_arpa(
                ctx, fst_context.fst_file, order=rescore_order, suffix=ARPA_RESCORE
            )

        # 3. mkgraph.sh
        self._mkgraph(ctx, ARPA, GRAMMAR)

        # 4. prepare_online_decoding.sh
        self._prepare_online_decoding(ctx, ARPA, GRAMMAR)

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
        lexicon = ctx.fst_context.lexicon
        with open(dictionary_path, "w", encoding="utf-8") as dictionary_file:
            missing_words = set()
            for word in sorted(ctx.vocab):
                word_found = False
                for word_pron in lexicon.lookup(word):
                    phonemes_str = " ".join(word_pron)
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

    def _prepare_lang(self, ctx: TrainingContext, *lang_types: str):
        for lang_type in lang_types:
            prepare_lang = [
                "bash",
                str(ctx.egs_utils_dir / "prepare_lang.sh"),
                str(ctx.dict_local_dir),
                ctx.unk,
                str(ctx.lang_local_dir(lang_type)),
                str(ctx.lang_dir(lang_type)),
            ]

            _LOGGER.debug(prepare_lang)
            subprocess.check_call(prepare_lang, cwd=ctx.train_dir, env=ctx.extended_env)

    def _create_arpa(
        self,
        ctx: TrainingContext,
        fst_file: TextIO,
        order: int = 3,
        suffix: str = ARPA,
        method: str = "witten_bell",
    ):
        lang_dir = ctx.lang_dir(suffix)
        fst_path = lang_dir / "G.arpa.fst"
        text_fst_path = fst_path.with_suffix(".fst.txt")

        with open(text_fst_path, "w", encoding="utf-8") as text_fst_file:
            fst_file.seek(0)
            shutil.copyfileobj(fst_file, text_fst_file)

        compile_fst = [
            "fstcompile",
            shlex.quote(f"--isymbols={lang_dir}/words.txt"),
            shlex.quote(f"--osymbols={lang_dir}/words.txt"),
            "--keep_isymbols=true",
            "--keep_osymbols=true",
            shlex.quote(str(text_fst_path)),
            shlex.quote(str(fst_path)),
        ]

        _LOGGER.debug(compile_fst)
        ctx.run(compile_fst)

        counts_path = lang_dir / "G.ngram_counts.fst"
        ngram_counts = [
            "ngramcount",
            f"--order={order}",
            shlex.quote(str(fst_path)),
            shlex.quote(str(counts_path)),
        ]

        _LOGGER.debug(ngram_counts)
        ctx.run(ngram_counts)

        lm_fst_path = lang_dir / "G.lm.fst"
        ngram_make = [
            "ngrammake",
            f"--method={method}",
            shlex.quote(str(counts_path)),
            shlex.quote(str(lm_fst_path)),
        ]

        _LOGGER.debug(ngram_make)
        ctx.run(ngram_make)

        arpa_path = lang_dir / "lm.arpa"
        ngram_print = [
            "ngramprint",
            "--ARPA",
            shlex.quote(str(lm_fst_path)),
            shlex.quote(str(arpa_path)),
        ]

        _LOGGER.debug(ngram_print)
        ctx.run(ngram_print)

        lang_local_dir = ctx.lang_local_dir(ARPA)
        arpa_gz_path = lang_local_dir / "lm.arpa.gz"
        with open(arpa_path, "r", encoding="utf-8") as arpa_file, gzip.open(
            arpa_gz_path, "wt", encoding="utf-8"
        ) as arpa_gz_file:
            shutil.copyfileobj(arpa_file, arpa_gz_file)

        format_lm = [
            "bash",
            str(ctx.egs_utils_dir / "format_lm.sh"),
            str(lang_dir),
            str(arpa_gz_path),
            str(ctx.dict_local_dir / "lexicon.txt"),
            str(lang_dir),
        ]

        _LOGGER.debug(format_lm)
        ctx.run(format_lm)

    def _create_grammar(self, ctx: TrainingContext, fst_file: TextIO):
        lang_dir = ctx.lang_dir(GRAMMAR)
        fst_path = lang_dir / "G.fst"
        unsorted_fst_path = fst_path.with_suffix(".fst.unsorted")
        text_fst_path = fst_path.with_suffix(".fst.txt")

        with open(text_fst_path, "w", encoding="utf-8") as text_fst_file:
            fst_file.seek(0)
            shutil.copyfileobj(fst_file, text_fst_file)

        compile_grammar = [
            "fstcompile",
            shlex.quote(f"--isymbols={lang_dir}/words.txt"),
            shlex.quote(f"--osymbols={lang_dir}/words.txt"),
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

    def _mkgraph(self, ctx: TrainingContext, *lang_types: str):
        for lang_type in lang_types:
            lang_dir = ctx.lang_dir(lang_type)
            if not lang_dir.is_dir():
                continue

            mkgraph = [
                "bash",
                str(ctx.egs_utils_dir / "mkgraph.sh"),
                "--self-loop-scale",
                "1.0",
                str(lang_dir),
                str(ctx.model_dir / "model"),
                str(ctx.graph_dir(lang_type)),
            ]
            _LOGGER.debug(mkgraph)
            ctx.run(mkgraph)

    def _prepare_online_decoding(self, ctx: TrainingContext, *lang_types: str):
        for lang_type in lang_types:
            extractor_dir = ctx.model_dir / "extractor"
            if not extractor_dir.is_dir():
                continue

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
                str(ctx.lang_dir(lang_type)),
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
        model_type: ModelType = ModelType.NNET3,
    ):
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.graph_dir = Path(graph_dir)
        self.kaldi_bin_dir = Path(kaldi_bin_dir)

        self.decode_proc: Optional[subprocess.Popen] = None
        self.decode_proc_async: "Optional[asyncio.subprocess.Process]" = None

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
        if self.model_type == ModelType.NNET3:
            text = self._transcribe_wav_nnet3(str(wav_path))
        elif self.model_type == ModelType.GMM:
            text = self._transcribe_wav_gmm(str(wav_path))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if text:
            # Success
            return self._fix_text(text).strip()

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

        return text

    def _transcribe_wav_gmm(self, wav_path: str) -> str:
        # GMM decoding steps:
        # 1. compute-mfcc-feats
        # 2. compute-cmvn-stats
        # 3. apply-cmvn
        # 4. add-deltas
        # 5. gmm-latgen-faster
        with tempfile.TemporaryDirectory() as temp_dir:
            words_txt = self.graph_dir / "words.txt"
            mfcc_conf = self.model_dir / "conf" / "mfcc.conf"

            # 1. compute-mfcc-feats
            feats_cmd = [
                str(self.kaldi_bin_dir / "compute-mfcc-feats"),
                f"--config={mfcc_conf}",
                f"scp:echo utt1 {wav_path}|",
                f"ark,scp:{temp_dir}/feats.ark,{temp_dir}/feats.scp",
            ]
            _LOGGER.debug(feats_cmd)
            subprocess.check_call(feats_cmd)

            # 2. compute-cmvn-stats
            stats_cmd = [
                str(self.kaldi_bin_dir / "compute-cmvn-stats"),
                f"scp:{temp_dir}/feats.scp",
                f"ark,scp:{temp_dir}/cmvn.ark,{temp_dir}/cmvn.scp",
            ]
            _LOGGER.debug(stats_cmd)
            subprocess.check_call(stats_cmd)

            # 3. apply-cmvn
            norm_cmd = [
                str(self.kaldi_bin_dir / "apply-cmvn"),
                f"scp:{temp_dir}/cmvn.scp",
                f"scp:{temp_dir}/feats.scp",
                f"ark,scp:{temp_dir}/feats_cmvn.ark,{temp_dir}/feats_cmvn.scp",
            ]
            _LOGGER.debug(norm_cmd)
            subprocess.check_call(norm_cmd)

            # 4. add-deltas
            delta_cmd = [
                str(self.kaldi_bin_dir / "add-deltas"),
                f"scp:{temp_dir}/feats_cmvn.scp",
                f"ark,scp:{temp_dir}/deltas.ark,{temp_dir}/deltas.scp",
            ]
            _LOGGER.debug(delta_cmd)
            subprocess.check_call(delta_cmd)

            # 5. decode
            decode_cmd = [
                str(self.kaldi_bin_dir / "gmm-latgen-faster"),
                f"--word-symbol-table={words_txt}",
                f"{self.model_dir}/model/final.mdl",
                f"{self.graph_dir}/HCLG.fst",
                f"scp:{temp_dir}/deltas.scp",
                f"ark,scp:{temp_dir}/lattices.ark,{temp_dir}/lattices.scp",
            ]
            _LOGGER.debug(decode_cmd)
            subprocess.check_call(decode_cmd)

            try:
                lines = (
                    subprocess.check_output(decode_cmd, stderr=subprocess.STDOUT)
                    .decode()
                    .splitlines()
                )
            except subprocess.CalledProcessError as e:
                _LOGGER.exception("_transcribe_wav_gmm")
                _LOGGER.error(e.output)
                lines = []

            text = ""
            for line in lines:
                if line.startswith("utt1 "):
                    parts = line.split(maxsplit=1)
                    if len(parts) > 1:
                        text = parts[1]
                    break

            return text

    async def transcribe_wav_async(self, wav_path: Union[str, Path]) -> Optional[str]:
        """Speech to text from WAV data."""
        text = await self._transcribe_wav_nnet3_async(str(wav_path))

        if text:
            # Success
            return self._fix_text(text).strip()

        # Failure
        return None

    async def transcribe_wav_nnet3_rescore_async(
        self,
        wav_path: str,
        old_lang_dir: Union[str, Path],
        new_lang_dir: Union[str, Path],
        tools_dir: Union[str, Path],
        rescore_acoustic_scale: float = 0.1,
    ) -> str:
        old_lang_dir = Path(old_lang_dir)
        new_lang_dir = Path(new_lang_dir)
        tools_dir = Path(tools_dir)
        kaldi_dir = tools_dir / "kaldi"
        openfst_dir = tools_dir / "openfst"

        extended_env = os.environ.copy()
        bin_dirs: List[str] = [
            str(kaldi_dir / "bin"),
            str(kaldi_dir / "utils"),
            str(openfst_dir / "bin"),
        ]
        lib_dirs: List[str] = [str(kaldi_dir / "lib"), str(openfst_dir / "lib")]

        current_path = extended_env.get("PATH")
        if current_path:
            bin_dirs.append(current_path)

        current_lib_path = extended_env.get("LD_LIBRARY_PATH")
        if current_lib_path:
            lib_dirs.append(current_lib_path)

        extended_env["PATH"] = os.pathsep.join(bin_dirs)
        extended_env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs)

        # Get id for #0 disambiguation state
        phi_cmd = " | ".join(
            (
                shlex.join(("grep", "-w", "#0", str(new_lang_dir / "words.txt"))),
                shlex.join(
                    (
                        "awk",
                        "{print $2}",
                    )
                ),
            )
        )
        _LOGGER.debug(phi_cmd)

        try:
            proc = await asyncio.create_subprocess_shell(
                phi_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=extended_env,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                _LOGGER.error(stderr.decode())
                return ""

            phi = int(stdout.decode().strip())
        except subprocess.CalledProcessError as e:
            _LOGGER.exception("transcribe_wav_nnet3_rescore_async")
            _LOGGER.error(e.output)
            return ""

        # Create Ldet.fst
        ldet_cmd = (
            " | ".join(
                (
                    shlex.join((("fstprint", str(new_lang_dir / "L_disambig.fst")))),
                    shlex.join(("awk", f"{{if($4 != {phi}){{print;}}}}")),
                    "fstcompile",
                    "fstdeterminizestar",
                    shlex.join(
                        (
                            "fstrmsymbols",
                            str(new_lang_dir / "phones" / "disambig.int"),
                        )
                    ),
                )
            )
            + " > "
            + shlex.quote(str(new_lang_dir / "Ldet.fst"))
        )
        _LOGGER.debug(ldet_cmd)

        try:
            proc = await asyncio.create_subprocess_shell(
                ldet_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=extended_env,
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                _LOGGER.error(stderr.decode())
                return ""
        except subprocess.CalledProcessError as e:
            _LOGGER.exception("transcribe_wav_nnet3_rescore_async")
            _LOGGER.error(e.output)
            return ""

        model_file = self.model_dir / "model" / "final.mdl"
        words_txt = self.graph_dir / "words.txt"
        online_conf = self.model_dir / "online" / "conf" / "online.conf"

        kaldi_cmd = " | ".join(
            (
                shlex.join(
                    (
                        str(self.kaldi_bin_dir / "online2-wav-nnet3-latgen-faster"),
                        "--online=false",
                        "--do-endpointing=false",
                        f"--word-symbol-table={words_txt}",
                        f"--config={online_conf}",
                        *self.kaldi_args,
                        str(model_file),
                        str(self.graph_dir / "HCLG.fst"),
                        "ark:echo utt1 utt1|",
                        f"scp:echo utt1 {wav_path}|",
                        "ark:-",
                    )
                ),
                shlex.join(("lattice-scale", "--lm-scale=0.0", "ark:-", "ark:-")),
                shlex.join(
                    ("lattice-to-phone-lattice", str(model_file), "ark:-", "ark:-")
                ),
                shlex.join(
                    (
                        "lattice-compose",
                        "ark:-",
                        str(new_lang_dir / "Ldet.fst"),
                        "ark:-",
                    )
                ),
                shlex.join(("lattice-determinize", "ark:-", "ark:-")),
                shlex.join(
                    (
                        "lattice-compose",
                        f"--phi-label={phi}",
                        "ark:-",
                        str(new_lang_dir / "G.fst"),
                        "ark:-",
                    )
                ),
                shlex.join(
                    (
                        "lattice-add-trans-probs",
                        "--transition-scale=1.0",
                        "--self-loop-scale=0.1",
                        str(model_file),
                        "ark:-",
                        "ark:-",
                    ),
                ),
                shlex.join(
                    (
                        "lattice-best-path",
                        f'--word-symbol-table={new_lang_dir / "words.txt"}',
                        f"--acoustic-scale={rescore_acoustic_scale}",
                        "ark:-",
                        "ark,t:-",
                    )
                ),
                # TODO: nbest
                # shlex.join(
                #     (
                #         "lattice-to-nbest",
                #         "--n=5",
                #         f"--acoustic-scale={rescore_acoustic_scale}",
                #         "ark:-",
                #         "ark:-",
                #     )
                # ),
                # shlex.join(
                #     (
                #         "nbest-to-linear",
                #         "ark:-",
                #         "ark:/dev/null",  # alignments
                #         "ark,t:-",  # transcriptions
                #     )
                # ),
                shlex.join(
                    (
                        str(kaldi_dir / "utils" / "int2sym.pl"),
                        "-f",
                        "2-",
                        str(new_lang_dir / "words.txt"),
                    )
                ),
            )
        )

        _LOGGER.debug(kaldi_cmd)

        try:
            proc = await asyncio.create_subprocess_shell(
                kaldi_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=extended_env,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                _LOGGER.error(stderr.decode())
                return ""

            lines = stdout.decode().splitlines()
        except subprocess.CalledProcessError as e:
            _LOGGER.exception("transcribe_wav_nnet3_rescore_async")
            _LOGGER.error(e.output)
            lines = []

        text = ""
        for line in lines:
            if line.startswith("utt1 "):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    text = parts[1]
                    _LOGGER.debug(text)

        return self._fix_text(text.strip())

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

        return text

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
        assert self.decode_proc.stdin is not None
        assert self.decode_proc.stdout is not None
        assert self.chunk_fifo_file is not None

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
                tokens.append(word)

            return self._fix_text(" ".join(t for t in tokens))

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
        assert self.decode_proc_async.stdin is not None
        assert self.decode_proc_async.stdout is not None
        assert self.chunk_fifo_file is not None

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
                tokens.append(word)

            return self._fix_text(" ".join(t for t in tokens))

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
                # TODO: Add lattice writing option for rescoring
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

    async def run_async(self, command: List[str], **kwargs) -> bytes:
        if "stderr" not in kwargs:
            kwargs["stderr"] = asyncio.subprocess.STDOUT

        proc = await asyncio.create_subprocess_exec(
            command[0],
            *command[1:],
            stdout=asyncio.subprocess.PIPE,
            **kwargs,
        )

        stdout, _stderr = await proc.communicate()
        return stdout

    def _fix_text(self, text: str) -> str:
        return text


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
