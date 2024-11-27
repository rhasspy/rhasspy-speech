import asyncio
import asyncio.subprocess
import gzip
import itertools
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from typing import List, Optional, Union

from .const import EPS, SIL, SPN, UNK, LangSuffix, ModelType
from .intent_fst import IntentsToFstContext
from .tools import KaldiTools

_LOGGER = logging.getLogger(__name__)


class KaldiTrainer:
    def __init__(
        self,
        train_dir: Union[str, Path],
        model_dir: Union[str, Path],
        tools: KaldiTools,
        fst_context: IntentsToFstContext,
        eps: str = EPS,
        unk: str = UNK,
        spn_phone: str = SPN,
        sil_phone: str = SIL,
    ) -> None:
        self.train_dir = Path(train_dir).absolute()
        self.model_dir = Path(model_dir).absolute()
        self.tools = tools
        self.fst_context = fst_context
        self.eps = eps
        self.unk = unk
        self.spn_phone = spn_phone
        self.sil_phone = sil_phone

    @property
    def conf_dir(self) -> Path:
        return self.train_dir / "conf"

    def graph_dir(self, suffix: Optional[str] = None) -> Path:
        if suffix:
            return self.train_dir / f"graph_{suffix}"

        return self.train_dir / "graph"

    @property
    def data_dir(self) -> Path:
        return self.train_dir / "data"

    @property
    def data_local_dir(self) -> Path:
        return self.data_dir / "local"

    def lang_dir(self, suffix: Optional[str] = None) -> Path:
        if suffix:
            return self.data_dir / f"lang_{suffix}"

        return self.data_dir / "lang"

    @property
    def dict_local_dir(self) -> Path:
        return self.data_local_dir / "dict"

    def lang_local_dir(self, suffix: Optional[str] = None) -> Path:
        if suffix:
            return self.data_local_dir / f"lang_{suffix}"

        return self.data_local_dir / "lang"

    # -------------------------------------------------------------------------

    async def train(
        self,
        rescore_order: Optional[int] = None,
    ) -> None:
        # Extend PATH
        self.train_dir.mkdir(parents=True, exist_ok=True)

        # Copy conf
        if self.conf_dir.exists():
            shutil.rmtree(self.conf_dir)

        shutil.copytree(self.model_dir / "conf", self.conf_dir)

        # Delete existing data/graph
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)

        for graph_dir in self.train_dir.glob("graph_*"):
            if not graph_dir.is_dir():
                continue

            shutil.rmtree(graph_dir)

        # ---------------------------------------------------------------------
        # Kaldi Training
        # ---------------------------------------------------------
        # 1. prepare_lang.sh
        # 2. format_lm.sh (or fstcompile)
        # 3. mkgraph.sh
        # 4. prepare_online_decoding.sh
        # ---------------------------------------------------------

        # Create empty path.sh
        path_sh = self.train_dir / "path.sh"
        if not path_sh.is_file():
            path_sh.write_text("")

        # Write pronunciation dictionary
        await self._create_lexicon()

        # Create utils link
        model_utils_link = self.train_dir / "utils"
        model_utils_link.unlink(missing_ok=True)
        model_utils_link.symlink_to(self.tools.egs_utils_dir, target_is_directory=True)

        # 1. prepare_lang.sh
        await self._prepare_lang(LangSuffix.GRAMMAR)
        await self._prepare_lang(LangSuffix.ARPA)

        # 2. Generate G.fst from skill graph
        await self._create_grammar(LangSuffix.GRAMMAR)
        await self._create_arpa(LangSuffix.ARPA)

        if rescore_order is not None:
            await self._prepare_lang(LangSuffix.ARPA_RESCORE)
            await self._create_arpa(LangSuffix.ARPA_RESCORE, order=5)

        # 3. mkgraph.sh
        await self._mkgraph(LangSuffix.GRAMMAR)
        await self._mkgraph(LangSuffix.ARPA)

        # 4. prepare_online_decoding.sh
        await self._prepare_online_decoding(LangSuffix.GRAMMAR)
        await self._prepare_online_decoding(LangSuffix.ARPA)

    # -------------------------------------------------------------------------

    async def _create_lexicon(self) -> None:
        _LOGGER.debug("Generating lexicon")
        dict_local_dir = self.data_local_dir / "dict"
        dict_local_dir.mkdir(parents=True, exist_ok=True)

        # Copy phones
        phones_dir = self.model_dir / "phones"
        for phone_file in phones_dir.glob("*.txt"):
            shutil.copy(phone_file, dict_local_dir / phone_file.name)

        # Create dictionary
        dictionary_path = dict_local_dir / "lexicon.txt"
        lexicon = self.fst_context.lexicon
        with open(dictionary_path, "w", encoding="utf-8") as dictionary_file:
            missing_words = set()
            for word in sorted(self.fst_context.vocab):
                if word in (self.unk,):
                    continue

                word_found = False
                for word_pron in lexicon.lookup(word):
                    phonemes_str = " ".join(word_pron)
                    print(word, phonemes_str, file=dictionary_file)
                    word_found = True

                if not word_found:
                    missing_words.add(word)

            if missing_words:
                g2p_model_path = self.model_dir.parent / "g2p.fst"
                missing_words_path = self.train_dir / "missing_words_dictionary.txt"
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
                        (
                            await self.tools.async_run(
                                str(self.tools.phonetisaurus_bin),
                                [
                                    f"--model={g2p_model_path}",
                                    f"--wordlist={missing_words_file.name}",
                                ],
                            )
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

            # Add <unk>
            print(self.unk, self.spn_phone, file=dictionary_file)

            for label in self.fst_context.meta_labels:
                print(label, self.sil_phone, file=dictionary_file)

    async def _prepare_lang(self, lang_type: LangSuffix) -> None:
        await self.tools.async_run(
            "bash",
            [
                str(self.tools.egs_utils_dir / "prepare_lang.sh"),
                str(self.dict_local_dir),
                self.unk,
                str(self.lang_local_dir(lang_type.value)),
                str(self.lang_dir(lang_type.value)),
            ],
            cwd=self.train_dir,
        )

    async def _create_arpa(
        self,
        lang_type: LangSuffix,
        order: int = 3,
        method: str = "witten_bell",
    ) -> None:
        lang_dir = self.lang_dir(lang_type.value)
        fst_path = lang_dir / "G.arpa.fst"
        text_fst_path = fst_path.with_suffix(".fst.txt")
        arpa_path = lang_dir / "lm.arpa"

        with open(text_fst_path, "w", encoding="utf-8") as text_fst_file:
            self.fst_context.fst_file.seek(0)
            shutil.copyfileobj(self.fst_context.fst_file, text_fst_file)

        await self.tools.async_run(
            "fstcompile",
            [
                shlex.quote(f"--isymbols={lang_dir}/words.txt"),
                shlex.quote(f"--osymbols={lang_dir}/words.txt"),
                "--keep_isymbols=true",
                "--keep_osymbols=true",
                shlex.quote(str(text_fst_path)),
                shlex.quote(str(fst_path)),
            ],
        )
        await self.tools.async_run_pipeline(
            [
                "ngramcount",
                f"--order={order}",
                shlex.quote(str(fst_path)),
                "-",
            ],
            [
                "ngrammake",
                f"--method={method}",
            ],
            [
                "ngramprint",
                "--ARPA",
                "-",
                shlex.quote(str(arpa_path)),
            ],
        )

        lang_local_dir = self.lang_local_dir(lang_type.value)
        arpa_gz_path = lang_local_dir / "lm.arpa.gz"
        with open(arpa_path, "r", encoding="utf-8") as arpa_file, gzip.open(
            arpa_gz_path, "wt", encoding="utf-8"
        ) as arpa_gz_file:
            shutil.copyfileobj(arpa_file, arpa_gz_file)

        await self.tools.async_run(
            "bash",
            [
                str(self.tools.egs_utils_dir / "format_lm.sh"),
                str(lang_dir),
                str(arpa_gz_path),
                str(self.dict_local_dir / "lexicon.txt"),
                str(lang_dir),
            ],
        )

    async def _create_grammar(self, lang_type: LangSuffix) -> None:
        fst_file = self.fst_context.fst_file
        lang_dir = self.lang_dir(lang_type.value)
        fst_path = lang_dir / "G.fst"
        text_fst_path = fst_path.with_suffix(".fst.txt")

        with open(text_fst_path, "w", encoding="utf-8") as text_fst_file:
            fst_file.seek(0)
            shutil.copyfileobj(fst_file, text_fst_file)

        await self.tools.async_run_pipeline(
            [
                "fstcompile",
                shlex.quote(f"--isymbols={lang_dir}/words.txt"),
                shlex.quote(f"--osymbols={lang_dir}/words.txt"),
                "--keep_isymbols=false",
                "--keep_osymbols=false",
                "--keep_state_numbering=true",
                shlex.quote(str(text_fst_path)),
                "-",
            ],
            ["fstdeterminize"],
            ["fstminimize"],
            [
                "fstarcsort",
                "--sort_type=ilabel",
                "-",
                shlex.quote(str(fst_path)),
            ],
        )

    async def _mkgraph(self, lang_type: LangSuffix) -> None:
        lang_dir = self.lang_dir(lang_type.value)
        if not lang_dir.is_dir():
            _LOGGER.warning("Lang dir does not exist: %s", lang_dir)
            return

        await self.tools.async_run(
            "bash",
            [
                str(self.tools.egs_utils_dir / "mkgraph.sh"),
                "--self-loop-scale",
                "1.0",
                str(lang_dir),
                str(self.model_dir / "model"),
                str(self.graph_dir(lang_type.value)),
            ],
        )

    async def _prepare_online_decoding(self, lang_type: LangSuffix) -> None:
        extractor_dir = self.model_dir / "extractor"
        if not extractor_dir.is_dir():
            _LOGGER.warning("Extractor dir does not exist: %s", extractor_dir)
            return

        # Generate online.conf
        mfcc_conf = self.model_dir / "conf" / "mfcc_hires.conf"
        await self.tools.async_run(
            "bash",
            [
                str(
                    self.tools.egs_steps_dir
                    / "online"
                    / "nnet3"
                    / "prepare_online_decoding.sh"
                ),
                "--mfcc-config",
                str(mfcc_conf),
                str(self.lang_dir(lang_type.value)),
                str(extractor_dir),
                str(self.model_dir / "model"),
                str(self.model_dir / "online"),
            ],
            cwd=self.train_dir,
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
                    .decode(encoding="utf-8")
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


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
