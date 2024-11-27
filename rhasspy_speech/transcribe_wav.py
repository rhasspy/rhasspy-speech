"""Transcribe WAV files."""

import asyncio
import shlex
from pathlib import Path
from typing import List, Optional, Union

from .tools import KaldiTools


class KaldiNnet3WavTranscriber:
    def __init__(
        self,
        model_dir: Union[str, Path],
        graph_dir: Union[str, Path],
        tools: KaldiTools,
        max_active: int = 7000,
        lattice_beam: float = 8.0,
        acoustic_scale: float = 1.0,
        beam: float = 24.0,
    ):
        self.model_dir = Path(model_dir)
        self.graph_dir = Path(graph_dir)
        self.tools = tools

        self.max_active = max_active
        self.lattice_beam = lattice_beam
        self.acoustic_scale = acoustic_scale
        self.beam = beam

    async def async_transcribe(self, wav_path: Union[str, Path]) -> Optional[str]:
        words_txt = self.graph_dir / "words.txt"
        online_conf = self.model_dir / "model" / "online" / "conf" / "online.conf"
        stdout = await self.tools.async_run(
            "online2-wav-nnet3-latgen-faster",
            [
                "--online=false",
                "--do-endpointing=false",
                f"--word-symbol-table={words_txt}",
                f"--config={online_conf}",
                f"--max-active={self.max_active}",
                f"--lattice-beam={self.lattice_beam}",
                f"--acoustic-scale={self.acoustic_scale}",
                f"--beam={self.beam}",
                str(self.model_dir / "model" / "model" / "final.mdl"),
                str(self.graph_dir / "HCLG.fst"),
                "ark:echo utt1 utt1|",
                f"scp:echo utt1 {wav_path}|",
                "ark:/dev/null",
            ],
            stderr=asyncio.subprocess.STDOUT,
        )

        lines = stdout.decode(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("utt1 "):
                parts = line.strip().split(maxsplit=1)
                if len(parts) > 1:
                    return parts[1]

        return None

    async def async_transcribe_rescore(
        self,
        wav_path: Union[str, Path],
        old_lang_dir: Union[str, Path],
        new_lang_dir: Union[str, Path],
        rescore_acoustic_scale: float = 0.5,
        nbest: int = 1,
    ) -> List[str]:
        old_lang_dir = Path(old_lang_dir)
        new_lang_dir = Path(new_lang_dir)

        # Get id for #0 disambiguation state
        phi: Optional[int] = None
        with open(str(new_lang_dir / "words.txt"), "r", encoding="utf-8") as words_file:
            for line in words_file:
                if line.startswith("#0 "):
                    phi = int(line.strip().split(maxsplit=1)[1])
                    break

        if phi is None:
            raise ValueError("No value for disambiguation state (#0)")

        # Create Ldet.fst
        await self.tools.async_run_pipeline(
            ["fstprint", str(new_lang_dir / "L_disambig.fst")],
            ["awk", f"{{if($4 != {phi}){{print;}}}}"],
            ["fstcompile"],
            ["fstdeterminizestar"],
            [
                "fstrmsymbols",
                str(new_lang_dir / "phones" / "disambig.int"),
                "-",
                shlex.quote(str(new_lang_dir / "Ldet.fst")),
            ],
        )

        model_file = self.model_dir / "model" / "model" / "final.mdl"
        words_txt = self.graph_dir / "words.txt"
        online_conf = self.model_dir / "model" / "online" / "conf" / "online.conf"

        stdout = await self.tools.async_run_pipeline(
            [
                "online2-wav-nnet3-latgen-faster",
                "--online=false",
                "--do-endpointing=false",
                f"--word-symbol-table={words_txt}",
                f"--config={online_conf}",
                f"--max-active={self.max_active}",
                f"--lattice-beam={self.lattice_beam}",
                f"--acoustic-scale={self.acoustic_scale}",
                f"--beam={self.beam}",
                str(model_file),
                str(self.graph_dir / "HCLG.fst"),
                "ark:echo utt1 utt1|",
                f"scp:echo utt1 {wav_path}|",
                "ark:-",
            ],
            ["lattice-scale", "--lm-scale=0.0", "ark:-", "ark:-"],
            ["lattice-to-phone-lattice", str(model_file), "ark:-", "ark:-"],
            [
                "lattice-compose",
                "ark:-",
                str(new_lang_dir / "Ldet.fst"),
                "ark:-",
            ],
            ["lattice-determinize", "ark:-", "ark:-"],
            [
                "lattice-compose",
                f"--phi-label={phi}",
                "ark:-",
                str(new_lang_dir / "G.fst"),
                "ark:-",
            ],
            [
                "lattice-add-trans-probs",
                "--transition-scale=1.0",
                "--self-loop-scale=0.1",
                str(model_file),
                "ark:-",
                "ark:-",
            ],
            [
                "lattice-to-nbest",
                f"--n={nbest}",
                f"--acoustic-scale={rescore_acoustic_scale}",
                "ark:-",
                "ark:-",
            ],
            [
                "nbest-to-linear",
                "ark:-",
                "ark:/dev/null",  # alignments
                "ark,t:-",  # transcriptions
            ],
            [
                str(self.tools.egs_utils_dir / "int2sym.pl"),
                "-f",
                "2-",
                str(new_lang_dir / "words.txt"),
            ],
            stderr=asyncio.subprocess.STDOUT,
        )

        texts: List[str] = []
        for line in stdout.decode().splitlines():
            if line.startswith("utt1-"):
                parts = line.strip().split(maxsplit=1)
                if len(parts) > 1:
                    texts.append(parts[1])

        return texts
