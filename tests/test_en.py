import asyncio
import sqlite3
import wave
from collections.abc import AsyncIterable, Iterable
from pathlib import Path

import pytest

from rhasspy_speech import train_model, KaldiTranscriber
from rhasspy_speech.g2p import get_sounds_like, LexiconDatabase, split_words
from unicode_rbnf import RbnfEngine

_DIR = Path(__file__).parent
_LOCAL_DIR = _DIR.parent / "local"

RATE = 16000
WIDTH = 2
CHANNELS = 1


@pytest.fixture(scope="module")
def transcriber() -> KaldiTranscriber:
    model_dir = _LOCAL_DIR / "models" / "en_US-rhasspy"
    train_dir = _DIR / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    train_model(
        language="en",
        sentence_files=[_DIR / "test_en.yaml"],
        kaldi_dir=_LOCAL_DIR / "kaldi",
        model_dir=model_dir,
        train_dir=train_dir,
        phonetisaurus_bin=_LOCAL_DIR / "phonetisaurus",
        opengrm_dir=_LOCAL_DIR / "opengrm",
        openfst_dir=_LOCAL_DIR / "openfst",
    )

    return KaldiTranscriber(
        model_dir=model_dir / "model",
        graph_dir=train_dir / "graph_grammar",
        kaldi_bin_dir=_LOCAL_DIR / "kaldi" / "bin",
    )


def wav_chunks(wav_path: Path, chunk_samples: int) -> Iterable[bytes]:
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == RATE
        assert wav_file.getsampwidth() == WIDTH
        assert wav_file.getnchannels() == CHANNELS

        while True:
            chunk = wav_file.readframes(chunk_samples)
            if not chunk:
                break

            yield chunk


async def wav_chunks_async(wav_path: Path, chunk_samples: int) -> AsyncIterable[bytes]:
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == RATE
        assert wav_file.getsampwidth() == WIDTH
        assert wav_file.getnchannels() == CHANNELS

        while True:
            chunk = wav_file.readframes(chunk_samples)
            if not chunk:
                break

            yield chunk


@pytest.mark.parametrize("wav_path", (_DIR / "wav").glob("*.wav"))
def test_transcribe_wav(transcriber: KaldiTranscriber, wav_path: Path) -> None:
    expected_transcript = wav_path.stem.replace("-", " ")
    actual_transcript = transcriber.transcribe_wav(wav_path)
    assert (
        actual_transcript == expected_transcript
    ), f"Expected '{expected_transcript}', got '{actual_transcript}' for {wav_path.name}"


@pytest.mark.parametrize("wav_path", (_DIR / "wav").glob("*.wav"))
@pytest.mark.asyncio(scope="module")
async def test_transcribe_wav_async(
    transcriber: KaldiTranscriber, wav_path: Path
) -> None:
    expected_transcript = wav_path.stem.replace("-", " ")
    actual_transcript = await transcriber.transcribe_wav_async(wav_path)
    assert (
        actual_transcript == expected_transcript
    ), f"Expected '{expected_transcript}', got '{actual_transcript}' for {wav_path.name}"


@pytest.mark.parametrize("wav_path", (_DIR / "wav").glob("*.wav"))
def test_transcribe_stream(transcriber: KaldiTranscriber, wav_path: Path) -> None:
    expected_transcript = wav_path.stem.replace("-", " ")
    actual_transcript = transcriber.transcribe_stream(
        wav_chunks(wav_path, 1024), RATE, WIDTH, CHANNELS
    )
    assert (
        actual_transcript == expected_transcript
    ), f"Expected '{expected_transcript}', got '{actual_transcript}' for {wav_path.name}"


@pytest.mark.parametrize("wav_path", (_DIR / "wav").glob("*.wav"))
@pytest.mark.asyncio(scope="module")
async def test_transcribe_stream_async(
    transcriber: KaldiTranscriber, wav_path: Path
) -> None:
    expected_transcript = wav_path.stem.replace("-", " ")
    actual_transcript = await transcriber.transcribe_stream_async(
        wav_chunks_async(wav_path, 1024), RATE, WIDTH, CHANNELS
    )
    assert (
        actual_transcript == expected_transcript
    ), f"Expected '{expected_transcript}', got '{actual_transcript}' for {wav_path.name}"


def test_sounds_like() -> None:
    model_dir = _LOCAL_DIR / "models" / "en_US-rhasspy"
    lexicon = LexiconDatabase(model_dir / "lexicon.db")

    # Mixed case should work despite database being lower case
    prons = get_sounds_like(["bee", "YAWN", "sAy"], lexicon)
    assert prons == [["b", "ˈi", "j", "ˈɔ", "n", "s", "ˈeɪ"]]

    partial_prons = get_sounds_like(
        ["[be]at", "[y]es", "l[awn]", "[s]o", "d[ay]"], lexicon
    )
    assert partial_prons == prons

    mixed_prons = get_sounds_like(["bee", "/j", "ˈɔ", "n/", "say"], lexicon)
    assert mixed_prons == prons


def test_split_words() -> None:
    model_dir = _LOCAL_DIR / "models" / "en_US-rhasspy"
    lexicon = LexiconDatabase(model_dir / "lexicon.db")
    engine = RbnfEngine.for_language("en")

    # Initialisms
    assert split_words("HVAC", lexicon, engine) == ["H", "V", "A", "C"]
    assert split_words("H.V.A.C.", lexicon, engine) == ["H", "V", "A", "C"]

    # Known pronunciation
    lexicon.add("hvac", get_sounds_like(["h", "[vac]uum"], lexicon))
    assert split_words("HVAC", lexicon, engine) == ["HVAC"]

    # Word + number
    assert split_words("PM2.5", lexicon, engine) == ["P", "M", "two", "point", "five"]
