import asyncio
import wave
from collections.abc import AsyncIterable, Iterable
from pathlib import Path

import pytest

from rhasspy_speech import train_model, KaldiTranscriber

_DIR = Path(__file__).parent
_LOCAL_DIR = _DIR.parent / "local"

RATE = 16000
WIDTH = 2
CHANNELS = 1


@pytest.fixture(scope="module")
def transcriber() -> KaldiTranscriber:
    model_dir = _LOCAL_DIR / "models" / "en_US-rhasspy"

    train_model(
        language="en",
        sentence_files=[_DIR / "test_en.yaml"],
        kaldi_dir=_LOCAL_DIR / "kaldi",
        model_dir=model_dir,
        train_dir=_DIR / "train",
        phonetisaurus_bin=_LOCAL_DIR / "phonetisaurus",
    )

    return KaldiTranscriber(
        model_dir=model_dir / "model",
        graph_dir=_DIR / "train" / "graph",
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
