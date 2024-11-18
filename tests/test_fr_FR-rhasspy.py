import tempfile
from pathlib import Path

import pytest
from yaml import safe_dump

from rhasspy_speech import train_model, KaldiTranscriber

MODEL = "fr_FR-rhasspy"

_DIR = Path(__file__).parent
_LOCAL_DIR = _DIR.parent / "local"
_WAV_DIR = _DIR / MODEL


@pytest.fixture(scope="module")
def transcriber() -> KaldiTranscriber:
    model_dir = _LOCAL_DIR / "models" / MODEL
    train_dir = _DIR / "train" / MODEL
    train_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as sentences_file:
        safe_dump(
            {
                "sentences": [
                    text_file.read_text(encoding="utf-8")
                    for text_file in _WAV_DIR.glob("*.txt")
                    if text_file.is_file()
                ]
            },
            sentences_file,
        )

        sentences_file.seek(0)
        train_model(
            language=MODEL.split("-", maxsplit=1)[0].split("_", maxsplit=1)[0],
            sentence_files=[_DIR / sentences_file.name],
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


@pytest.mark.parametrize("wav_path", _WAV_DIR.glob("*.wav"))
def test_transcribe_wav(transcriber: KaldiTranscriber, wav_path: Path) -> None:
    text_path = wav_path.with_suffix(".txt")
    expected_transcript = text_path.read_text(encoding="utf-8").strip()
    actual_transcript = transcriber.transcribe_wav(wav_path)
    assert (
        actual_transcript == expected_transcript
    ), f"Expected '{expected_transcript}', got '{actual_transcript}' for {wav_path.name}"
