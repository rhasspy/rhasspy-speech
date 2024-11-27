import asyncio
import sqlite3
import wave
from collections.abc import AsyncIterable, Iterable
from pathlib import Path

import pytest
import pytest_asyncio

from . import TOOLS, TESTS_DIR, LOCAL_DIR

from rhasspy_speech import train_model
from rhasspy_speech.transcribe_wav import KaldiNnet3WavTranscriber
from rhasspy_speech.intent_fst import get_matching_scores

# from rhasspy_speech.g2p import get_sounds_like, LexiconDatabase, split_words
# from unicode_rbnf import RbnfEngine

_DIR = Path(__file__).parent


@pytest_asyncio.fixture(scope="module")
async def transcriber() -> KaldiNnet3WavTranscriber:
    # await train_model(
    #     language="en",
    #     sentence_files=[TESTS_DIR / "test_en.yaml"],
    #     model_dir=LOCAL_DIR / "models" / "en_US-rhasspy",
    #     train_dir=TESTS_DIR / "train_en",
    #     tools=TOOLS,
    #     rescore_order=5,
    # )

    return KaldiNnet3WavTranscriber(
        model_dir=LOCAL_DIR / "models" / "en_US-rhasspy",
        graph_dir=TESTS_DIR / "train_en" / "graph_arpa",
        tools=TOOLS,
    )


@pytest.mark.parametrize("wav_path", (_DIR / "wav").glob("*.wav"))
@pytest.mark.asyncio
async def test_transcribe_wav(
    transcriber: KaldiNnet3WavTranscriber, wav_path: Path
) -> None:
    expected_transcript = wav_path.stem.replace("-", " ")
    if expected_transcript.startswith("oov_"):
        # Out of vocabulary
        expected_transcript = ""

    nbest = await transcriber.async_transcribe_rescore(
        wav_path,
        old_lang_dir=TESTS_DIR / "train_en" / "data" / "lang_arpa",
        new_lang_dir=TESTS_DIR / "train_en" / "data" / "lang_arpa_rescore",
        nbest=5,
    )
    best = get_matching_scores(nbest, TESTS_DIR / "train_en" / "sentences.db")
    actual_transcript = best[0]
    norm_score = best[1] / len(actual_transcript)
    if norm_score > 0.15:
        # Too many changes required
        actual_transcript = ""

    assert (
        actual_transcript == expected_transcript
    ), f"Expected '{expected_transcript}', got '{actual_transcript}' for {wav_path.name}"


# def test_sounds_like() -> None:
#     model_dir = _LOCAL_DIR / "models" / "en_US-rhasspy"
#     lexicon = LexiconDatabase(model_dir / "lexicon.db")

#     # Mixed case should work despite database being lower case
#     prons = get_sounds_like(["bee", "YAWN", "sAy"], lexicon)
#     assert prons == [["b", "ˈi", "j", "ˈɔ", "n", "s", "ˈeɪ"]]

#     partial_prons = get_sounds_like(
#         ["[be]at", "[y]es", "l[awn]", "[s]o", "d[ay]"], lexicon
#     )
#     assert partial_prons == prons

#     mixed_prons = get_sounds_like(["bee", "/j", "ˈɔ", "n/", "say"], lexicon)
#     assert mixed_prons == prons


# def test_split_words() -> None:
#     model_dir = _LOCAL_DIR / "models" / "en_US-rhasspy"
#     lexicon = LexiconDatabase(model_dir / "lexicon.db")
#     engine = RbnfEngine.for_language("en")

#     # Initialisms
#     assert split_words("HVAC", lexicon, engine) == ["H", "V", "A", "C"]
#     assert split_words("H.V.A.C.", lexicon, engine) == ["H", "V", "A", "C"]

#     # Known pronunciation
#     lexicon.add("hvac", get_sounds_like(["h", "[vac]uum"], lexicon))
#     assert split_words("HVAC", lexicon, engine) == ["HVAC"]

#     # Word + number
#     assert split_words("PM2.5", lexicon, engine) == ["P", "M", "two", "point", "five"]
