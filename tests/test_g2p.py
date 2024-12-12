"""Tests for grapheme-to-phoneme (g2p) methods."""
from rhasspy_speech.g2p import get_sounds_like, LexiconDatabase, split_words
from unicode_rbnf import RbnfEngine

from . import LOCAL_DIR


def test_sounds_like() -> None:
    model_dir = LOCAL_DIR / "models" / "en_US-rhasspy"
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
    model_dir = LOCAL_DIR / "models" / "en_US-rhasspy"
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
