import io
from hassil.intents import Intents

from rhasspy_speech.hassil_fst import intents_to_fst, SPACE, G2PInfo
from rhasspy_speech.const import WordCasing
from rhasspy_speech.g2p import LexiconDatabase

INTENTS_YAML = """
language: en
intents:
  GetTime:
    data:
      - sentences:
          - "what time is it"
          - "what is the time"
  TurnOn:
    data:
      - sentences:
          - "turn on [the] {name}"

  GetState:
    data:
      - sentences:
          - "what is the {name}'s state"

  Todo:
    data:
      - sentences:
          - "add {item} to todo"

  SetBrightness:
    data:
      - sentences:
          - "set brightness to {brightness} percent"

  ShoppingList:
    data:
      - sentences:
          - "add {food} to shopping list"

lists:
  name:
    values:
      - tv
      - light
  item:
    wildcard: true
  brightness:
    range:
      from: 20
      to: 22
  food:
    values:
      - A1 Steak Sauce
      - NASA Moon Cakes
      - 0 A.D. DVD
"""


def test_text_only() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"GetTime"})
    assert fst.words == {SPACE, "what", "time", "is", "it", "the"}

    assert set(tuple(t) for t in fst.to_tokens()) == {
        ("what", SPACE, "time", SPACE, "is", SPACE, "it"),
        ("what", SPACE, "is", SPACE, "the", SPACE, "time"),
    }
    assert set(fst.to_strings(False)) == {"what time is it", "what is the time"}

    fst_without_spaces = fst.remove_spaces()
    assert fst_without_spaces.words == {"what", "time", "is", "it", "the"}
    assert set(tuple(t) for t in fst_without_spaces.to_tokens()) == {
        ("what", "time", "is", "it"),
        ("what", "is", "the", "time"),
    }
    assert set(fst_without_spaces.to_strings(True)) == {
        "what time is it",
        "what is the time",
    }


def test_lists() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"TurnOn"})
    assert set(fst.to_strings(False)) == {
        "turn on tv",
        "turn on light",
        "turn on the tv",
        "turn on the light",
    }

    fst = intents_to_fst(intents, include_intents={"GetState"}).remove_spaces()
    assert set(fst.to_strings(True)) == {
        "what is the tv's state",
        "what is the light's state",
    }

    fst = intents_to_fst(
        intents, number_language="en", include_intents={"SetBrightness"}
    ).remove_spaces()
    assert set(fst.to_strings(True)) == {
        "set brightness to twenty percent",
        "set brightness to twenty one percent",
        "set brightness to twenty two percent",
    }


def test_prune() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"Todo"})

    # Wildcard branch is dead
    assert not fst.to_strings(False)
    assert not fst.to_tokens(only_connected=True)

    # Branch is still in FST
    assert fst.to_tokens(only_connected=False) == [["add", SPACE, "{item}"]]

    # Branch is pruned
    fst.prune()
    assert not fst.to_tokens(only_connected=False)


def test_g2p() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    lexicon = LexiconDatabase()
    lexicon.add("NASA", [["nah", "suh"]])

    fst = intents_to_fst(
        intents,
        include_intents={"ShoppingList"},
        number_language="en",
        g2p_info=G2PInfo(lexicon, WordCasing.get_function(WordCasing.LOWER)),
    ).remove_spaces()
    assert set(fst.to_strings(True)) == {
        "add a one steak sauce to shopping list",
        "add nasa moon cakes to shopping list",
        "add zero a d d v d to shopping list",
    }
