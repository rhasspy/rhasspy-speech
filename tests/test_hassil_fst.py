import io
from hassil.intents import Intents

from rhasspy_speech.hassil_fst import intents_to_fst, SPACE

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

lists:
  name:
    values:
      - tv
      - light
  item:
    wildcard: true
"""


def test_text_only() -> None:
    with io.StringIO(INTENTS_YAML) as intents_file:
        intents = Intents.from_yaml(intents_file)

    fst = intents_to_fst(intents, include_intents={"GetTime"})
    assert set(tuple(t) for t in fst.to_tokens()) == {
        ("what", SPACE, "time", SPACE, "is", SPACE, "it"),
        ("what", SPACE, "is", SPACE, "the", SPACE, "time"),
    }
    assert set(fst.to_strings(False)) == {"what time is it", "what is the time"}

    fst_without_spaces = fst.remove_spaces()
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
