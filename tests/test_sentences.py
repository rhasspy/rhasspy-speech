from yaml import safe_load

import pytest
from unicode_rbnf import RbnfEngine

from rhasspy_speech.sentences import generate_sentences


@pytest.fixture
def number_engine() -> RbnfEngine:
    return RbnfEngine.for_language("en")


def test_in_out() -> None:
    sentences_yaml = safe_load(
        """
    sentences:
      - in: input text
        out: output text
      - in: just in text
      - in:
          - input text no out 1
          - input text no out 2
      - in:
          - input text with out 1
          - input text with out 2
        out: output text for multiple in
      - just input text
    """
    )

    sentences = list(generate_sentences(sentences_yaml))
    assert set(sentences) == {
        # one in, with out
        ("input text", "output text"),
        # one in, no out
        ("just in text", "just in text"),
        # multiple in, no out
        ("input text no out 1", "input text no out 1"),
        ("input text no out 2", "input text no out 2"),
        # multiple in, without
        ("input text with out 1", "output text for multiple in"),
        ("input text with out 2", "output text for multiple in"),
        # just text
        ("just input text", "just input text"),
    }


def test_in_out_list() -> None:
    sentences_yaml = safe_load(
        """
    sentences:
      - in: input {test}
        out: output {test}
    lists:
      test:
        values:
          - test 1
          - in: test 2
            out: test two
    """
    )

    sentences = list(generate_sentences(sentences_yaml))
    assert set(sentences) == {
        ("input test 1", "output test 1"),
        ("input test 2", "output test two"),
    }


def test_range(number_engine: RbnfEngine) -> None:
    sentences_yaml = safe_load(
        """
    sentences:
      - test {number}
    lists:
      number:
        range:
          from: 5
          to: 15
          step: 5
    """
    )

    sentences = list(generate_sentences(sentences_yaml, number_engine))
    assert set(sentences) == {
        ("test five", "test 5"),
        ("test ten", "test 10"),
        ("test fifteen", "test 15"),
    }


def test_list_context() -> None:
    sentences_yaml = safe_load(
        """
    sentences:
      - in: a {test}
        requires_context:
          key1: value 1
        excludes_context:
          key2: value 2
    lists:
      test:
        values:
          - in: test 1
          - in: test 2
            context:
              key1: value 1
          - in: test 3
            context:
              key1: value 1
              key2: value 2
          - in: test 4
            context:
              key2: value 2
    """
    )

    sentences = list(generate_sentences(sentences_yaml))

    # test 1 does not have context, so it passes
    # test 2 has matching context
    assert set(sentences) == {("a test 1", "a test 1"), ("a test 2", "a test 2")}
