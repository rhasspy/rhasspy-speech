from yaml import safe_load

from rhasspy_speech.sentences import generate_sentences


def test_in_out() -> None:
    sentences_yaml = safe_load(
        """
    sentences:
      - in: input text
        out: output text
      - just input text
    """
    )

    sentences = list(generate_sentences(sentences_yaml, "en"))
    assert set(sentences) == {
        ("input text", "output text"),
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

    sentences = list(generate_sentences(sentences_yaml, "en"))
    assert set(sentences) == {
        ("input test 1", "output test 1"),
        ("input test 2", "output test two"),
    }


def test_range() -> None:
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

    sentences = list(generate_sentences(sentences_yaml, "en"))
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

    sentences = list(generate_sentences(sentences_yaml, "en"))
    assert set(sentences) == {("a test 2", "a test 2")}
