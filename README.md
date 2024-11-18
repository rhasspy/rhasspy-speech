# Rhasspy Speech

Port of the speech-to-text system from [Rhasspy](https://rhasspy.readthedocs.io/en/latest/). This uses [Kaldi](https://kaldi-asr.org/) under the hood to recognize sentences from a set of pre-defined templates.

For example, the template:

``` yaml
sentences:
  - turn (on|off) [the] light
```

will allow `rhasspy-speech` to recognize the sentences:

* turn on light
* turn off light
* turn on the light
* turn off the light

## Supported Languages

[Pre-built models](https://huggingface.co/datasets/rhasspy/rhasspy-speech/tree/main) and derived from the corresponding [voice2json models](https://github.com/synesthesiam/voice2json-profiles).

* Czech, Czech Republic
* German, Germany
* English, United States
* Spanish, Spain
* French, France
* Italian, Italy
* Dutch, Netherlands
* Russian, Russia

## Handling Out of Vocabulary

`rhasspy-speech` generates two different Kaldi models from the sentence templates: one with a rigid grammar that only accepts the possible sentences, and another with a language model that allows new sentences to be made from the existing words.

Using both the grammar and language model, it's possible to robustly reject sentences outside of the templates. After transcripts are returned from both models, they can be compared to decide whether to accept or reject the grammar transcript.
