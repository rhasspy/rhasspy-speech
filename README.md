# Rhasspy Speech

![logo](logo.png)

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

[Pre-built models](https://huggingface.co/datasets/rhasspy/rhasspy-speech/tree/main/models) and derived from the corresponding [voice2json models](https://github.com/synesthesiam/voice2json-profiles).

* Czech, Czech Republic
* German, Germany
* English, United States
* Spanish, Spain
* French, France
* Italian, Italy
* Dutch, Netherlands
* Russian, Russia

## Tools and Dependencies

[Pre-built tools](https://huggingface.co/datasets/rhasspy/rhasspy-speech/tree/main/tools) must be downloaded for `rhasspy-speech` to work. This includes:

* [Kaldi](https://kaldi-asr.org/)
* [openfst](https://www.openfst.org)
* [opengrm](https://www.opengrm.org)
* [phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus)

See the `build_*` scripts in `script/` for how these tools are built. See the `Dockerfile` and `script/build_docker.sh` for how they are packaged.

You must also have the following system packages installed at runtime:

* `libopenblas0`
* `libencode-perl`

## Handling Out of Vocabulary

`rhasspy-speech` generates two different Kaldi models from the sentence templates: one with a rigid grammar that only accepts the possible sentences, and another with a language model that allows new sentences to be made from the existing words.

Using both the grammar and language model, it's possible to robustly reject sentences outside of the templates. After transcripts are returned from both models, they can be compared to decide whether to accept or reject the grammar transcript.
