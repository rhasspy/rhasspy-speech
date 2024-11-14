import argparse
import csv
import logging
import os
import sys

from .kaldi import KaldiTranscriber
from .train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--language", required=True)
    train_parser.add_argument("--kaldi-dir", required=True)
    train_parser.add_argument("--model-dir", required=True)
    train_parser.add_argument("--train-dir", required=True)
    train_parser.add_argument(
        "--sentence-file",
        required=True,
        action="append",
        help="Path to YAML sentence file",
    )
    train_parser.add_argument("--phonetisaurus-bin", required=True)
    train_parser.add_argument("--opengrm-dir", required=True)
    train_parser.add_argument("--openfst-dir", required=True)

    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("--kaldi-dir", required=True)
    transcribe_parser.add_argument("--model-dir", required=True)
    transcribe_parser.add_argument("--train-dir", required=True)
    transcribe_parser.add_argument("--wav", action="append", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.command == "train":
        train_model(
            language=args.language,
            sentence_files=args.sentence_file,
            kaldi_dir=args.kaldi_dir,
            model_dir=args.model_dir,
            train_dir=args.train_dir,
            phonetisaurus_bin=args.phonetisaurus_bin,
            opengrm_dir=args.opengrm_dir,
            openfst_dir=args.openfst_dir,
        )
    elif args.command == "transcribe":
        transcriber = KaldiTranscriber(
            model_dir=os.path.join(args.model_dir, "model"),
            graph_dir=os.path.join(args.train_dir, "graph"),
            sentences_db_path=os.path.join(args.train_dir, "sentences.db"),
            kaldi_bin_dir=os.path.join(args.kaldi_dir, "bin"),
        )
        writer = csv.writer(sys.stdout, delimiter="|")
        for wav_path in args.wav:
            text = transcriber.transcribe_wav(wav_path) or ""
            writer.writerow((wav_path, text))


if __name__ == "__main__":
    main()
