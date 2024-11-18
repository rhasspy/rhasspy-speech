#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import tempfile
import os

parser = argparse.ArgumentParser()
parser.add_argument("wav", nargs="+")
args = parser.parse_args()

with tempfile.TemporaryDirectory() as temp_dir:
    for wav_path in args.wav:
        subprocess.check_call(
            [
                "sox",
                wav_path,
                "-r",
                "16000",
                "-c",
                "1",
                "-t",
                "wav",
                os.path.join(temp_dir, "temp.wav"),
            ]
        )
        shutil.copy(os.path.join(temp_dir, "temp.wav"), wav_path)
        print(wav_path)
