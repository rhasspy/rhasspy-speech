#!/usr/bin/env python3
from pathlib import Path

import setuptools
from setuptools import setup

this_dir = Path(__file__).parent

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r", encoding="utf-8") as requirements_file:
        requirements = requirements_file.read().splitlines()

module = "rhasspy_speech"

# -----------------------------------------------------------------------------


setup(
    name=module,
    version="1.0.0",
    description="Rhasspy Speech-to-text",
    url="http://github.com/rhasspy/rhasspy-speech",
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="rhasspy wyoming stt",
)
