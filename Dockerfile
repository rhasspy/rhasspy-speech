FROM debian:bullseye AS build
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential cmake patchelf \
        python3 libopenblas-dev

WORKDIR /build

# openFST
COPY openfst/ ./openfst/
COPY script/build_openfst.sh ./script/
RUN bash script/build_openfst.sh

# opengrm
COPY opengrm/ ./opengrm/
COPY script/build_opengrm.sh ./script/
RUN bash script/build_opengrm.sh

# Kaldi
COPY kaldi/ ./kaldi/
COPY script/build_kaldi.sh ./script/
RUN bash script/build_kaldi.sh

# phonetisaurus
COPY phonetisaurus/ ./phonetisaurus/
COPY script/build_phonetisaurus.sh ./script/
RUN bash script/build_phonetisaurus.sh

# Distribution
RUN tar -C local -czf "rhasspy-speech_${TARGETARCH}${TARGETVARIANT}.tar.gz" \
    openfst \
    opengrm \
    phonetisaurus \
    kaldi

# -----------------------------------------------------------------------------

FROM scratch

COPY --from=build /build/rhasspy-speech_*.tar.gz ./
