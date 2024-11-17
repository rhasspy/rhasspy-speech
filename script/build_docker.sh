#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

    # --platform 'linux/amd64,linux/arm64,linux/arm/v7' \

pushd "${program_dir}"
docker buildx build . \
    --platform 'linux/amd64' \
    --output 'type=local,dest=dist'
popd
