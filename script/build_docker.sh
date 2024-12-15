#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

platforms='linux/amd64,linux/arm64'
if [ -n "$1" ]; then
    platforms="$1"
fi

pushd "${program_dir}"
docker buildx build . \
    --platform "${platforms}" \
    --output 'type=local,dest=dist'
popd
