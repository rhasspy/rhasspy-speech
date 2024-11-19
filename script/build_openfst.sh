#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

src_dir="${program_dir}/openfst"
local_dir="${program_dir}/local/openfst"

mkdir -p "${local_dir}"

pushd "${src_dir}"
export CFLAGS='-O2'
export CXXFLAGS='-O2'
./configure --prefix="${local_dir}" --enable-grm --enable-far --enable-bin
make -j$(nproc)
make install

# Necessary on ARM for some reason
make -C src/extensions/far -j$(nproc) install
popd
