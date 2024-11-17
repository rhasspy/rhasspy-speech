#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

src_dir="${program_dir}/openfst"
local_dir="${program_dir}/local/openfst"

mkdir -p "${local_dir}"

pushd "${src_dir}"
export CFLAGS='-O2'
export CXXFLAGS='-O2'
./configure --prefix="${local_dir}" --enable-grm
make -j$(nproc)
make install

find "${local_dir}/openfst/bin" -type f -executable \
    -exec patchelf --set-rpath '$ORIGIN/../lib' {}
popd
