#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

src_dir="${program_dir}/openfst"
local_dir="${program_dir}/local/openfst"

mkdir -p "${local_dir}"

pushd "${src_dir}"
export LDFLAGS="-Wl,-rpath,'$ORIGIN/../lib'"
./configure --prefix="${local_dir}" --enable-grm
make -j8
make install
popd
