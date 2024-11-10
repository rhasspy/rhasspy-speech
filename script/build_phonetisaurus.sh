#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

src_dir="${program_dir}/phonetisaurus"
build_dir="${program_dir}/build/phonetisaurus"
local_dir="${program_dir}/local"

mkdir -p "${local_dir}"

mkdir -p "${build_dir}"
pushd "${build_dir}"
cmake "${src_dir}" -- -j8
cmake --build . -- -j8
cp phonetisaurus "${local_dir}/"
popd
