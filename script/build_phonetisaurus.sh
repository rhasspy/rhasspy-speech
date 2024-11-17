#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

src_dir="${program_dir}/phonetisaurus"
kaldi_build_dir="${program_dir}/build/kaldi"
kaldi_openfst_dir="${kaldi_build_dir}/openfst/src/include"
build_dir="${program_dir}/build/phonetisaurus"
local_dir="${program_dir}/local"

if [ ! -d "${kaldi_openfst_dir}" ]; then
    echo 'You must run build_kaldi.sh first'
    exit 1
fi

mkdir -p "${local_dir}" "${build_dir}"

pushd "${build_dir}"

# Use Kaldi's openFST since it's old enough
export CFLAGS='-O2'
export CXXFLAGS="-O2 -I${kaldi_openfst_dir}"
export LDFLAGS="-L${kaldi_build_dir}"
cmake "${src_dir}"
cmake --build . -- -j$(nproc)
cp phonetisaurus "${local_dir}/"

# local/phonetisaurus RPATH -> local/kaldi/lib
patchelf --set-rpath '$ORIGIN/kaldi/lib' "${local_dir}/phonetisaurus"
popd
