#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

kaldi_source_dir="${program_dir}/kaldi"
kaldi_build_dir="${program_dir}/build/kaldi"
local_dir="${program_dir}/local"
kaldi_install_dir="$(realpath ${local_dir}/kaldi)"

mkdir -p "${kaldi_build_dir}" "${kaldi_install_dir}"

pushd "${kaldi_build_dir}"
cmake \
    -DCMAKE_INSTALL_PREFIX="${kaldi_install_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DMATHLIB=OpenBLAS \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    -DKALDI_BUILD_TEST=OFF \
    "${kaldi_source_dir}"
cmake \
    --build . \
    --target install \
    -- -j8
popd

# Copy wsj example
cp -R "${kaldi_source_dir}/egs/wsj/s5/steps" "${kaldi_install_dir}/"
cp -R "${kaldi_source_dir}/egs/wsj/s5/utils" "${kaldi_install_dir}/"
