#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

stt_source_dir="${program_dir}/coqui_stt/src"
stt_build_dir="${program_dir}/build/coqui_stt/stt_onlyprobs"
local_dir="${program_dir}/local"

mkdir -p "${local_dir}"
stt_install_dir="$(realpath ${local_dir})"
mkdir -p "${stt_build_dir}" "${stt_install_dir}"

# requires git and ca-certificates
pushd "${stt_build_dir}"
cmake \
    -DCMAKE_INSTALL_PREFIX="${stt_install_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DNATIVE_FEATURES=ON \
    -DTFLITE_ENABLE_BENCHMARK=OFF \
    "${stt_source_dir}"
cmake \
    --build . \
    --target install \
    -- -j$(nproc)
popd
