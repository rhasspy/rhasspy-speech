#!/usr/bin/env bash
this_dir="$( cd "$( dirname "$0" )" && pwd )"
program_dir="$(realpath "${this_dir}/..")"

src_dir="${program_dir}/opengrm"
openfst_local_dir="${program_dir}/local/openfst"
local_dir="${program_dir}/local/opengrm"

if [ ! -d "${openfst_local_dir}" ]; then
    echo 'You must run build_openfst.sh first'
    exit 1
fi

mkdir -p "${local_dir}"

pushd "${src_dir}"
export CFLAGS='-O2'
export CXXFLAGS="-O2 -I${openfst_local_dir}/include"
export LDFLAGS="-L${openfst_local_dir}/lib"
./configure --prefix="${local_dir}"
make -j8
make install
popd
