#!/bin/bash
# Tests the command line binary ngramsymbols.

set -eou pipefail

readonly BIN="../bin"
readonly TESTDATA="${srcdir}/testdata"
readonly TEST_TMPDIR="${TEST_TMPDIR:-$(mktemp -d)}"

"${BIN}/ngramsymbols" "${TESTDATA}/earnest.txt" "${TEST_TMPDIR}/earnest.sym"

cmp "${TESTDATA}/earnest.sym" "${TEST_TMPDIR}/earnest.sym"
