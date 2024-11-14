#!/bin/bash
# Tests the command line binary ngrammake.

set -eou pipefail

readonly BIN="../bin"
readonly TESTDATA="${srcdir}/testdata"
readonly TEST_TMPDIR="${TEST_TMPDIR:-$(mktemp -d)}"

compile_test_fst() {
  fstcompile \
    --isymbols="${TESTDATA}/${1}.sym" \
    --osymbols="${TESTDATA}/${1}.sym" \
    --keep_isymbols \
    --keep_osymbols \
    --keep_state_numbering \
    "${TESTDATA}/${1}.txt" \
    "${TEST_TMPDIR}/${1}.ref"
}

# Default method.
compile_test_fst earnest.cnts
compile_test_fst earnest.mod
"${BIN}/ngrammake" \
  --check_consistency \
  "${TEST_TMPDIR}/earnest.cnts.ref" \
  "${TEST_TMPDIR}/earnest.mod"
fstequal \
  "${TEST_TMPDIR}/earnest.mod.ref" \
  "${TEST_TMPDIR}/earnest.mod"

# Specified methods.
for METHOD in absolute katz witten_bell kneser_ney unsmoothed; do
  compile_test_fst "earnest-${METHOD}.mod"
  "${BIN}/ngrammake" \
    --method="${METHOD}" \
    --check_consistency \
    "${TEST_TMPDIR}/earnest.cnts.ref" \
    "${TEST_TMPDIR}/earnest-${METHOD}.mod"
  fstequal \
    "${TEST_TMPDIR}/earnest-${METHOD}.mod.ref" \
    "${TEST_TMPDIR}/earnest-${METHOD}.mod"
done

# Fractional counting.
farcompilestrings \
  --fst_type=compact \
  --symbols="${TESTDATA}/earnest.sym" \
  --keep_symbols \
  "${TESTDATA}/earnest.txt" \
  "${TEST_TMPDIR}/earnest.far"

"${BIN}/ngramcount" \
  --method=histograms \
  "${TEST_TMPDIR}/earnest.far" \
  "${TEST_TMPDIR}/earnest.hsts"

"${BIN}/ngramcount" \
  --method=counts \
  "${TEST_TMPDIR}/earnest.far" \
  "${TEST_TMPDIR}/earnest.cnts"

"${BIN}/ngrammake" \
  --method=katz_frac \
  --check_consistency \
  "${TEST_TMPDIR}/earnest.hsts" \
  "${TEST_TMPDIR}/earnest-katz_frac.mod"

"${BIN}/ngrammake" \
  --method=katz \
  --bins=4 \
  --check_consistency \
  "${TEST_TMPDIR}/earnest.cnts" \
  "${TEST_TMPDIR}/earnest-katz_frac.mod.ref"

fstequal \
  "${TEST_TMPDIR}/earnest-katz_frac.mod.ref" \
  "${TEST_TMPDIR}/earnest-katz_frac.mod"
