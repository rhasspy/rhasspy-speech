#!/bin/bash
# Tests the command line binary ngramshrink.

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

compile_test_fst earnest-witten_bell.mod
for METHOD in count_prune relative_entropy seymore; do
  case "${METHOD}" in
    count_prune) PARAM="--count_pattern=3+:2" ;;
    relative_entropy) PARAM="--theta=.00015" ;;
    seymore) PARAM="--theta=4" ;;
  esac

  compile_test_fst "earnest-${METHOD}.pru"
  "${BIN}/ngramshrink" \
    --method="${METHOD}" \
    --check_consistency \
    "${PARAM}" \
    "${TEST_TMPDIR}/earnest-witten_bell.mod.ref" \
    "${TEST_TMPDIR}/${METHOD}.pru"

  fstequal \
    "${TEST_TMPDIR}/earnest-${METHOD}.pru.ref" \
    "${TEST_TMPDIR}/${METHOD}.pru"
done

for METHOD in relative_entropy seymore; do
  case "${METHOD}" in
    relative_entropy) TARGET=5897 ;;
    seymore) TARGET=5276 ;;
  esac

  "${BIN}/ngramshrink" \
    --method="${METHOD}" \
    --check_consistency \
    --target_number_of_ngrams="${TARGET}" \
    "${TEST_TMPDIR}/earnest-witten_bell.mod.ref" \
    "${TEST_TMPDIR}/${METHOD}.target.pru"

  fstequal \
    "${TEST_TMPDIR}/earnest-${METHOD}.pru.ref" \
    "${TEST_TMPDIR}/${METHOD}.target.pru"
done
