#!/bin/bash
# Random test of distributed training functions.

set -eou pipefail

readonly TEST_TMPDIR="${TEST_TMPDIR:-$(mktemp -d)}"

readonly BIN="../bin"

readonly RANDF="$1"  # Input file prefix.
readonly ORDER="$2"  # Order of ngrams.
readonly VERBOSE="$3"

# Tests FST equality after assuring same ordering.
ngramequal() {
  "${BIN}/ngramsort" "$1" "${TEST_TMPDIR}/${RANDF}.eq1"
  "${BIN}/ngramsort" "$2" "${TEST_TMPDIR}/${RANDF}.eq2"
  fstequal \
     -v=1 \
     --delta=0.01 \
    "${TEST_TMPDIR}/${RANDF}.eq1" \
    "${TEST_TMPDIR}/${RANDF}.eq2"
}

distributed_test() {
  # Non-distributed version.
  "$bin/ngramdisttrain" \
    --order="$ORDER" \
    "${VERBOSE}" \
    --round_to_int \
    --itype=fst_sents \
    --ifile="${TEST_TMPDIR}/${RANDF}.tocount.far" \
    --ofile="${TEST_TMPDIR}/${RANDF}.nodist" \
    --symbols="${TEST_TMPDIR}/${RANDF}.sym" \
    "$@"

  # Distributed version.
  "$bin/ngramdisttrain" \
    --contexts="${TEST_TMPDIR}/${RANDF}.cntxs" \
    --merge_contexts \
    --order="${ORDER}" \
    "${VERBOSE}" \
    --round_to_int \
    --itype=fst_sents \
    --ifile="${TEST_TMPDIR}/${RANDF}.tocount.far."* \
    --ofile="${TEST_TMPDIR}/${RANDF}.dist" \
    --symbols="${TEST_TMPDIR}/${RANDF}.sym" \
    "$@"

  # Verifies non-distributed and distributed versions give the same result.
  ngramequal "${TEST_TMPDIR}/${RANDF}.nodist" "${TEST_TMPDIR}/${RANDF}.dist"
}

distributed_test --otype=counts

distributed_test --otype=lm --smooth_method=katz
distributed_test --otype=lm --smooth_method=absolute
distributed_test --otype=lm --smooth_method=witten_bell

distributed_test \
  --otype=pruned_lm \
  --smooth_method=katz \
  --shrink_method=relative_entropy \
  --theta=.00015
distributed_test \
  --otype=pruned_lm \
  --smooth_method=katz \
  --shrink_method=seymore \
  --theta=4
distributed_test \
  --otype=pruned_lm \
  --smooth_method=witten_bell \
  --shrink_method=relative_entropy \
  --theta=.00015
