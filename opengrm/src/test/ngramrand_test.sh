# Random test of ngram functionality.
#
# Can set environment variable NGRAMRANDTRIALS, otherwise defaults to 0 trials.
# When called with an integer argument, will run that many trials, otherwise
# the default number as explained above.

set -eou pipefail

readonly BIN="../bin"
readonly TEST_TMPDIR="${TEST_TMPDIR:-$(mktemp -d)}"

readonly DEFTRIALS=${NGRAMRANDTRIALS:-0}
readonly TRIALS=${1:-$DEFTRIALS}
readonly VARFILE="${TEST_TMPDIR}/ngramrandtest.vars"

i=0
while [[ $i -lt $TRIALS ]]; do
  : $((i+=1))
  rm -rf "${TEST_TMPDIR}/."*
  # Runs random test, outputs various count and model files and variables.
  "./ngramrandtest" --directory="${TEST_TMPDIR}" --vars="${VARFILE}"
  # Reads in variables from rand test (SEED, ORDER ...).
  . "${VARFILE}"
  "${BIN}/ngramdistrand" "$SEED" "$ORDER"
  rm -rf "${TEST_TMPDIR}/${SEED}."* "${TEST_TMPDIR}/."*
done
