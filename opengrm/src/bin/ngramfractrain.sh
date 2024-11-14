#!/bin/bash
# Convenience script for training (fractional count derived) models in a
# distributed fashion.

set -euo pipefail

readonly BIN="../bin"
readonly DIR="$(mktemp -dt $$.XXXXX)"

trap "rm -fr ${DIR}" EXIT

IFILE=""
OFILE=""
ITYPE=""
OTYPE=""
CONTEXTS=""
MERGE_CONTEXTS=false
SYMBOLS=""
USAGE=false
VERBOSE=false

# (1) text sents -> FST sents flags
OOV_SYMBOL=""

# (2) FST sents -> counts flags
ORDER=3
EPSILON_AS_BACKOFF=false
ROUND_TO_INT=false

# (3) FST counts -> LM flags
BINS=-1

# (4) LM -> pruned LM flags
THETA=.0
SHRINK_METHOD=seymore

UPDATED=false

while [[ $# != 0 ]]; do
  # Parse 'option=optarg' word
  OPT="$(awk 'BEGIN { split(ARGV[1], a, "=") ; print a[1] }' "${1}")"
  ARG="$(awk 'BEGIN { split(ARGV[1], a, "=") ; print a[2] }' "${1}")"
  shift
  case "${OPT}" in
    --bins|-bins)
      BINS="${ARG}" ;;
    --contexts|-contexts)
      CONTEXTS="${ARG}" ;;
    --epsilon_as_backoff|-epsilon_as_backoff)
      EPSILON_AS_BACKOFF=true ;;
    --help|-help)
      USAGE=true ;;
    --ifile|-ifile)
      IFILE="${ARG}" ;;
    --itype|-itype)
      ITYPE="${ARG}"
      if [[ ${ITYPE} != text_sents && ${ITYPE} != fst_sents && \
          ${ITYPE} != counts && ${ITYPE} != lm ]]; then
        echo "ERROR: bad input type: ${ITYPE}"
        exit 1
      fi ;;
    --merge_contexts|-merge_contexts)
      MERGE_CONTEXTS=true ;;
    --ofile|-ofile)
      OFILE="${ARG}" ;;
    --order|-order)
      ORDER="${ARG}" ;;
    --otype|-otype)
      OTYPE="${ARG}"
      if [[ ${OTYPE} != fst_sents && ${OTYPE} != counts && \
        ${OTYPE} != lm && ${OTYPE} != pruned_lm ]]; then
        echo "ERROR: bad output type: ${OTYPE}"
        exit 1
      fi ;;
    --shrink_method|-shrink_method)
      SHRINK_METHOD="${ARG}" ;;
    --symbols|-symbols)
      SYMBOLS="${ARG}" ;;
    --OOV_symbol|-OOV_symbol)
      OOV_SYMBOL="${ARG}" ;;
    --theta|-theta)
      THETA="${ARG}" ;;
    --verbose|-verbose)
      VERBOSE=true ;;
    *)
      echo "bad option: $OPT"
      exit 1 ;;
  esac
done

if [[ -z ${IFILE} || -z ${OFILE} || -z ${ITYPE} || -z ${OTYPE} || \
    ${USAGE} = true ]]; then
  echo "Usage: $0  [--options] --ifile <infile> --ofile <outfile>\
 --itype <input type> --otype <output type>"
  echo
  echo "General flags:"
  echo "  --contexts            context pattern filename"
  echo "  --ifile               input filename pattern"
  echo "  --itype               input format, one of:"
  echo "    \"text_sents\", \"fst_sents\", \"counts\", \"lm\""
  echo "  --merge_contexts      merge_contexts in result"
  echo "  --ofile               output filename pattern"
  echo "  --otype               output format, one of:"
  echo "    \"fst_sents\", \"counts\", \"lm\", \"pruned_lm\""
  echo "  --symbols             symbol_table"
  echo
  echo "Sentence compilation flags"
  echo "  --OOV_symbol          out-of-vocabulary symbol (default: "")"
  echo
  echo "Counting flags:"
  echo "  --epsilon_as_backoff  treat epsilon in the input Fsts as backoff"
  echo "  --order               set maximal order of ngrams to be counted"
  echo
  echo "Smoothing flags:"
  echo "  --bins                no. of bins for katz or absolute discounting"
  echo
  echo "Shrinking flags:"
  echo "  --smooth_method       one of:"
  echo "    \"absolute\", \"seymore\" (default)"
  echo "  --theta               pruning threshold theta"
  echo "  --verbose             show progress"
  echo
  echo "Environment variables:"
  echo "    TMPDIR              working dir for temp results (default:/tmp)"
  exit 1
fi

message() {
  if [[ $VERBOSE = true ]]; then
    echo "[$(date)] $1"
  fi
}

# Moves output to input.
update() {
  rm -fr "${DIR}/input"
  mv "${DIR}/output" "${DIR}/input"
  mkdir "${DIR}/output"
  rm -f "${DIR}/tmp/"*
  UPDATED=true
}

# Creates FST sentences.
compile_sentences() {
  message "Compiling text sentences"
  if [[ -z $SYMBOLS ]]; then
    echo "ERROR: symbol table must be provided to compile sentences"
    exit 1
  fi
  for INF in "${DIR}/input/"*; do
    OUTF="${DIR}/output/$(basename "${INF}")"
    farcompilestrings \
        --fst_type=compact \
        --symbols="${SYMBOLS}" \
        --unknown_symbol="${OOV_SYMBOL}" \
        --keep_symbols \
        "${INF}" \
        "${OUTF}"
  done
  update
  ITYPE=fst_sents
}

# Splits each data shard by context.
ngram_split() {
  message "Splitting count shards by context"
  for INF in "${DIR}/input/"*; do
    OUTF="${DIR}/output/$(basename "${INF}").c"
    "${BIN}/ngramsplit" \
      --method=histogram_split \
      --contexts="${CONTEXTS}" \
      "${INF}" \
      "${OUTF}"
  done
  update
}

# Merges all data shards of the same context.
ngram_merge_counts() {
  message "Merging context shards with the same context"
  while read C ignore; do
    OUTF="${DIR}/output/c${C}"
    "${BIN}/ngrammerge" \
      --check_consistency \
      --complete \
      --round_to_int="${ROUND_TO_INT}" \
      --method=histogram_merge \
      --ofile="${OUTF}" \
      "$DIR/input/d"*".c${C}"
  done < "${DIR}/side/idcontexts"
  update
}

# Splits each context shard by context.
ngram_sub_split() {
  message "Splitting contexts shards by context"
  while read C ignore; do
    INF="${DIR}/input/c${C}"
    OUTF="${DIR}/output/c${C}.s"
    "${BIN}/ngramsplit" \
      --method=histogram_split \
      --complete \
      --contexts="${CONTEXTS}" \
      "${INF}" \
      "${OUTF}"
    # Replaces diagonal with the original context shard.
    rm -f "${DIR}/output/c${C}.s${C}"
    ln "${INF}" "${DIR}/output/c${C}.s${C}"
  done < "${DIR}/side/idcontexts"
  update
}

# Transfers from sub-context shards.
# Argument(s) are additional options to ngramtransfer.
ngram_transfer_to() {
  message "Transferring to sub-context shards from context shards"
  while read S J ignore; do
    OUTF="${DIR}/output/s${S}.c"
    "${BIN}/ngramtransfer" \
      "${@}" \
      --method=histogram_transfer \
      --contexts="${CONTEXTS}" \
      --transfer_from=false \
      --index="${J}" \
      --complete \
      --ofile="${OUTF}" \
      "${DIR}/input/c"*".s${S}"
    # Adds diagonal using the original context shard.
    ln "${DIR}/input/c${S}.s${S}" "${OUTF}${S}"
  done < "${DIR}/side/idcontexts"
  update
}

# Transfers from sub-context shards.
# Argument(s) are additional options to ngramtransfer.
ngram_transfer_from() {
  message "Transferring from sub-context shards to context shards"
  while read C I ignore; do
    OUTF="${DIR}/output/c${C}"
    "${BIN}/ngramtransfer" \
      "${@}" \
      --method=histogram_transfer \
      --contexts="${CONTEXTS}" \
      --transfer_from=true \
      --index="${I}" \
      --complete \
      --ofile="${OUTF}" \
      "${DIR}/input/s"*".c${C}"
  done < "${DIR}/side/idcontexts"
  update
}

# Completes each context shard.
# Argument(s) are additional options to (final) ngramtransfer.
ngram_complete() {
  ngram_sub_split
  ngram_transfer_to
  ngram_transfer_from "${@}"
}

# Counts n-grams.
ngram_count() {
  message "Counting n-grams"
  for INF in "${DIR}/input/"*; do
    OUTF="${DIR}/output/$(basename "${INF}")"
    "${BIN}/ngramcount" \
      --method=histograms \
      --order="${ORDER}" \
      --epsilon_as_backoff="${EPSILON_AS_BACKOFF}" \
      --round_to_int="${ROUND_TO_INT}" \
      "${INF}" \
      "${OUTF}"
  done
  update
  ITYPE=counts
  if [[ -n ${CONTEXTS} ]]; then
    awk \
      '{ printf "%05d %d %s\n", (NR-1), (NR-1), $0 }' \
      "${CONTEXTS}" \
      > "${DIR}/side/idcontexts"
    ngram_split
    ngram_merge_counts
    ngram_complete
  fi
}

# Computes count of counts.
ngram_count_of_counts() {
  message "Computing count of counts"
  while read C I CONTEXT; do
    INF="${DIR}/input/c${C}"
    OUTF="${DIR}/tmp/c${C}"
    "${BIN}/ngramcount" \
      --method=count_of_histograms \
      --context_pattern="${CONTEXT}" \
      "${INF}" \
      "${OUTF}"
  done < "${DIR}/side/idcontexts"
  "${BIN}/ngrammerge" \
    --check_consistency \
    --ofile="${DIR}/side/count_of_counts" \
    --method=count_merge \
    --contexts="${CONTEXTS}" \
    "${DIR}/tmp/c"*
  rm -f "${DIR}/tmp/c"*
}


# Smooths model.
# Argument(s) are additional options to ngrammake.
ngram_make() {
  message "Smoothing model"
  for INF in "${DIR}/input/"*; do
    OUTF="${DIR}/output/$(basename "${INF}")"
    "${BIN}/ngrammake" \
      "$@" \
      --check_consistency \
      --method=katz_frac \
      --bins="${BINS}" \
      "${INF}" \
      "${OUTF}"
  done
  update
  ITYPE=lm
}

# Shrinks model.
ngram_shrink() {
  message "Shrinking model"
  if [[ -n ${CONTEXTS} ]]; then
    while read C I CONTEXT; do
      INF="${DIR}/input/c${C}"
      OUTF="${DIR}/output/c${C}"
      "${BIN}/ngramshrink" \
        --check_consistency \
        --method="${SHRINK_METHOD}" \
        --context_pattern="${CONTEXT}" \
        --theta="${THETA}" \
        "${INF}" \
        "${OUTF}"
    done < "${DIR}/side/idcontexts"
  else
    INF="${DIR}/input/d00000"
    OUTF="${DIR}/output/d00000"
    "${BIN}/ngramshrink" \
      --check_consistency \
      --method="${SHRINK_METHOD}" \
      --theta="${THETA}" \
      "${INF}" \
      "${OUTF}"
  fi
  update
  ITYPE=pruned_lm
}

# Merges contexts into single result.
ngram_merge_contexts() {
  message "Merging context shards"
  case "${OTYPE}" in
    fst_sents|far)
      echo "ERROR: bad output type (${OTYPE}) for merging contexts"
      exit 1 ;;
    counts)
      NORMALIZE="" ;;
    *)
      NORMALIZE="--normalize" ;;
  esac
  OUTF="${DIR}/output/merged"
  "${BIN}/ngrammerge" \
    ${NORMALIZE} \
    --check_consistency \
    --method=context_merge \
    --contexts="${CONTEXTS}" \
    --ofile="${OUTF}" \
    "${DIR}/input/c"*
  update
}

run_pipeline() {
   if [[ ${ITYPE} = text_sents ]]; then
     compile_sentences
   fi
   if [[ ${OTYPE} = fst_sents ]] ; then
     return
   fi
   if [[ ${ITYPE} = fst_sents ]]; then
     ngram_count
   fi
   if [[ ${OTYPE} = counts ]]; then
     return
   fi
   if [[ ${ITYPE} = counts ]]; then
     if [[ -n ${CONTEXTS} ]]; then
       ngram_count_of_counts
       ngram_make --count_of_counts="${DIR}/side/count_of_counts"
     else
       ngram_make
     fi
   fi
   if [[ ${OTYPE} = lm ]]; then
     return
   fi
   if [[ ${ITYPE} = lm ]]; then
     ngram_shrink
   fi
}

mkdir "${DIR}/input" "${DIR}/output" "${DIR}/side" "${DIR}/tmp"

# Copies renamed input data to working directory.
message "Copying data to working directory"
J=0;
for INF in ${IFILE}; do
  I="$(printf "%05d\n" $J)"
  : $((J += 1))
  cp "${INF}" "${DIR}/input/d${I}"
done

if [[ -z $CONTEXTS && ${I} -gt 1 ]]; then
  echo "ERROR: contexts flag must be specified with multiple input files"
  exit 1
fi

# Processes input.
run_pipeline

if [[ $UPDATED = false ]]; then
  echo "ERROR: bad input type (${ITYPE}) for output type (${OTYPE})"
  exit 1
fi

# Returns result.
if [[ -z $CONTEXTS ]]; then
  mv "${DIR}/input/"* "${OFILE}"
elif [[ ${MERGE_CONTEXTS} = true ]]; then
  ngram_merge_contexts
  mv "${DIR}/input/"* "${OFILE}"
else
  for INF in "${DIR}/input/"*; do
    OUTF="${OFILE}.$(basename "${INF}")"
    mv "${INF}" "${OUTF}"
  done
fi

message "Done"
