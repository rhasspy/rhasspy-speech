#!/bin/bash
# Tests using fractional count representations with distributed shrinking.

FRAC_DIST=true;
source "${srcdir}/disttestsetup.sh" || exit

distributed_test \
  --otype=pruned_lm \
  --shrink_method=relative_entropy \
  --theta=.00015
distributed_test --otype=pruned_lm --shrink_method=seymore --theta=4
