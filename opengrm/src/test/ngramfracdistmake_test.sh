#!/bin/bash
# Tests using fractional count representations with distributed training.

FRAC_DIST=true;
source "${srcdir}/disttestsetup.sh" || exit

distributed_test --otype=lm
