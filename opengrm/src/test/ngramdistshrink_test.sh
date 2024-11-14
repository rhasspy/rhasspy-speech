#!/bin/bash
# Tests distributed training and pruning of language models.

source "${srcdir}/disttestsetup.sh" || exit

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
