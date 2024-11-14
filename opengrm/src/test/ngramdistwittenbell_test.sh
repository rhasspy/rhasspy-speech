#!/bin/bash
# Tests distributed training of Witten-Bell language models, requiring transfer.

source "${srcdir}/disttestsetup.sh" || exit

distributed_test \
    --otype=pruned_lm \
    --smooth_method=witten_bell \
    --shrink_method=relative_entropy \
    --theta=.00015
