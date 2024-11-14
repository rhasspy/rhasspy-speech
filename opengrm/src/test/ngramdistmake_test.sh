#!/bin/bash
# Tests distributed training of language models.

source "${srcdir}/disttestsetup.sh" || exit

distributed_test --otype=lm --smooth_method=katz
distributed_test --otype=lm --smooth_method=absolute
