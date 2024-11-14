#!/bin/bash
# Tests distributed counting of language models.

source "${srcdir}/disttestsetup.sh" || exit

distributed_test --otype=counts
