#!/bin/bash
if [ "$1" = "src" ]; then
    rm -f nmt_model.py nmt_utils.py nmt_vocab.py run_nmt.py
    cp ../../../src/nmt_model.py .
    cp ../../../src/nmt_utils.py .
    cp ../../../src/nmt_vocab.py .
    cp ../../../src/run_nmt.py .

elif [ "$1" = "data" ]; then
    mkdir data
    cp ../../../datasets/en_chr_data/* ./data/.

else
    echo "Invalid option"
fi