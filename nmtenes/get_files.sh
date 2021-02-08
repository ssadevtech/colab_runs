#!/bin/bash
if [ "$1" = "src" ]; then
    rm -f nmt_model.py nmt_utils.py nmt_vocab.py run_nmt_enes.py
    cp ../../../src/{nmt_model,nmt_utils,nmt_vocab,run_nmt_enes}* .

elif [ "$1" = "data" ]; then
    mkdir data
    cp ../../../datasets/en_es_data/{dev,test,train}* ./data/.

else
    echo "Invalid option"
fi