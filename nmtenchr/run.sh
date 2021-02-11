
if [ "$1" = "train" ]; then
    CUDA_VISIBLE_DEVICES=0 python run_nmt.py train --train-src=./data/train.chr --train-tgt=./data/train.en --dev-src=./data/dev.chr --dev-tgt=./data/dev.en --vocab=./data/vocab.json --embed-size=1024 --hidden-size=1024 --tok=spm --spmmodel-src=./data/src.model --spmmodel-tgt=./data/tgt.model --beam-size=10 --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=0.3 --cuda
elif [ "$1" = "test" ]; then
    CUDA_VISIBLE_DEVICES=0 python run_nmt.py decode model.bin ./data/test.chr ./data/test.en outputs.txt --cuda
else
    echo "Invalid option"
fi
