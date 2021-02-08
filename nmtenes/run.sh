if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run_nmt_enes.py train --train-src=./data/train.es --train-tgt=./data/train.en --dev-src=./data/dev.es --dev-tgt=./data/dev.en --vocab=./data/vocab.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run_nmt_enes.py decode model.bin ./data/test.es ./data/test.en test_outputs.txt --cuda
else
	echo "Invalid Option Selected"
fi