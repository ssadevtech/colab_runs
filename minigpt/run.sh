if [ "$1" = "d_train" ]; then
    python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path birth_places_train.tsv    
elif [ "$1" = "d_dev" ]; then
    python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions
elif [ "$1" = "d_test" ]; then
    python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test.predictions
elif [ "$1" = "d_london" ]; then
    python src/london_baseline.py --eval_corpus_path birth_dev.tsv --city London
elif [ "$1" = "f_pretrain" ]; then
    python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.pretrain.params
elif [ "$1" = "f_finetune" ]; then
    python src/run.py finetune vanilla wiki.txt --reading_params_path vanilla.pretrain.params --writing_params_path vanilla.finetune.params --finetune_corpus_path birth_places_train.tsv
elif [ "$1" = "f_dev" ]; then
    python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.pretrain.dev.predictions    
elif [ "$1" = "f_test" ]; then
    python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.finetune.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.pretrain.test.predictions
else
    echo "Invalid option"
fi