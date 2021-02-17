if [ "$1" = "d_train" ]; then
    python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path birth_places_train.tsv    
elif [ "$1" = "d_dev" ]; then
    python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions
elif [ "$1" = "de_test" ]; then
    python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_test_inputs.tsv --outputs_path vanilla.nopretrain.test. predictions
else
    echo "Invalid option"
fi