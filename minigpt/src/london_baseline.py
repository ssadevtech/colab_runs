# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils
import argparse

argp = argparse.ArgumentParser()
argp.add_argument('--city',
    help="City to always guess", default='London')
argp.add_argument('--eval_corpus_path',
    help="Path of the corpus to evaluate on", default=None)
args = argp.parse_args()
    
total = correct = 0
num_lines = sum(1 for line in open(args.eval_corpus_path,'r'))
predictions = [args.city]*num_lines

total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
if total > 0:
    print(f'Correct: {correct} out of {total}: {correct/total*100}%')