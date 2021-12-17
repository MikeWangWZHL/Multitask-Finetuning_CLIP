## an example of performing training and meta-testing
## a full list of avaible dataset and subdatasets can be found in README.md 

# run multitask-finetuning & classical-finetuning & zeroshot on ClevrCounting for 5 runs with different random seeds
python3 train.py -i ClevrCounting -d counting -N 10 -itr 5

# run FOMAML on ClevrCounting for 5 runs with different random seeds
python3 train_MAML.py -i ClevrCounting -d counting -N 10 -itr 5