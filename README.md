## Downloading our datasets
- https://drive.google.com/file/d/1CfomsX6qmdCLfFutptqrQnp1RlaJEpXh/view?usp=sharing
- extract and put the `/data` folder under the same root as `/src`

## Dataset structure
- Each dataset may have several subdatasets (most of them only have one) 
```
|<dataset_root>
    |dataset/
        -|<subdataset_1>
            -|<label_1>
            -|<label_2>
        -|<subdataset_2>
            ...
    |pickled/
        -|tensor_dict.pt
```
- The pickle file `tensor_dict.pt` has the following format:
```
{
    'subdataset_1':{
        'label_1':{
            'image_tensors':np.array((N,3,224,224)), # N: image number
            'input_ids':np.array(S), # S: token length of the filled template text
            'attention_masks':np.array(S),
            'template_input_ids':np.array(S_), # S_: token length of the un-filled template text
            'template_attention_masks':np.array(S_),
        },
        'label_2':{
            ...
        }
    },
    ...
}
```
- ABO dataset contains an additional `label_to_text.json` file, which provides text template for each subdataset and label.
### A list of available datasets and subdatasets
Dataset | dataset name (-i) | subdataset name (-d)
--- | --- | ---
Clevr Counting | `ClevrCounting` | `counting`
Amazon Berkeley Objects (ABO) |`ABO`| `material`,`color`
Caltech-UCSD Birds 200 (CUB)| `CUB`| `classification`
Fungi | `Fungi`| `classification`
Mini-imagenet | `mini` | `classification`


## Training with provided datasets
`run.sh` provided example code for performing training and meta-testing on our datasets. 
### Output format
Each model checkpoint dir contains two files:
- `step1.ckpt`: model checkpoint after training phase
- `dev_test_results.json`: scores on each task configuration on dev and test set during meta-testing
### Loading checkpoint
- Here is an example snippet for loading `step1.ckpt` from **multitask-finetuning/classical-finetuning/zeroshot** models:
```
    model = MultitaskFinetuneCLIP()
    model = model.load_from_checkpoint(checkpoint_path="<path to log dir>/step1.ckpt")
```
- Here is an example snippet for loading `step1.ckpt` from **fomaml** models:
```
    model = LightningCLIP()
    model = l2l.algorithms.MAML(model, lr=1e-5 first_order=True)
    model.load_state_dict(torch.load("<path to log dir>/step1.ckpt"))
```

## Training with custom datasets
### preprocess dataset
- put your new dataset in the same format as provided dataset into `data/`
- Specify `template_function` or the path to `label_to_text` json file (an example file can be found in `/data/ABO/label_to_text.json`) at line `350` and `355` in `data.py`
- `preprocess.sh` provides an example of running `data.py` to create pickle file for your new dataset
- add your dataset into `construct_dataset()`: line `77` in `train.py` and line `80` in `train_MAML.py`
### train
- modify `run.sh` to train and meta-test on your own dataset
- refer to `train.py` and `train_MAML.py` for default and tuning hyperparameters for each algorithm

## Citation


