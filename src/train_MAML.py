# import learn2learn as l2l
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
from torchvision import transforms
from model import LightningCLIP
import learn2learn as l2l
from data import MultitaskFinetuneDataset
from pathlib import Path
import os
import json
import copy
import argparse

def construct_dataset(
    dataset_type = 'ClevrCounting', 
    using_dataset = ['counting'], 
    dataset_splited = False,
    N = 5, 
    K = 10, 
    split = [0.6,0.2,0.2]
    ):
    
     '''dataset roots'''
    if dataset_type == "ClevrCounting":
        print('[INFO] using Clevr dataset')
        dataset_root = '../data/ClevrCounting/dataset'
        pickeled_path = '../data/ClevrCounting/pickeled/tensor_dict.pt'
        using_datasets = using_dataset
        template_function = lambda x: f'An image of {x} objects'
        label_to_text = None
    
    elif dataset_type == 'ABO': # 50/30/30
        print(f'[INFO] using ABO {using_dataset} dataset')
        dataset_splited = True
        train_datasets_root = '../data/ABO/dataset/train'
        dev_datasets_root = '../data/ABO/dataset/val'
        test_datasets_root = '../data/ABO/dataset/test'
        
        train_pickeled_path = '../data/ABO/picked/train_tensor_dict.pt'
        dev_pickeled_path = '../data/ABO/picked/dev_tensor_dict.pt'
        test_pickeled_path = '../data/ABO/picked/test_tensor_dict.pt'
        
        label_to_text_path = '../data/ABO/label_to_text.json'
        label_to_text = json.load(open(label_to_text_path))
        template_function = None
        using_datasets = using_dataset
    
    elif dataset_type == 'CUB':
        print('[INFO] using CUB dataset')
        dataset_root = '../data/CUB/dataset'
        pickeled_path = '../data/CUB/pickled/tensor_dict.pt'
        using_datasets = using_dataset
        template_function = lambda x: f'A photo of {x}'
        label_to_text = None
    
    elif dataset_type == 'Fungi':
        print('[INFO] using Fungi dataset')
        dataset_root = '../data/Fungi/dataset'
        pickeled_path = '../data/Fungi/pickled/tensor_dict.pt'
        using_datasets = using_dataset
        template_function = lambda x: f'A photo of {x}'
        label_to_text = None

    elif dataset_type == 'mini':
        print('[INFO] using mini_imagenet dataset')
        dataset_root = '../data/mini_imagenet/dataset'
        pickeled_path = '../data/mini_imagenet/pickled/tensor_dict.pt'
        using_datasets = using_dataset
        template_function = lambda x: f'A photo of {x}'
        label_to_text = None
    
    # add an elif here for your own dataset
    
    
    ''' processor '''
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    ''' data augmentation '''
    # if 'color' not in using_datasets:
    #
    print('[Note]: not using color jitter augmentation')
    train_augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale = (0.9,1.0), ratio = (1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # transforms.RandomAffine(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    '''set up datasets'''
    
    if dataset_splited:
        print('[INFO] using splited data')
        train_phase = 'all_train'
        dev_phase = 'all_dev'
        test_phase = 'all_test'
    else:
        print(f'[INFO] split data according to {split}')
        train_phase = 'train'
        dev_phase = 'dev'
        test_phase = 'test'
        train_datasets_root = dataset_root
        dev_datasets_root = dataset_root
        test_datasets_root = dataset_root
        train_pickeled_path = pickeled_path
        dev_pickeled_path = pickeled_path
        test_pickeled_path = pickeled_path
    
    # training split, training phase
    train_support_dataset = MultitaskFinetuneDataset(
        train_datasets_root, 
        using_datasets, 
        processor, 
        phase = train_phase, # ['train','dev','test','all_train','all_dev','all_test']
        meta_phase = 'meta_train', # 'meta_test'
        mode = 'multitask', # ['multitask','singletask']
        N = N, # a scaler N 
        num_tasks = K, 
        task_idx = None, # task index k in [0,num_tasks)
        use_pickled = train_pickeled_path, 
        template_function = template_function,
        label_to_text = label_to_text,
        transforms = train_augmentation,
        split = split
    )

    # dev split, meta_test phase
    train_query_dataset = copy.deepcopy(train_support_dataset)
    train_query_dataset.meta_phase = 'meta_train_query'
    train_query_dataset.construct_input_samples_all()
    
    # dev split, meta_train phase
    dev_support_dataset = MultitaskFinetuneDataset(
        dev_datasets_root, 
        using_datasets, 
        processor, 
        phase = dev_phase, # ['train','dev','test','all_train','all_dev','all_test']
        meta_phase = 'meta_train', # 'meta_test'
        mode = 'multitask', # ['multitask','singletask']
        N = N, # a scaler N 
        num_tasks = K, 
        task_idx = None, # task index k in [0,num_tasks)
        use_pickled = dev_pickeled_path, 
        template_function = template_function,
        label_to_text = label_to_text,
        transforms = train_augmentation,
        split = split
    )
    # dev split, meta_test phase
    dev_query_dataset = copy.deepcopy(dev_support_dataset)
    dev_query_dataset.meta_phase = 'meta_test'
    dev_query_dataset.construct_input_samples_all()

    # test split, meta_train phase
    test_support_dataset = MultitaskFinetuneDataset(
        test_datasets_root, 
        using_datasets, 
        processor, 
        phase = test_phase, # ['train','dev','test','all_train','all_dev','all_test']
        meta_phase = 'meta_train', # 'meta_test'
        mode = 'multitask', # ['multitask','singletask']
        N = N, # a scaler N 
        num_tasks = K, 
        task_idx = None, # task index k in [0,num_tasks)
        use_pickled = test_pickeled_path, 
        template_function = template_function,
        label_to_text = label_to_text,
        transforms = train_augmentation,
        split = split
    )
    # test split, meta_test phase
    test_query_dataset = copy.deepcopy(test_support_dataset)
    test_query_dataset.meta_phase = 'meta_test'
    test_query_dataset.construct_input_samples_all()

    stat = {'N':N,'K':K,'dataset_type':dataset_type,'using_datasets':using_datasets}
    return train_support_dataset, train_query_dataset, dev_support_dataset, dev_query_dataset, test_support_dataset, test_query_dataset, stat

def main(
        train_support_dataset, 
        train_query_dataset, 
        dev_support_dataset, 
        dev_query_dataset, 
        test_support_dataset, 
        test_query_dataset,
        stat,
        features = None,
        first_order = False,
        allow_unused = None,
        allow_nograd = False,
        loss = None,
        training_batch_size = 32,
        finetuning_batch_size = 8,
        training_epoch = 8,
        finetuning_epoch = 5,
        lr = 1e-6,
        adaptation_lr = 1e-5,
        optimizer_choice = 'Adam',
        adaptation_epoch = 1,
        weight_decay = 0.0,
        train_log_root = './log/MAML_train_log',
        test_log_root = './log/MAML_test_log',
        output_note = ''
    ):

    N = stat['N']
    K = stat['K']
    using_dataset = stat['using_datasets']
    dataset_type = stat['dataset_type']

    ''' set up model '''
    model = LightningCLIP(
        features=features,
        loss=loss,
        lr=lr,
        scheduler_decay = 1.0,
        scheduler_step = 20,
        weight_decay = weight_decay,
        optimizer_choice = 'Adam'
    )
    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)
    
    # create output dir
    if len(using_dataset) == 1:
        dataset_name_string = using_dataset[0]
    else:
        dataset_name_string = '.'.join(using_dataset)

    if first_order:
        output_dir_name = f'CLIP_FOMAML_{dataset_type}_{dataset_name_string}_N-{N}_K-{K}_lr-{lr}_adalr-{adaptation_lr}_trainB-{training_batch_size}_finetuneB-{finetuning_batch_size}_trainEp-{training_epoch}_finetuneEp-{finetuning_epoch}_adaptEp-{adaptation_epoch}_{output_note}'
    else:
        output_dir_name = f'CLIP_MAML_{dataset_type}_{dataset_name_string}_N-{N}_K-{K}_lr-{lr}_adalr-{adaptation_lr}_trainB-{training_batch_size}_finetuneB-{finetuning_batch_size}_trainEp-{training_epoch}_finetuneEp-{finetuning_epoch}_adaptEp-{adaptation_epoch}_{output_note}'
    output_dir_path = os.path.join(train_log_root, output_dir_name)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    

    '''set up model'''
    model = l2l.algorithms.MAML(model, lr=adaptation_lr, first_order=first_order, allow_nograd=allow_nograd,allow_unused=allow_unused)
    model.to(device)
    
    ''' set up and optimizer '''
    if optimizer_choice == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay = weight_decay)
    elif optimizer_choice == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay = weight_decay)

    print('>>> starting step 1')
    for k in range(K):
        assert train_support_dataset.mode == 'multitask'
        assert train_query_dataset.mode == 'multitask'
        train_support_dataset.task_idx = k
        train_support_dataloader_k = DataLoader(train_support_dataset, batch_size = training_batch_size, shuffle=True, num_workers=4)
        
        train_query_dataset.task_idx = k
        train_query_dataloader_k = DataLoader(train_query_dataset, batch_size = training_batch_size, shuffle=True, num_workers=4)
        
        for ep in range(training_epoch):
            ### adaptation
            clone = model.clone()
            for batch in train_support_dataloader_k:
                for key,value in batch.items():
                    batch[key] = value.to(device)
                for step in range(adaptation_epoch):
                    error,_,_ = clone.forward(batch)
                    clone.adapt(error)
        
            ### update_initialization
            running_acc = 0.0
            running_loss = 0.0
            for batch in train_query_dataloader_k:
                for key,value in batch.items():
                    batch[key] = value.to(device)
                optimizer.zero_grad()
                error,acc,f1_p_r = clone.forward(batch)
                error.backward()
                optimizer.step()

                running_acc += acc
                running_loss += error.item()
            task_loss = running_loss/len(train_query_dataloader_k)
            task_acc = running_acc/len(train_query_dataloader_k)
            print(f'[LOG] training task:{k} epoch:{ep} loss={task_loss} acc={task_acc}')

    # save step one 
    print('>>> saving model from step 1')
    PATH = os.path.join(output_dir_path, "step1.ckpt")
    torch.save(model.state_dict(), PATH)
    
    '''step 2: finetuning and testing on sampled tasks'''
    print('>>> starting step 2')
    model.load_state_dict(torch.load(PATH))
    model.to(device)

    result_dict = {'dev':{'task_accs':[],'task_loss':[]}, 'test':{'task_accs':[],'task_loss':[]}}
    # on dev set
    for k in range(K):
        assert dev_support_dataset.mode == 'multitask'
        assert dev_query_dataset.mode == 'multitask'
        dev_support_dataset.task_idx = k
        dev_support_dataloader_k = DataLoader(dev_support_dataset, batch_size = finetuning_batch_size, shuffle=True, num_workers=4)

        dev_query_dataset.task_idx = k
        dev_query_dataloader_k = DataLoader(dev_query_dataset, batch_size = 1, shuffle=False, num_workers=4)

        for ep in range(finetuning_epoch):
            ### adaptation
            clone = model.clone()
            for batch in dev_support_dataloader_k:
                for key,value in batch.items():
                    batch[key] = value.to(device)
                for step in range(adaptation_epoch):
                    error,_,_ = clone.forward(batch)
                    clone.adapt(error)
        
        ### update_initialization
        running_acc = 0.0
        running_loss = 0.0
        for batch in dev_query_dataloader_k:
            for key,value in batch.items():
                batch[key] = value.to(device)
            error,acc,f1_p_r = clone.validate(batch)

            running_acc += acc
            running_loss += error.item()

        task_loss = running_loss/len(dev_query_dataloader_k)
        task_acc = running_acc/len(dev_query_dataloader_k)
        print(f'[LOG] dev task:{k} loss={task_loss} acc={task_acc}')

        # log results
        result_dict['dev']['task_accs'].append(task_acc)
        result_dict['dev']['task_loss'].append(task_loss)
    result_dict['dev']['avg_acc'] = sum(result_dict['dev']['task_accs'])/len(result_dict['dev']['task_accs'])

    # reload model from step1
    model.load_state_dict(torch.load(PATH))
    model.to(device)

    # on test set
    for k in range(K):
        assert test_support_dataset.mode == 'multitask'
        assert test_query_dataset.mode == 'multitask'
        test_support_dataset.task_idx = k
        test_support_dataloader_k = DataLoader(test_support_dataset, batch_size = finetuning_batch_size, shuffle=True, num_workers=4)
    
        test_query_dataset.task_idx = k
        test_query_dataloader_k = DataLoader(test_query_dataset, batch_size = 1, shuffle=False, num_workers=4)
        
        ### adaptation
        for ep in range(finetuning_epoch):
            ### adaptation
            clone = model.clone()
            for batch in test_support_dataloader_k:
                for key,value in batch.items():
                    batch[key] = value.to(device)
                for step in range(adaptation_epoch):
                    error,_,_ = clone.forward(batch)
                    clone.adapt(error)
        
        ### update_initialization
        running_acc = 0.0
        running_loss = 0.0
        for batch in test_query_dataloader_k:
            for key,value in batch.items():
                batch[key] = value.to(device)
            error,acc,f1_p_r = clone.validate(batch)

            running_acc += acc
            running_loss += error.item()

        task_loss = running_loss/len(test_query_dataloader_k)
        task_acc = running_acc/len(test_query_dataloader_k)
        print(f'[LOG] test task:{k} loss={task_loss} acc={task_acc}')

        # log results
        result_dict['test']['task_accs'].append(task_acc)
        result_dict['test']['task_loss'].append(task_loss)
    result_dict['test']['avg_acc'] = sum(result_dict['test']['task_accs'])/len(result_dict['test']['task_accs'])

    # write result dict
    model_log_path = os.path.join(output_dir_path,f'dev_test_result.json')
    with open(model_log_path,'w') as f:
        json.dump(result_dict,f,indent=4)

def freeze_features(model, require_grad_layers = ['text_head','vision_head']):
    for name, param in model.named_parameters():
        param.requires_grad = False
        for required in require_grad_layers:
            if required in name:
                param.requires_grad = True
                break

def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

def get_n_k_pair(nrange, N):
    import numpy as np
    """
        nrange: np.array containing ints
    """
    # N is the total number of classes. n is the n-way classification, k is the task number. missing prob is the bound of probability we allow that one of the classes is not covered
    missing_prob = 0.001
    k = lambda n, N: np.rint(np.log(missing_prob)/np.log(1 - n/N)).astype(int)
    n_k_pair = [(nrange[i], k(nrange, N)[i]) for i in range(len(nrange))]
    # print(f'n, k pairs are: {n_k_pair}')
    return n_k_pair

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose dataset')
    # dataset type: can choose from existing datasets: ['ClevrCounting','ABO','CUB','Fungi','mini']
    parser.add_argument('-i', '--dataset_type', required=True) 
    # subdataset: can choose from existing subdatasets for each dataset type: 
        # 'ClevrCounting': 'counting'
        # 'ABO': 'material','color'
        # 'CUB': 'classification'
        # 'Fungi': 'classification'
        # 'mini': 'classification'
    parser.add_argument('-d', '--using_dataset', required=True) # subdataset, e.g. 'counting' (one dataset can have multiple subdataset, such as ABO)
    parser.add_argument('-N', '--class_num', required=True) # total number of classes
    parser.add_argument('-trlg', '--train_log_dir', required=False, default='./log/multitask_classical_zeroshot') # where to store checkpoint and dev/test results
    parser.add_argument('-trep', '--train_epoch', required=False, default=8) # number of epochs in training phase
    parser.add_argument('-ftep', '--finetune_epoch', required=False, default=5) # number of finetuning epochs during meta-testing
    parser.add_argument('-tb', '--train_batch', required=False, default=32) # training batch size
    parser.add_argument('-fb', '--finetune_batch', required=False, default=8) # finetuning batch size during meta-testing
    parser.add_argument('-itr', '--num_runs', required=False, default=5) # number of runs with different random seed
    args = vars(parser.parse_args())
    
    dataset_type = args['dataset_type']
    using_dataset = [args['using_dataset']]
    N = int(args['class_num'])
    train_log_dir = args['train_log_dir']
    num_itr = int(args['num_runs'])

    # starting class number
    start_n = 2
    if dataset_type == 'Fungi':
        start_n = 10
    print(f'start with n: {start_n}')
    
    # epoch
    tr_ep = int(args['train_epoch']) # default 8
    ft_ep = int(args['finetune_epoch']) # default 5
    tb = int(args['train_batch']) # in this paper, when using "CUB", we set default tb=16 
    fb = int(args['finetune_batch'])  
    print('training ep:',tr_ep)
    print('finetuning ep:',ft_ep)
    print('training batch:',tb)
    print('finetuning batch:',fb)

    '''set random seeds'''
    seed_val = 311
    for i in range(num_itr):
        seed_val += 1
        pl.seed_everything(seed_val, workers=True)
        for nn,kk in get_n_k_pair(np.arange(start_n,N), N=N):
            train_support_dataset, train_query_dataset, dev_support_dataset, dev_query_dataset, test_support_dataset, test_query_dataset, stat \
                = construct_dataset(dataset_type=dataset_type, using_dataset = using_dataset, dataset_splited = False, N = nn, K = kk, split = [0.6,0.2,0.2])
            main(
                train_support_dataset, 
                train_query_dataset, 
                dev_support_dataset, 
                dev_query_dataset, 
                test_support_dataset, 
                test_query_dataset,
                stat,
                features = None,
                first_order = True,
                allow_unused = None,
                allow_nograd = False,
                loss = None,
                training_batch_size = tb,
                finetuning_batch_size = fb,
                training_epoch = tr_ep,
                finetuning_epoch = ft_ep, # better than 1
                lr = 1e-6,
                adaptation_lr = 1e-7, # better than 1e-6, 1e-5
                optimizer_choice = 'Adam',
                adaptation_epoch = 1, # better than 5
                weight_decay = 0.0,
                train_log_root = train_log_dir,
                output_note = f'seed-{seed_val}'
            )
