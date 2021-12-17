import torch
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
import requests
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch_lightning as pl
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from collections import defaultdict
import shutil
from glob import glob
from torchvision import transforms
import copy
''' set random seeds '''
# seed_val = 312
# pl.seed_everything(seed_val, workers=True)

class MultitaskFinetuneDataset(Dataset):
    def __init__(
        self, 
        dataset_root, 
        using_datasets, 
        processor, 
        phase, # ['train','dev','test','all_train','all_dev','all_test']
        meta_phase, # 'meta_train', 'meta_test', 'none' 
        mode, # ['multitask','singletask']
        N, # a scaler N 
        num_tasks, 
        task_idx = None, # task index k in [0,num_tasks)
        use_pickled = None, 
        template_function=None,
        label_to_text = None,
        transforms = None,
        split = [0.6,0.2,0.2]
    ):
        if template_function is None and label_to_text is None:
            print('Need to input template function or input label to text dict for each label in each dataset!')
            quit()

        if label_to_text:
            print('label_to_text keys:')
            for dataset_name in label_to_text.keys():
                print('\t',dataset_name, list(label_to_text[dataset_name].keys()))
        
        self.dataset_root = dataset_root
        self.using_datasets = using_datasets
        self.processor = processor # e.g., CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.transforms = transforms
        self.mode = mode # ['multitask','singletask']
        self.phase = phase
        self.meta_phase = meta_phase
        self.num_tasks = num_tasks
        self.task_idx = task_idx
        self.N = N
        # global label to idx dict
        self.label_to_idx = {} # key is (dataset_name, label_name)

        if use_pickled:
            self.tensor_dict = torch.load(use_pickled)
        else:
            self.tensor_dict = prcess_image_text(dataset_root, using_datasets, processor, output_dir=None, template_function=template_function, label_to_text=label_to_text)
            
        if phase == 'train':
            self.idx_start_ratio = 0
            self.idx_end_ratio = split[0]
        elif phase == 'dev':
            self.idx_start_ratio = split[0]
            self.idx_end_ratio = split[0] + split[1]
        elif phase == 'test':
            self.idx_start_ratio = split[0] + split[1]
            self.idx_end_ratio = 1.0
        elif phase in ['all_train','all_dev','all_test']:
            self.idx_start_ratio = 0.0
            self.idx_end_ratio = 1.0

        """sampled tasks for each dataset(list of [list of labels])"""
        self.tasks = defaultdict(list) 

        for dataset_name in self.using_datasets: 
            dataset_path = os.path.join(dataset_root, dataset_name) # e.g., counting
            labels = [os.path.basename(p) for p in glob(os.path.join(dataset_path,'*'))]
            print('dataset:',dataset_name,'|', 'labels',labels)
            
            # assign global label idx
            for label in labels:
                assert (dataset_name,label) not in self.label_to_idx
                self.label_to_idx[(dataset_name,label)] = len(self.label_to_idx)

            # sample num_tasks subset of labels, each has N labels
            ## tasks
            if self.mode == 'multitask':
                for k in range(self.num_tasks):
                    chosen_labels = random.sample(labels, k=N)
                    self.tasks[dataset_name].append(chosen_labels)
            elif self.mode == 'singletask':
                self.tasks[dataset_name].append(labels)
        
        # call construct_input_samples_all when switching from meta_train to meta_test
        self.construct_input_samples_all()

        print(f'datasets: {self.using_datasets}')
        print(f'num of tasks K={self.num_tasks}')
        print(f'sampled tasks {self.tasks}')
        print(f'init task index k={self.task_idx}')
        print(f'sample subset lenght N={self.N}')
        print('====================================================================================')
    
    def construct_input_samples_all(self):
        input_samples_list = []
        if self.mode == 'singletask':
            input_samples_list.append(self.construct_input_samples_k(0))
            assert len(input_samples_list) == 1
        else:
            for k in range(self.num_tasks):
                input_samples_list.append(self.construct_input_samples_k(k))
            assert len(input_samples_list) == self.num_tasks
        self.input_samples_list = input_samples_list

    def construct_input_samples_k(self, k):
        # print(f'construct input samples for mode={self.mode}, k={self.task_idx}')
        input_samples = []
        for dataset_name in self.using_datasets:
            tasks = self.tasks[dataset_name]
            if self.mode == 'singletask':
                assert len(tasks) == 1
                task = tasks[0]
                print(f'---------------------------')
                print(f'construct input samples for mode={self.mode}, meta_phase={self.meta_phase}, task: ({dataset_name},{task})')
            elif self.mode == 'multitask':
                print(f'---------------------------')
                print(f'construct input samples for mode={self.mode}, meta_phase={self.meta_phase}, task: ({dataset_name},{tasks[k]})')
                task = tasks[k]
            

            
            # mapping label to gt idx for this dataset_name
            label_to_gt_idx = {}
            sorted_global_idx = [(l,self.label_to_idx[(dataset_name,l)]) for l in task]
            sorted_global_idx = sorted(sorted_global_idx, key = lambda x:x[1])
            # print(sorted_global_idx)
            for i in range(len(sorted_global_idx)):
                item = sorted_global_idx[i]
                label_to_gt_idx[item[0]] = i
            
            ### used for meta_test 
            all_label_input_ids = [self.tensor_dict[dataset_name][l[0]]['input_ids'][0] for l in sorted_global_idx]
            all_label_attention_masks = [self.tensor_dict[dataset_name][l[0]]['attention_masks'][0] for l in sorted_global_idx]
            all_label_template_input_ids = [self.tensor_dict[dataset_name][l[0]]['template_input_ids'][0] for l in sorted_global_idx]
            all_label_template_attention_masks = [self.tensor_dict[dataset_name][l[0]]['template_attention_masks'][0] for l in sorted_global_idx]

            all_label_input_ids = torch.stack(all_label_input_ids)
            all_label_attention_masks = torch.stack(all_label_attention_masks)
            all_label_template_input_ids = torch.stack(all_label_template_input_ids)
            all_label_template_attention_masks = torch.stack(all_label_template_attention_masks)
            ###

            for item in sorted_global_idx:
                label = item[0]
                label_id = item[1]
                num_img = len(self.tensor_dict[dataset_name][label]['image_tensors'])
                starting_idx = int(self.idx_start_ratio * num_img)
                ending_idx = int(self.idx_end_ratio * num_img)

                image_tensor_list = self.tensor_dict[dataset_name][label]['image_tensors'][starting_idx:ending_idx]
                # random.Random(seed_val).shuffle(image_tensor_list)

                input_ids = self.tensor_dict[dataset_name][label]['input_ids'][0]
                attention_masks = self.tensor_dict[dataset_name][label]['attention_masks'][0]
                template_input_ids = self.tensor_dict[dataset_name][label]['template_input_ids'][0]
                template_attention_masks = self.tensor_dict[dataset_name][label]['template_attention_masks'][0]

                if self.meta_phase == 'meta_train':
                    image_tensor_list = image_tensor_list[:int(0.5*len(image_tensor_list))]
                elif self.meta_phase in ['meta_test','meta_train_query']:
                    image_tensor_list = image_tensor_list[int(0.5*len(image_tensor_list)):]

                for img_tensor in image_tensor_list:
                    # do data augmentation
                    if self.transforms:
                        img_tensor = self.transforms(img_tensor)
                    if self.meta_phase in ['meta_train','none','meta_train_query']:
                        input_sample = {
                            'input_ids':input_ids,
                            'attention_masks':attention_masks,
                            'template_input_ids':template_input_ids,
                            'template_attention_masks':template_attention_masks,
                            'pixel_values':img_tensor,
                            'label_idx':torch.tensor(label_id)
                        }
                    elif self.meta_phase == 'meta_test':
                        input_sample = {
                            'input_ids':all_label_input_ids,
                            'attention_masks':all_label_attention_masks,
                            'template_input_ids':all_label_template_input_ids,
                            'template_attention_masks':all_label_template_attention_masks,
                            'pixel_values':img_tensor,
                            'label_gt_idx':torch.tensor(label_to_gt_idx[label]).long()
                        }
                    input_samples.append(input_sample)
        print(f'num of samples for k={k}:', len(input_samples))
        return input_samples


    def __len__(self):
        if self.mode == 'multitask':
            if self.task_idx is None:
                print('ERROR: please assign task_idx')
                quit()
            else:
                self.input_samples = self.input_samples_list[self.task_idx]
        else:
            self.input_samples = self.input_samples_list[0]

        return len(self.input_samples)

    def __getitem__(self, idx):
        if self.mode == 'multitask':
            if self.task_idx is None:
                print('ERROR: please assign task_idx')
                quit()
            else:
                self.input_samples = self.input_samples_list[self.task_idx]
        elif self.mode == 'singletask':
            self.input_samples = self.input_samples_list[0]
        # print(f'get item: {idx}-{self.task_idx}')
        
        return self.input_samples[idx]
            
    def visualize_batch(self, idx, vis_dir = '../vis'):
        shutil.rmtree(vis_dir)
        batch = self.meta_tasks[idx]
        dataset_name = batch['dataset']
        print('img_num per batch:',len(batch['image_idx']))
        for i in range(self.ways):
            l = batch['label'][i]
            label_directory = os.path.join(vis_dir,f'{i}')
            image_paths = sorted(glob(os.path.join(self.dataset_root, f'{dataset_name}/{l}/*')))

            if not os.path.exists(label_directory):
                os.makedirs(label_directory)
            else:
                shutil.rmtree(label_directory)
                os.makedirs(label_directory)

            for j in range(i*2*self.shots,(i+1)*2*self.shots):
                # print(j)
                label,img_idx = batch['image_idx'][j]
                img_file_name = f'{i}-label-{l}-{img_idx}.jpg'
                shutil.copyfile(image_paths[img_idx],os.path.join(label_directory,img_file_name))

################################################
def get_text_input_default(label):
    return f'An image of {label} objects'

def prcess_image_text(dataset_root, using_datasets, processor, output_dir = None, output_name = None, template_function = None, label_to_text=None):

    def find_max_text_token_length(dataset_root,using_datasets, processor, template_function, label_to_text):
        all_labels = []
        all_text = []
        for dataset_name in using_datasets: 
            dataset_path = os.path.join(dataset_root, dataset_name) # e.g., counting
            labels = sorted([os.path.basename(p) for p in glob(os.path.join(dataset_path,'*'))]) # e.g., 3
            if label_to_text is not None:
                all_text += [label_to_text[dataset_name][l] for l in labels]
            else:
                all_text += [template_function(l) for l in labels]
            all_labels += labels
        print(all_text)
        tmp = processor(text=all_text, images=None, return_tensors="pt", padding=True)
        label_tmp = processor(text=all_labels, images=None, return_tensors="pt", padding=True)

        return tmp['input_ids'][0].size()[0], label_tmp['input_ids'][0].size()[0]
    
    if template_function is None:
        template_function = get_text_input_default

    tensor_dict = defaultdict(dict)
    # find the max padding length for all label text
    max_padding_length, max_padding_length_label = find_max_text_token_length(dataset_root,using_datasets,processor,template_function, label_to_text)
    print('max padding length:', max_padding_length, 'max padding length label:', max_padding_length_label)
    for dataset_name in using_datasets: 
        dataset_path = os.path.join(dataset_root, dataset_name) # e.g., counting
        labels = sorted([os.path.basename(p) for p in glob(os.path.join(dataset_path,'*'))]) # e.g., 3
        print(dataset_name,labels)
        for label in tqdm(labels):
            image_paths = sorted(glob(os.path.join(dataset_root, f'{dataset_name}/{label}/*')))
            if len(image_paths) == 0:
                continue
            # print(image_paths)
            images = [Image.open(img) for img in image_paths]
            # convert RGBA to RGB
            images = [im.convert('RGB') for im in images]
            if label_to_text is not None:
                text = [label_to_text[dataset_name][label]]
                template_text = label_to_text[dataset_name]['template_text']
                label_text = [label]
            else:
                text = [template_function(label)]
                template_text = [template_function('')]
                label_text = [label]

            inputs = processor(text=text, images=images, return_tensors="pt", padding='max_length', max_length=max_padding_length)
            # add template text:
            template_text_inputs = processor(text=template_text, images=None, return_tensors="pt", padding='max_length', max_length=max_padding_length)
            # add label text:
            print(label_text)
            label_text_inputs = processor(text=label_text, images=None, return_tensors="pt", padding='max_length', max_length=max_padding_length_label)

            tensor_dict[dataset_name][label] = {
                'input_ids':inputs['input_ids'],
                'template_input_ids':template_text_inputs['input_ids'],
                'label_input_ids':label_text_inputs['input_ids'],
                'attention_masks':inputs['attention_mask'],
                'template_attention_masks':template_text_inputs['attention_mask'],
                'label_attention_masks':label_text_inputs['attention_mask'],
                'image_tensors':inputs['pixel_values']
            }

    if output_dir is not None:
        if output_name is None:
            output_name = 'tensor_dict.pt'
        torch.save(tensor_dict, os.path.join(output_dir,output_name))
        print('done saving...')

    return tensor_dict

if __name__ == '__main__':
    '''preprocess your custom dataset into pytorch pickle file'''
    parser = argparse.ArgumentParser(description='choose dataset')
    parser.add_argument('-i', '--dataset_root', required=True) # e.g. /data/ClevrCounting/dataset/
    parser.add_argument('-ii', '--subdatasets', required=True) # use ',' to seperate multiple subdatasets: e.g. color,material
    parser.add_argument('-o', '--output_dir', required=True) # e.g. /data/ClevrCounting/pickled/
    args = vars(parser.parse_args())

    dataset_root = args.dataset_root
    subdatasets = args.subdatasets.split(',')  
    output_dir = args.output_dir
    output_name = 'tensor_dict.pt'

    # change the following line if you need to use a different image&text processer
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # change the following line to use your own template function for all tasks
    template_function = lambda x: f"A photo of a {x}"

    # change the following line to use your own template functions for each task and label defined in a json file
    label_to_text = None 

    process_image_text(dataset_root, subdataset, processor, output_dir=output_dir, output_name=output_name, template_function=template_function, label_to_text=label_to_text)
