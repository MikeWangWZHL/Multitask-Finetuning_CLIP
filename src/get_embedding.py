import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
from torchvision import transforms
from model import MultitaskFinetuneCLIP, LightningCLIP
import learn2learn as l2l
# from data import MultitaskFinetuneDataset
from pathlib import Path
import os
import json
import copy
import argparse


def main_zeroshot(pickeled_dataset, using_dataset, output_dir, output_name = 'zeroshot_embedding.pt'):
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)
    
    # set up model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    model.to(device)

    for key,value in pickeled_dataset[using_dataset].items():

        pixel_values_batch = value['image_tensors'].to(device)
        input_ids_input = value['input_ids'].to(device)
        attention_masks_input = value['attention_masks'].to(device)
        output = model(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
        image_embeds, text_embeds = output.image_embeds, output.text_embeds
        value['zeroshot_image_embedding'] = image_embeds.cpu().detach()
        print(f'done: {key}','embedding size:',image_embeds.shape)

    torch.save(pickeled_dataset, os.path.join(output_dir,output_name))

def main_finetuned(pickeled_dataset, using_dataset, checkpoint, output_dir, output_name = 'our_embedding.pt', if_MAML = False):
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    device = torch.device(dev)
    
    # set up model
    if if_MAML:
        model = LightningCLIP(
            features=None,
            loss=None,
            lr=1e-6,
            scheduler_decay = 1.0,
            scheduler_step = 20,
            weight_decay = 0.0,
            optimizer_choice = 'Adam'
        )
        model = l2l.algorithms.MAML(model, lr=1e-5, first_order=True, allow_nograd=False,allow_unused=None)
        model.load_state_dict(torch.load(checkpoint))  
    else:
        model = MultitaskFinetuneCLIP(
            features=None, #default: CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            text_head=None, 
            vision_head=None,
            loss=None,
            lr=1e-6,
            weight_decay = 0.0,
            output_dim = 512,
            use_original_CLIP = False,
            optimizer_choice = 'Adam'
        )
        model = model.load_from_checkpoint(checkpoint_path=checkpoint)
    
    model.to(device)

    for key,value in pickeled_dataset[using_dataset].items():

        pixel_values_batch = value['image_tensors'].to(device)
        input_ids_input = value['input_ids'].to(device)
        attention_masks_input = value['attention_masks'].to(device)
        image_embeds, text_embeds = model.encode(input_ids_input,attention_masks_input,pixel_values_batch)
        value['zeroshot_image_embedding'] = image_embeds.cpu().detach()
        print(f'done: {key}','embedding size:',image_embeds.shape)

    torch.save(pickeled_dataset, os.path.join(output_dir,output_name))



if __name__ == '__main__':
    """
        example code for getting image embedding from zeroshot/finetuned models
    """

    pickeled_dataset = torch.load('../data/ClevrCounting/pickeled/tensor_dict.pt')

    '''zero shot embedding Counting'''
    using_dataset = 'counting'
    output_dir = './visualization/embedding'

    main_zeroshot(pickeled_dataset, using_dataset, output_dir, output_name = 'zeroshot_embedding.pt')
    
    '''finetuned embedding Counting '''
    multitask_checkpoint = './log/multitask_classical_zeroshot/<multitask-finetuned model name>/step1.ckpt'
    fomaml_checkpoint = './log/fomaml/<fomaml model name>/step1.ckpt'

    main_finetuned(pickeled_dataset, using_dataset, multitask_checkpoint, output_dir, output_name = 'multitask_finetuned_embedding.pt', if_MAML = False)
    main_finetuned(pickeled_dataset, using_dataset, fomaml_checkpoint, output_dir, output_name = 'fomaml_finetuned_embedding.pt', if_MAML = True)