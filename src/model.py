import torch
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPFeatureExtractor
import requests
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import pytorch_lightning as pl
import learn2learn as l2l
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, precision_recall_fscore_support
import numpy as np

def flat_accuracy(preds, labels):
    # preds: numpy array: N * C
    # labels: numpy array: N 
    pred_flat = np.argmax(preds, axis=1).flatten()  
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat), pred_flat, labels_flat

class MultitaskFinetuneCLIP(pl.LightningModule):
    def __init__(
        self,
        features=None,
        text_head=None,
        vision_head=None,
        loss=None,
        lr=1e-5,
        scheduler_decay = 1.0,
        scheduler_step = 20,
        weight_decay = 0.0,
        output_dim = 512,
        use_original_CLIP = False,
        optimizer_choice = 'Adam'
    ):
        super(MultitaskFinetuneCLIP, self).__init__()
        if features is None:
            features = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        '''use randomly init head'''
        if text_head is None:
            text_head = nn.Linear(512, output_dim)
            nn.init.kaiming_normal_(text_head.weight)
        if vision_head is None:
            vision_head = nn.Linear(768, output_dim)
            nn.init.kaiming_normal_(vision_head.weight)
        if loss is None:
            loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
        self.loss = loss
        self.softmax = nn.LogSoftmax(dim=1)
        self.features = features
        self.text_head = text_head
        self.vision_head = vision_head
        self.lr = lr
        self.scheduler_decay = scheduler_decay,
        self.scheduler_step = scheduler_step,
        self.weight_decay = weight_decay
        self.use_original_CLIP = use_original_CLIP
        self.save_hyperparameters({
            'lr':self.lr,
            "scheduler_decay":self.scheduler_decay,
            "scheduler_step":self.scheduler_step,
            "weight_decay":self.weight_decay,
            "output_dim":output_dim,
            "use_original_CLIP":self.use_original_CLIP
        })
        self.optimizer_choice = optimizer_choice # SGD

    def init_heads(self):
        nn.init.kaiming_normal_(self.text_head.weight)
        nn.init.kaiming_normal_(self.vision_head.weight)
    
    def encode(self, input_ids_input, attention_masks_input, pixel_values_batch):
        output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
        image_embeds_hidden, text_embeds_hidden = output.vision_model_output[1], output.text_model_output[1]
        image_embeds, text_embeds = self.vision_head(image_embeds_hidden), self.text_head(text_embeds_hidden)
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return image_embeds, text_embeds

    def forward(self, batch):
        input_ids_batch, attention_masks_batch, pixel_values_batch, labels = batch['input_ids'],batch['attention_masks'], batch['pixel_values'],batch['label_idx']

        labels = labels.cpu().detach().numpy()
        # print('labels:', labels)

        input_ids_dict = {}
        attention_masks_dict = {}
        unique_labels = [] 
        for i in range(len(labels)):
            l = labels[i]
            if l not in unique_labels:
                unique_labels.append(l)
                input_ids_dict[l] = input_ids_batch[i]
                attention_masks_dict[l] = attention_masks_batch[i]
        unique_labels = sorted(unique_labels)

        # print(unique_labels)
        label_idx_to_gt_idx = {}
        for i in range(len(unique_labels)):
            label_idx_to_gt_idx[unique_labels[i]] = i
        input_ids = [input_ids_dict[l] for l in unique_labels]
        attention_masks = [attention_masks_dict[l] for l in unique_labels]
        # print(label_idx_to_gt_idx)

        input_ids_input = torch.stack(input_ids)
        attention_masks_input = torch.stack(attention_masks)
        assert len(label_idx_to_gt_idx) == input_ids_input.size()[0] 
        assert attention_masks_input.size() == input_ids_input.size() 
        # construct groud truth
        gt_vector = np.zeros(len(pixel_values_batch))
        for i in range(len(pixel_values_batch)):
            gt_vector[i] = label_idx_to_gt_idx[labels[i]]
        gt_vector = torch.from_numpy(gt_vector)
        gt = gt_vector.long()

        # print(input_ids_input.size())
        # print(attention_masks_input.size())
        # print(pixel_values_batch.size())
        # print(gt)

        if self.use_original_CLIP:
            output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
            image_embeds, text_embeds = output.image_embeds, output.text_embeds
        else:
            output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
            image_embeds_hidden, text_embeds_hidden = output.vision_model_output[1], output.text_model_output[1]
            image_embeds, text_embeds = self.vision_head(image_embeds_hidden), self.text_head(text_embeds_hidden)
            # normalized features
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # gt to device
        device_idx = image_embeds.get_device()
        device = torch.device(f'cuda:{device_idx}')
        gt = gt.to(device)
        
        logit_scale = self.features.logit_scale.exp()  
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T # (0.5*(shots*ways), ways)
        assert logits_per_image.size()[0] == gt.size()[0]
        error = self.loss(logits_per_image,gt)
        acc, pred_flat, labels_flat = flat_accuracy(logits_per_image.cpu().detach().numpy(), gt.cpu().detach().numpy())
        f1_p_r = precision_recall_fscore_support(pred_flat, labels_flat, average='macro', zero_division=0)
        return error, acc, f1_p_r

    @torch.enable_grad()
    def validate(self, batch):
        input_ids_batch, attention_masks_batch, pixel_values_batch, label_gt_idx = batch['input_ids'],batch['attention_masks'], batch['pixel_values'],batch['label_gt_idx']
        assert input_ids_batch.size()[0] == 1
        input_ids_input = input_ids_batch[0]
        attention_masks_input = attention_masks_batch[0]
        pixel_values_batch = pixel_values_batch
        gt = label_gt_idx
        
        if self.use_original_CLIP:
            output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
            image_embeds, text_embeds = output.image_embeds, output.text_embeds
        else:
            output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
            image_embeds_hidden, text_embeds_hidden = output.vision_model_output[1], output.text_model_output[1]
            image_embeds, text_embeds = self.vision_head(image_embeds_hidden), self.text_head(text_embeds_hidden)
            # normalized features
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        logit_scale = self.features.logit_scale.exp()  
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T # (0.5*(shots*ways), ways)
        # assert logits_per_image.size()[0] == gt.size()
        # print(logits_per_image.size(), gt.size())
        # print(text_embeds.size(), image_embeds.size())
        # print(logits_per_text.size(), logits_per_image.size())
        # print(logits_per_image, gt)

        error = self.loss(logits_per_image,gt)
        acc, pred_flat, labels_flat = flat_accuracy(logits_per_image.cpu().detach().numpy(), gt.cpu().detach().numpy())
        f1_p_r = precision_recall_fscore_support(pred_flat, labels_flat, average='macro', zero_division=0)

        return error, acc, f1_p_r

    def configure_optimizers(self):
        # print('optim weight decay:',self.weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        print(f'[INFO] setting lr to {self.optimizer_choice}:',self.lr)
        if self.optimizer_choice == 'Adam':
            print('[OPTIM] using Adam')
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay = self.weight_decay)
        elif self.optimizer_choice == 'SGD':
            print('[OPTIM] using SGD')
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay = self.weight_decay)

        # lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=self.scheduler_step,
        #     gamma=self.scheduler_decay,
        # )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        train_loss, train_accuracy, f1_p_r = self.forward(train_batch)
        self.log(
            "train_loss",
            train_loss.item(),
            on_step=False,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        valid_loss, valid_accuracy, f1_p_r = self.validate(val_batch)
        self.log(
            "valid_loss",
            valid_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_accuracy",
            valid_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return valid_loss.item()

    def test_step(self, test_batch, batch_idx):
        test_loss, test_accuracy, f1_p_r = self.validate(test_batch)
        self.log(
            "test_loss",
            test_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_accuracy",
            test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_precision",
            f1_p_r[0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_recall",
            f1_p_r[1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_f1",
            f1_p_r[2],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return test_loss.item()

class LightningCLIP(pl.LightningModule):
    def __init__(
        self,
        features=None,
        loss=None,
        lr=1e-6,
        scheduler_decay = 1.0,
        scheduler_step = 20,
        weight_decay = 0.0,
        optimizer_choice = 'Adam'
    ):
        super(LightningCLIP, self).__init__()
        if features is None:
            features = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        '''use randomly init head'''
        if loss is None:
            loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
        self.loss = loss
        self.softmax = nn.LogSoftmax(dim=1)
        self.lr = lr
        self.features = features
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.weight_decay = weight_decay
        self.save_hyperparameters({
            'lr':self.lr,
            "scheduler_decay":self.scheduler_decay,
            "scheduler_step":self.scheduler_step,
            "weight_decay":self.weight_decay,
        })
        self.optimizer_choice = optimizer_choice # SGD
    
    def encode(self, input_ids_input, attention_masks_input, pixel_values_batch):
        output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
        return output.image_embeds, output.text_embeds
    
    def forward(self, batch):    
        input_ids_batch, attention_masks_batch, pixel_values_batch, labels = batch['input_ids'],batch['attention_masks'], batch['pixel_values'],batch['label_idx']

        labels = labels.cpu().detach().numpy()
        # print('labels:', labels)

        input_ids_dict = {}
        attention_masks_dict = {}
        unique_labels = [] 
        for i in range(len(labels)):
            l = labels[i]
            if l not in unique_labels:
                unique_labels.append(l)
                input_ids_dict[l] = input_ids_batch[i]
                attention_masks_dict[l] = attention_masks_batch[i]
        unique_labels = sorted(unique_labels)

        # print(unique_labels)
        label_idx_to_gt_idx = {}
        for i in range(len(unique_labels)):
            label_idx_to_gt_idx[unique_labels[i]] = i
        input_ids = [input_ids_dict[l] for l in unique_labels]
        attention_masks = [attention_masks_dict[l] for l in unique_labels]
        # print(label_idx_to_gt_idx)

        input_ids_input = torch.stack(input_ids)
        attention_masks_input = torch.stack(attention_masks)
        assert len(label_idx_to_gt_idx) == input_ids_input.size()[0] 
        assert attention_masks_input.size() == input_ids_input.size() 
        # construct groud truth
        gt_vector = np.zeros(len(pixel_values_batch))
        for i in range(len(pixel_values_batch)):
            gt_vector[i] = label_idx_to_gt_idx[labels[i]]
        gt_vector = torch.from_numpy(gt_vector)
        gt = gt_vector.long()

        output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
        logit_scale = self.features.logit_scale.exp()  

        image_embeds, text_embeds = output.image_embeds, output.text_embeds
    
        # gt to device
        device_idx = image_embeds.get_device()
        device = torch.device(f'cuda:{device_idx}')
        gt = gt.to(device)
        
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T # (0.5*(shots*ways), ways)
        assert logits_per_image.size()[0] == gt.size()[0]
        error = self.loss(logits_per_image,gt)
        acc, pred_flat, labels_flat = flat_accuracy(logits_per_image.cpu().detach().numpy(), gt.cpu().detach().numpy())
        f1_p_r = precision_recall_fscore_support(pred_flat, labels_flat, average='macro', zero_division=0)
        return error, acc, f1_p_r

    @torch.enable_grad()
    def validate(self, batch):
        input_ids_batch, attention_masks_batch, pixel_values_batch, label_gt_idx = batch['input_ids'],batch['attention_masks'], batch['pixel_values'],batch['label_gt_idx']
        assert input_ids_batch.size()[0] == 1
        input_ids_input = input_ids_batch[0]
        attention_masks_input = attention_masks_batch[0]
        pixel_values_batch = pixel_values_batch
        gt = label_gt_idx
        
        output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
        image_embeds, text_embeds = output.image_embeds, output.text_embeds
        
        logit_scale = self.features.logit_scale.exp()  
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T # (0.5*(shots*ways), ways)

        error = self.loss(logits_per_image,gt)
        acc, pred_flat, labels_flat = flat_accuracy(logits_per_image.cpu().detach().numpy(), gt.cpu().detach().numpy())
        f1_p_r = precision_recall_fscore_support(pred_flat, labels_flat, average='macro', zero_division=0)

        return error, acc, f1_p_r

    def configure_optimizers(self):
        # print('optim weight decay:',self.weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        print(f'[INFO] setting lr to {self.optimizer_choice}:',self.lr)
        if self.optimizer_choice == 'Adam':
            print('[OPTIM] using Adam')
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay = self.weight_decay)
        elif self.optimizer_choice == 'SGD':
            print('[OPTIM] using SGD')
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay = self.weight_decay)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        train_loss, train_accuracy, _ = self.forward(val_batch)

        self.log(
            "train_loss",
            train_loss.item(),
            on_step=False,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        valid_loss, valid_accuracy, f1_p_r = self.validate(val_batch)
        self.log(
            "valid_loss",
            valid_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_accuracy",
            valid_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return valid_loss.item()

    def test_step(self, test_batch, batch_idx):
        test_loss, test_accuracy, f1_p_r = self.validate(test_batch)
        self.log(
            "test_loss",
            test_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_accuracy",
            test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_precision",
            f1_p_r[0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_recall",
            f1_p_r[1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_f1",
            f1_p_r[2],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return test_loss.item()

class LightningANIL_CLIP(pl.LightningModule):
    def __init__(
        self,
        features=None,
        text_head = None,
        vision_head = None,
        text_adaptation_lr = 1e-7,
        vision_adaptation_lr = 1e-7,
        loss=None,
        lr=1e-6,
        scheduler_decay = 1.0,
        scheduler_step = 20,
        weight_decay = 0.0,
        optimizer_choice = 'Adam',
        output_dim = 512
    ):
        super(LightningANIL_CLIP, self).__init__()
        if features is None:
            features = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        '''use randomly init head'''
        if text_head is None:        
            text_head = nn.Linear(512, output_dim)
            # nn.init.kaiming_uniform_(text_head.weight)
            nn.init.kaiming_normal_(text_head.weight)
            # nn.init.xavier_uniform_(text_head.weight)
        if vision_head is None:
            vision_head = nn.Linear(768, output_dim)
            # nn.init.kaiming_uniform_(vision_head.weight)
            nn.init.kaiming_normal_(vision_head.weight)
            # nn.init.xavier_uniform_(vision_head.weight)
        
        if loss is None:
            loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
        self.loss = loss
        self.softmax = nn.LogSoftmax(dim=1)
        self.lr = lr
        self.features = features
        # self.text_head = l2l.algorithms.MAML(text_head, lr=text_adaptation_lr)
        # self.vision_head = l2l.algorithms.MAML(vision_head, lr=vision_adaptation_lr)
        self.text_head = text_head
        self.vision_head = vision_head
        self.text_adaptation_lr = text_adaptation_lr
        self.vision_adaptation_lr = vision_adaptation_lr
        self.scheduler_decay = scheduler_decay
        self.scheduler_step = scheduler_step
        self.weight_decay = weight_decay
        self.save_hyperparameters({
            'lr':self.lr,
            "scheduler_decay":self.scheduler_decay,
            "scheduler_step":self.scheduler_step,
            "weight_decay":self.weight_decay,
        })
        self.optimizer_choice = optimizer_choice # SGD

    def init_heads(self):
        text_head = nn.Linear(512, 512)
        # nn.init.kaiming_uniform_(text_head.weight)
        nn.init.kaiming_normal_(text_head.weight)
        # nn.init.xavier_uniform_(text_head.weight)
        vision_head = nn.Linear(768, 512)
        # nn.init.kaiming_uniform_(vision_head.weight)
        nn.init.kaiming_normal_(vision_head.weight)
        # nn.init.xavier_uniform_(vision_head.weight)
        self.text_head = text_head
        self.vision_head = vision_head
 
    def forward(self, batch):    
        input_ids_batch, attention_masks_batch, pixel_values_batch, labels = batch['input_ids'],batch['attention_masks'], batch['pixel_values'],batch['label_idx']

        labels = labels.cpu().detach().numpy()
        # print('labels:', labels)

        input_ids_dict = {}
        attention_masks_dict = {}
        unique_labels = [] 
        for i in range(len(labels)):
            l = labels[i]
            if l not in unique_labels:
                unique_labels.append(l)
                input_ids_dict[l] = input_ids_batch[i]
                attention_masks_dict[l] = attention_masks_batch[i]
        unique_labels = sorted(unique_labels)

        # print(unique_labels)
        label_idx_to_gt_idx = {}
        for i in range(len(unique_labels)):
            label_idx_to_gt_idx[unique_labels[i]] = i
        input_ids = [input_ids_dict[l] for l in unique_labels]
        attention_masks = [attention_masks_dict[l] for l in unique_labels]
        # print(label_idx_to_gt_idx)

        input_ids_input = torch.stack(input_ids)
        attention_masks_input = torch.stack(attention_masks)
        assert len(label_idx_to_gt_idx) == input_ids_input.size()[0] 
        assert attention_masks_input.size() == input_ids_input.size() 
        # construct groud truth
        gt_vector = np.zeros(len(pixel_values_batch))
        for i in range(len(pixel_values_batch)):
            gt_vector[i] = label_idx_to_gt_idx[labels[i]]
        gt_vector = torch.from_numpy(gt_vector)
        gt = gt_vector.long()

        output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
        logit_scale = self.features.logit_scale.exp()  

        image_embeds_tmp, text_embeds_tmp = output.vision_model_output[1], output.text_model_output[1]
        image_embeds = self.vision_head(image_embeds_tmp)
        text_embeds = self.text_head(text_embeds_tmp)
        image_embeds = image_embeds/image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds/text_embeds.norm(dim=-1, keepdim=True)
        
        # gt to device
        device_idx = image_embeds.get_device()
        device = torch.device(f'cuda:{device_idx}')
        gt = gt.to(device)
        
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T # (0.5*(shots*ways), ways)
        assert logits_per_image.size()[0] == gt.size()[0]
        error = self.loss(logits_per_image,gt)
        acc, pred_flat, labels_flat = flat_accuracy(logits_per_image.cpu().detach().numpy(), gt.cpu().detach().numpy())
        f1_p_r = precision_recall_fscore_support(pred_flat, labels_flat, average='macro', zero_division=0)
        
        return error, acc, f1_p_r
        
    @torch.enable_grad()
    def validate(self, batch):
        input_ids_batch, attention_masks_batch, pixel_values_batch, label_gt_idx = batch['input_ids'],batch['attention_masks'], batch['pixel_values'],batch['label_gt_idx']
        assert input_ids_batch.size()[0] == 1
        input_ids_input = input_ids_batch[0]
        attention_masks_input = attention_masks_batch[0]
        pixel_values_batch = pixel_values_batch
        gt = label_gt_idx
        
        output = self.features(input_ids=input_ids_input,attention_mask=attention_masks_input,pixel_values=pixel_values_batch,return_dict=True)
        image_embeds_tmp, text_embeds_tmp = output.vision_model_output[1], output.text_model_output[1]
        image_embeds =  self.vision_head(image_embeds_tmp)
        text_embeds =  self.text_head(text_embeds_tmp)
        image_embeds = image_embeds/image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds/text_embeds.norm(dim=-1, keepdim=True)
        
        logit_scale = self.features.logit_scale.exp()  
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T # (0.5*(shots*ways), ways)

        error = self.loss(logits_per_image,gt)
        acc, pred_flat, labels_flat = flat_accuracy(logits_per_image.cpu().detach().numpy(), gt.cpu().detach().numpy())
        f1_p_r = precision_recall_fscore_support(pred_flat, labels_flat, average='macro', zero_division=0)

        return error, acc, f1_p_r

    def configure_optimizers(self):
        # print('optim weight decay:',self.weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        print(f'[INFO] setting lr to {self.optimizer_choice}:',self.lr)
        if self.optimizer_choice == 'Adam':
            print('[OPTIM] using Adam')
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay = self.weight_decay)
        elif self.optimizer_choice == 'SGD':
            print('[OPTIM] using SGD')
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, weight_decay = self.weight_decay)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        train_loss, train_accuracy, _ = self.forward(val_batch)

        self.log(
            "train_loss",
            train_loss.item(),
            on_step=False,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        valid_loss, valid_accuracy, f1_p_r = self.validate(val_batch)
        self.log(
            "valid_loss",
            valid_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "valid_accuracy",
            valid_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return valid_loss.item()

    def test_step(self, test_batch, batch_idx):
        test_loss, test_accuracy, f1_p_r = self.validate(test_batch)
        self.log(
            "test_loss",
            test_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "test_accuracy",
            test_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_precision",
            f1_p_r[0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_recall",
            f1_p_r[1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_f1",
            f1_p_r[2],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return test_loss.item()

def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
