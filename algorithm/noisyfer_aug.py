'''
Aum Sri Sai Ram
Implementation of Darshan Gera, Vikas G N, and Balasubramanian S. Handling Ambigu-ous Annotations for Facial Expression Recognition in the Wild. 
In Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP’21), December 19–22, 2021, Jodhpur, India.
https://doi.org/10.1145/3490035.3490289
Authors: Darshan Gera, Vikas G N and Dr. S. Balasubramanian, SSSIHL
Date: 10-10-2021
Email: darshangera@sssihl.edu.in

'''

# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from model.cnn import resModel
import numpy as np
from common.utils import accuracy
import os
from algorithm.loss import * 
from algorithm.ema import EMA


class noisyfer:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        
        self.relabel_epochs = args.relabel_epochs       
        self.margin = args.margin
        self.relabled_count = 0
        self.eps = args.eps
        self.warmup_epochs = args.warmup_epochs
        self.alpha = args.alpha
        
        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq        
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        self.co_lambda_max = args.co_lambda_max
        self.beta = args.beta
        self.num_classes  = args.num_classes
        self.max_epochs = args.n_epoch

        if  args.model_type=="res":
            self.model = resModel(args)   
            self.ema_model  = resModel(args)
            

        self.model = self.model.to(device)
        self.ema_model =self.ema_model.to(device)
        
        self.ema = EMA(self.model, alpha=0.99)
        self.ema.apply_shadow(self.ema_model)
        
        filter_list = ['module.classifier.weight', 'module.classifier.bias']
        
        base_parameters_model = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, self.model.named_parameters()))))
        
       
        self.optimizer = torch.optim.Adam([{'params': base_parameters_model}, {'params': list(self.model.module.classifier.parameters()), 'lr': learning_rate}], lr=1e-3)
                 
                                             
        print('\n Initial learning rate is:')
        for param_group in self.optimizer.param_groups:
            print(  param_group['lr'])                              
        
        if args.resume:
           
           pretrained = torch.load(args.resume)
           pretrained_state_dict1 = pretrained['model']   
            
           model_state_dict =  self.model.state_dict()
           loaded_keys = 0
           total_keys = 0
           for key in pretrained_state_dict1:                
               if  ((key=='module.fcx.weight')|(key=='module.fcx.bias')):
                   print(key)
                   pass
               else:    
                   model_state_dict[key] = pretrained_state_dict1[key]
                   total_keys+=1
                   if key in model_state_dict :
                      loaded_keys+=1
           print("Loaded params num:", loaded_keys)
           print("Total params num:", total_keys)
           self.model.load_state_dict(model_state_dict) 
            
           print('Model loaded from ',args.resume)
           
        else:
           print('\n No checkpoint found.\n')         
        
        self.ce_loss = torch.nn.CrossEntropyLoss().to(device)
        
        self.m1_statedict =  self.model.state_dict()
        self.o_statedict = self.optimizer.state_dict()  

        self.adjust_lr = args.adjust_lr
    
    
        
    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model.eval()  
        correct1 = 0
        total1 = 0
        correct  = 0
        with torch.no_grad():
            for images,_, labels, _ in test_loader:
                images = (images).to(self.device)
                logits1 = self.model(images)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()
            
                 
                _, avg_pred = torch.max(outputs1, 1)
                correct += (avg_pred.cpu() == labels).sum()
                
            acc1 = 100 * float(correct1) / float(total1)
           
           
        return acc1
       
    def save_model(self, epoch, acc, noise):
        torch.save({' epoch':  epoch,
                    'model': self.m1_statedict,
                    'optimizer':self.o_statedict,},                          
                     os.path.join('checkpoints/', "epoch_"+str(epoch)+'_noise_'+noise+"_acc_"+str(acc)[:5]+".pth")) 
        print('Models saved '+os.path.join('checkpoints/', "epoch_"+str(epoch)+'_noise_'+noise+"_acc_"+str(acc)[:5]+".pth")) 
    
    
    
               
    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model.train() 
        
        
        if epoch > 0:
           self.adjust_learning_rate(self.optimizer, epoch)
        
        train_total = 0
        train_correct = 0
        
        pure_ratio_1_list = []
        pure_ratio_2_list = []
        
        threshold_clean = dynamic_clean_threshold(co_lambda_max = self.co_lambda_max, beta = self.beta , epoch_num = epoch, max_epochs = self.n_epoch)
        
        print('\n thredhold_clean ', threshold_clean)
        if epoch < self.warmup_epochs:
            print('\n Warm up stage using supervision loss based on clean samples')
        elif epoch == self.warmup_epochs:
            print('\n Robust learning stage using consistency loss and supervision loss with psudeo labeling')
        
        for i, (images1, images2, labels, indexes) in enumerate(train_loader):
        
            ind = indexes.cpu().numpy().transpose()
        
            if i > self.num_iter_per_epoch:
                break
                
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            labels = labels.to(self.device)
            # Forward + Backward + Optimize
            
            logits1 = self.model(images1)
            logits2 = self.model(images2)
            
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            
            N, C = logits1.shape
            #given_labels = torch.full(size=(N, C), fill_value=0.0).to(self.device)
            #given_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1.0)
            given_labels = torch.full(size=(N, C), fill_value=self.eps/(C - 1)).to(self.device)
            given_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-self.eps)
            #print('Labels',given_labels)
            #print(labels[0], given_labels[0])
            #assert False
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1
            
            with torch.no_grad():
                logits1_ema = self.ema_model(images1)
                logits2_ema = self.ema_model(images2)
                soft_labels = (F.softmax(logits1_ema, dim=1) + F.softmax(logits2_ema, dim=1)) / 2
                prob_clean = 1 - js_div(probs1, given_labels)
                #print('prob_clean ',js_div(probs1, given_labels), prob_clean ) 
            
            if epoch < self.warmup_epochs:               
               losses = cross_entropy(logits1,given_labels, reduction='none') + cross_entropy(logits2, given_labels, reduction='none')
               loss = losses[prob_clean >= threshold_clean].mean()
               #loss = self.ce_loss(logits1,labels) + self.ce_loss(logits2, labels)
            else:
               target_labels = given_labels.clone()
               # clean samples
               idx_clean = (prob_clean >= threshold_clean).nonzero().squeeze(dim=1)
               """
               _, preds1 = probs1.topk(1, 1, True, True)
               _, preds2 = probs2.topk(1, 1, True, True)
                
               agree = (preds1 == preds2).squeeze(dim=1)
               unclean = (prob_clean < threshold_clean)                
               idx_id = (agree * unclean).nonzero().squeeze(dim=1)
               target_labels[idx_id] = soft_labels[idx_id]
               """
               # classification loss
               losses = cross_entropy(logits1, target_labels, reduction='none') + cross_entropy(logits2, target_labels, reduction='none')
               loss_c =  losses[idx_clean].mean()
               #loss_c = losses.mean()

               # consistency loss
               losses_o = symmetric_kl_div(probs1, probs2) 
               loss_o = losses_o.mean()

               # final loss
               loss = (1 - self.alpha) * loss_c + loss_o * self.alpha
               
               """
               #relabelling  
               
               _, preds1 = probs1.topk(1, 1, True, True)
               _, preds2 = probs2.topk(1, 1, True, True)
                
               agree = (preds1 == preds2).squeeze(dim=1)
               #print('agree: ',agree.cpu().numpy())
               #threshold_clean = 0.9
               #print('prob_clean, threshold_clean ',prob_clean.cpu().numpy(),threshold_clean)
               unclean = (prob_clean < threshold_clean)
               #print('unclean: ',unclean.cpu().numpy())                
               update_idx = (agree * unclean).nonzero().squeeze(dim=1)
               label_idx = indexes[update_idx] # get samples' index in train_loader
               
               #print('label_idx: ',label_idx.cpu().numpy())
               #print('update_idx: ',update_idx.cpu().numpy())
               
               if update_idx.numel()> 0:
                  _, relabels = torch.max(soft_labels[update_idx],1)
                  #print('relabel: ',relabels.cpu().numpy().shape)
                  #print('label_idx: ',label_idx.cpu().numpy().shape)
                  #print('update_idx: ',update_idx.shape)
                  
                  all_labels = np.array(train_loader.dataset.label)
                  all_labels[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
                  train_loader.dataset.label = all_labels
                  #print('relabelled: ',update_idx.numel())
                  self.relabled_count = self.relabled_count + update_idx.numel()
              
               #else:
                  #print('No relabelling index found')
               """   
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            self.ema.update_params(self.model)
            self.ema.apply_shadow(self.ema_model)

            
            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, loss.data.item() ))

        train_acc1 = float(train_correct) / float(train_total)
        #print('\n Epoch and relabelled ', epoch, self.relabled_count)
        #self.relabled_count = 0
        return train_acc1
    
    def adjust_learning_rate(self, optimizer, epoch):
        print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           print(param_group['lr'])              
        print('******************************')
    

    
    
    
    
    def adjust_learning_rate(self, optimizer, epoch):
        #print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           #print(param_group['lr'])              
        #print('******************************')
    
