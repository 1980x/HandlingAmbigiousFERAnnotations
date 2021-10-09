'''
Aum Sri Sai Ram
Implementation of Darshan Gera, Vikas G N, and Balasubramanian S. 2021. Handling Ambigu-ous Annotations for Facial Expression Recognition in the Wild. 
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
from algorithm.loss import Ourloss 


class noisyfer:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        
        self.relabel_epochs = 0       
        self.margin = args.margin
        self.relabled_count = 0
        
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
            

        self.model.to(device)
        
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
        
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)
        
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
            for images, labels, _ in test_loader:
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
    
    
    
               
    
    
    def adjust_learning_rate(self, optimizer, epoch):
        print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           print(param_group['lr'])              
        print('******************************')
    

    
    
    
               
    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model.train() 
        
        
        if epoch > 0:
           self.adjust_learning_rate(self.optimizer, epoch)
        
        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        train_total3 = 0
        train_correct3 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
        
            if i > self.num_iter_per_epoch:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            # Forward + Backward + Optimize
            
            logits1 = self.model(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            
             
            avg_prec = accuracy((logits1), labels, topk=(1,))
            
            loss = self.loss_fn(logits1, labels)
            

            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            
            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, loss.data.item() ))
                    
        train_acc = float(train_correct) / float(train_total)
        
        
        if epoch >= self.relabel_epochs:
                
                sm1 = torch.softmax(logits1 , dim = 1)
                Pmax1, predicted_labels1 = torch.max(sm1, 1) # predictions
                
                Pgt1 = torch.gather(sm1, 1, labels.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
                
                true_or_false = (Pmax1 - Pgt1 > self.margin) 
                update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                label_idx = indexes[update_idx] # get samples' index in train_loader
                relabels = predicted_labels1[update_idx] # predictions where (Pmax - Pgt > margin_2)
                #print('relabel: ',relabels.cpu().numpy())
                #print('label_idx: ',label_idx.cpu().numpy())
                #print('update_idx: ',update_idx)
                if update_idx.numel()> 0:
                   all_labels = np.array(train_loader.dataset.label)
                   all_labels[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
                   train_loader.dataset.label = all_labels
                   #print('relabelled: ',update_idx.numel())
                   self.relabled_count = self.relabled_count + update_idx.numel()
                
                #else:
                   #print('No relabelling index found')
        
        return train_acc, self.relabled_count
    
    def adjust_learning_rate(self, optimizer, epoch):
        #print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           #print(param_group['lr'])              
        #print('******************************')
    
