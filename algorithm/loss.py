'''
Aum Sri Sai Ram
Implementation of Darshan Gera, Vikas G N, and Balasubramanian S.  Handling Ambigu-ous Annotations for Facial Expression Recognition in the Wild. 
In Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP’21), December 19–22, 2021, Jodhpur, India.
https://doi.org/10.1145/3490035.3490289
Authors: Darshan Gera, Vikas G N and Dr. S. Balasubramanian, SSSIHL
Date: 10-10-2021
Email: darshangera@sssihl.edu.in

'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
 

        
def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')        
        
def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        

def dynamic_clean_threshold(co_lambda_max = 0.9, beta = 0.65, epoch_num = 1, max_epochs = 40):
    e = epoch_num
      
    e_r = 0.9 * max_epochs
     
    clean_threshold = co_lambda_max * math.exp(-1.0 * beta * (1.0 - e / e_r ) ** 2) #Dynamic balancing factor using Gaussian like ramp-up 
    return clean_threshold 
    
#Dynamic balancing of Suprevision Loss and Consistency Loss
def Ourloss(y_1, y_2, y_3, t, co_lambda_max = 0.9, beta = 0.65, epoch_num = 1, max_epochs = 40): 
    ''' 
        y_1, y_2, y_3 are predictions of 3 networks and t is target labels. 
    '''
    e = epoch_num
      
    e_r = 0.9 * max_epochs
     
    co_lambda = co_lambda_max * math.exp(-1.0 * beta * (1.0 - e / e_r ) ** 2) #Dynamic balancing factor using Gaussian like ramp-up function
     
    loss_ce_1 = F.cross_entropy(y_1, t) 
    
    loss_ce_2 =  F.cross_entropy(y_2, t)
    
    loss_ce_3 = F.cross_entropy(y_3, t) 
    
    loss_ce =   (1 - co_lambda) * 0.33 * (loss_ce_1 + loss_ce_2 + loss_ce_3)    #Supervision Loss weighted by (1 - dynamic balancing factor)
    
    consistencyLoss =  co_lambda * 0.33 * ( kl_loss_compute(y_1, y_2) +  kl_loss_compute(y_2, y_1)  + kl_loss_compute(y_1, y_3) +  kl_loss_compute(y_3, y_1) + kl_loss_compute(y_3, y_2) +  kl_loss_compute(y_2, y_3))  #Consistency Loss weighted by dynamic balancing factor
     
    loss  =  (consistencyLoss + loss_ce).cpu()
    
    return loss     
    

