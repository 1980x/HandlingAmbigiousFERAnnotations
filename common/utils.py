'''
Aum Sri Sai Ram
Implementation of Darshan Gera, Vikas G N, and Balasubramanian S. Handling Ambigu-ous Annotations for Facial Expression Recognition in the Wild. 
In Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP’21), December 19–22, 2021, Jodhpur, India.
https://doi.org/10.1145/3490035.3490289
Authors: Darshan Gera, Vikas G N and Dr. S. Balasubramanian, SSSIHL
Date: 10-10-2021
Email: darshangera@sssihl.edu.in

'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]
    
    
def to_one_hot(inp, num_classes, device):
    """
    creates a one hot encoding that is a representation of categorical variables as binary vectors for the given label.
    Args:
        inp: label of a sample.
        num_classes: the number of labels or classes that we have in the multi class classification task.
    Returns:
        one hot encoding vector of the specific target.
    """
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return y_onehot.to(device)    
