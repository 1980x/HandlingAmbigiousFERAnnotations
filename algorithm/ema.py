
'''
Aum Sri Sai Ram
Implementation of Darshan Gera, Vikas G N, and Balasubramanian S. 2021. Handling Ambigu-ous Annotations for Facial Expression Recognition in the Wild. 
In Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP’21), December 19–22, 2021, Jodhpur, India.
https://doi.org/10.1145/3490035.3490289
Authors: Darshan Gera, Vikas G N and Dr. S. Balasubramanian, SSSIHL
Date: 10-10-2021
Email: darshangera@sssihl.edu.in

Code borrowed from : # https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/utils/ema.py
'''

class EMA(object):
    """
    Usage:
        model = ResNet(config)
        ema = EMA(model, alpha=0.999)
        ... # train an epoch
        ema.update_params(model)
        ema.apply_shadow(model)
    """
    def __init__(self, model, alpha=0.999):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]
        self.alpha = alpha

    def init_params(self, model):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]

    def update_params(self, model):
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(self.alpha * self.shadow[name] + (1 - self.alpha) * state[name])

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow, strict=True)
