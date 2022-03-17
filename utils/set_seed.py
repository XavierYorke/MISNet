#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/03/17 15:14:20
@Author  :   XavierYorke
@Contact :   mzlxavier1230@gmail.com
'''


import torch
import random
import numpy as np
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True