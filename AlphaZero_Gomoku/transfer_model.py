import pickle
from collections import OrderedDict
import torch

with open('best_policy_8_8_5.model2', 'rb') as file:
    param_theano = pickle.load(file, encoding='latin1')

# param_theano = pickle.load(open(open('best_policy_8_8_5.model', 'rb'), encoding='latin-1'))
keys = ['conv1.weight' ,'conv1.bias' ,'conv2.weight' ,'conv2.bias' ,'conv3.weight' ,'conv3.bias'  
    ,'act_conv1.weight' ,'act_conv1.bias' ,'act_fc1.weight' ,'act_fc1.bias'     
    ,'val_conv1.weight' ,'val_conv1.bias' ,'val_fc1.weight' ,'val_fc1.bias' ,'val_fc2.weight' ,'val_fc2.bias']
param_pytorch = OrderedDict()
for key, value in zip(keys, param_theano):
    if 'fc' in key and 'weight' in key:
        param_pytorch[key] = torch.FloatTensor(value.T)
    elif 'conv' in key and 'weight' in key:
        param_pytorch[key] = torch.FloatTensor(value[:,:,::-1,::-1].copy())
    else:
        param_pytorch[key] = torch.FloatTensor(value)


torch.save(param_pytorch, 'best_policy_8_8_5_2torch.pth')