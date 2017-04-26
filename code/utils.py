import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import nn
import torchvision
from torchvision import models
import random

def wordToNum(word, word_dict):
    if word not in word_dict:
        # return word_dict['<UNK>']
        return word_dict['<unk>']
    else:
        return word_dict[word]
            
def lineToNum(line, word_dict, length=19):
    if line[-1] == '.':
        line = line[:-1]
    line = line.lower().split()
    if len(line) > length - 3:
        line = line[:length - 3]
    result = np.ones((length))
    for i in range(1, len(line)+1):
        result[i] = wordToNum(line[i-1], word_dict)
    # result[len(line)+1] = wordToNum('<END>', word_dict)
    result[len(line)+1] = wordToNum('<end>', word_dict)
    for i in range(len(line)+2, length):
        # result[i] = wordToNum('<NULL>', word_dict)
        result[i] = wordToNum('<pad>', word_dict)
    return result

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in xrange(N):
        words = []
        for t in xrange(T):
            word = idx_to_word[captions[i, t]]
            # if word != '<NULL>':
            if word == '<end>':
                break
            if word != '<pad>':
                words.append(word)
            # if word == '<END>':
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def coco_minibatch(batch_size, features, captions):
    size = captions.shape[0]
    mask = np.random.choice(size, batch_size)
    captions_batch = captions[mask]
    features_batch = features[mask/5]
    return captions_batch, features_batch, mask


def GetBatchData(features_all, captions_all, bsz = 100, isLSTM=True):
    minibatch = coco_minibatch(batch_size=bsz, features=features_all, captions=captions_all)
    captions_o, features_o, masks_o = minibatch
#     print captions_o.shape, features_o.shape, len(masks_o)

    ##change captions
    captions = captions_o.astype(np.int64)
    captions = torch.LongTensor(captions).t() 
    
    ##make input and target Variable:
    caption_in = Variable(captions[:-1])
    target = Variable(captions[1:]).contiguous().view(-1)
    
    ##change features to init hidden layer
    hidden_0 = features_o[np.newaxis, :]
    hidden_0 = Variable(torch.FloatTensor(hidden_0))
    
    if isLSTM:
        c_0 = hidden_0.clone()
        hidden_0 = (hidden_0, c_0)
    
    return caption_in, target, hidden_0, masks_o, captions_o


def imshow2(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean
    plt.imshow(img)

def imshow(img, targets):
    for i in range(len(targets)):
        print targets[i]
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean
    plt.imshow(img)

def GetFeaturesInBatch(data, extractor, batch_size=5, isLSTM=True):
    features_all = []
    length_data = len(data)
    length = length_data if (length_data < batch_size) else batch_size
    imgs = Variable(data)
    features = extractor.forward(imgs)['fc'].squeeze() 
    hidden_0 = features.unsqueeze(0)   
    if isLSTM:
        c_0 = hidden_0.clone()
        hidden_0 = (hidden_0, c_0)       
    return hidden_0


def ExtFeature(cap, extractor, startNum, BatchNum):
    imgs = []
    for i in range(BatchNum):
        img, _ = cap[startNum + i]
        imgs.append(img.unsqueeze(0))    
    inputs = torch.cat(imgs, 0) 
    inputs = Variable(inputs)
    features = extractor.forward(inputs)['fc'].squeeze()
    return features.data.numpy()



import torchvision
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_names):
        super(FeatureExtractor, self).__init__()
        self._model = model
        self._layer_names = set(layer_names)
        
    def forward(self, x):
        outs = {}
        for name, module in self._model._modules.iteritems():
            if name in self._layer_names:
                outs[name] = x
            if isinstance(module, nn.Linear):
                x = x.view(x.size(0), -1)
            x = module(x)
        return outs
