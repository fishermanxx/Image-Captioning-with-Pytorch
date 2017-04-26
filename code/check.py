from utils import *
from model import *
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='test', help='load file name')
parser.add_argument('--mode', type=str, default='LSTM', help='option is : LSTM, RNN_TANH')
parser.add_argument('--data', type=str, default='train2014', 
    help='option: train2014, val2014')
parser.add_argument('--check_name', type=str, default='train', 
    help='check for the extract features or the captions, option: train, val')

args = parser.parse_args()

########################################################################################################
'''
Load the Data
TODO put this part into utils.py
'''
########################################################################################################
###import the extracted features and coded captions
import numpy as np
import h5py
import os, json

LoadFlag = args.check_name
if LoadFlag == 'train':
    h5f = h5py.File('../data/features_0_40000.h5','r')
    features1 = h5f['dataset_1'][:]
    h5f.close()

    h5f = h5py.File('../data/features_40000_80000.h5','r')
    features2 = h5f['dataset_1'][:]
    h5f.close()    

    h5f = h5py.File('../data/captions_encode_8000.h5','r')
    # h5f = h5py.File('../data/captions_encode_revised.h5','r')
    captions_all = h5f['dataset_1'][:]
    h5f.close()

    features_all = np.vstack((features1, features2))

if LoadFlag == 'val':
    h5f = h5py.File('../data/features_val.h5','r')
    features = h5f['dataset_1'][:]
    features_all = features
    h5f.close()    

    h5f = h5py.File('../data/captions_encode_val.h5','r')
    captions_all = h5f['dataset_1'][:]
    h5f.close()


print 'features: ', type(features_all), features_all.shape, features_all.dtype
print 'captions: ', type(captions_all), captions_all.shape, captions_all.dtype
print '='*60
##import the dictionary

vocab_path = '../data/vocab.pkl'
from build_vocab import Vocabulary
import pickle
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
mydict = {}
mydict['idx_to_word'] = vocab.idx2word
mydict['word_to_idx'] = vocab.word2idx

# base_dir='../data'
# dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
# mydict = {}
# with open(dict_file, 'r') as f:
#     dict_data = json.load(f)
#     for k, v in dict_data.iteritems():
#         mydict[k] = v

########################################################################################################
'''
Load the model
'''
########################################################################################################
mydir = '../model/model_LSTM_newdic_%s.pt' % args.name
ntoken = len(mydict['idx_to_word'])
model = RNN_caption(ntoken, rnn_type=args.mode, nlayers=1, dropout=0.2)
model.load_state_dict(torch.load(mydir))
print 'Load model successfully!'

########################################################################################################
'''
Load the Image and check with training data
'''
########################################################################################################
import torch
from torch import nn
import torchvision
from torchvision import models

import sys
sys.path.append('/home/xuxin/work/python/data/coco-master/PythonAPI')

import torchvision.datasets as dset
import torchvision.transforms as transforms

dataDir = '/home/xuxin/work/python/data'
imageDir = '%s/%s' % (dataDir, args.data)
capDir = '%s/Micro-coco/annotations/captions_%s.json' % (dataDir, args.data)
transforms = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

cap = dset.CocoCaptions(root=imageDir, annFile=capDir, transform=transforms)
print type(cap)
print('Number of samples: ', len(cap))


########################################################################################################
'''
check one, give input
'''
########################################################################################################
import time

isLSTM = (args.mode == 'LSTM')
check_flag1 = False
if check_flag1:
    
    criterion = nn.CrossEntropyLoss()
    def evaluate(model, lr=1, epochs=200, print_every=10):
        model.eval()
        total_loss = 0
        loss_record = 0
        start_time = time.time()
        ntokens = len(mydict['idx_to_word'])
        for epoch in range(epochs):
            caption_in, target, hidden_0, _, _= GetBatchData(features_all, captions_all, bsz = 100, isLSTM=isLSTM)
            output, hidden = model(caption_in, hidden_0)
            loss = criterion(output.view(-1, ntokens), target)
            total_loss += loss.data.numpy()

            if epoch % print_every == 0 and epoch > 0:
                elapsed = time.time() - start_time
                cur_loss = total_loss / print_every
                print('| epoch {:3d} | loss {:3} | time {:5.2f}s'.format(epoch, cur_loss, elapsed))           
                start_time = time.time()
                loss_record += cur_loss
                total_loss = 0
        return loss_record / (epochs/print_every - 1)

    loss_eval = evaluate(model, lr=1, epochs=30, print_every=5)   
    print '='*80
    print 'val loss: ', loss_eval
    print '='*80
# check_num = 5
# isLSTM = (args.mode == 'LSTM')
# caption_in, target, hidden_0, masks_o, captions_o = GetBatchData(features_all, captions_all, bsz = 100, isLSTM=isLSTM)

# model.eval()
# output, _ = model(caption_in, hidden_0)
# _, topi = output.data.topk(1)
# guess = topi.numpy().squeeze().T

# for i in range(check_num):
#     guess_str = decode_captions(guess[i], mydict['idx_to_word'])
#     print '='*80
#     print 'No %d: ' % (i+1)
#     print guess_str    
#     caption_str = decode_captions(captions_all[masks_o[i]], mydict['idx_to_word'])
#     print caption_str  
#     print '='*80	
#     img, _ = cap[(masks_o/5)[i]]
#     imshow2(img)
#     plt.show()

########################################################################################################
'''
check two, give just one <Start>
'''
########################################################################################################

bsz = 5
_, _, hidden_0, masks_o, _ = GetBatchData(features_all, captions_all, bsz = bsz, isLSTM=isLSTM)

start_caption = Variable(torch.ones(1, bsz).long())
seq_length = 17

hidden = hidden_0
caption_in = start_caption
# print type(caption_in), caption_in.size()

output_all = []
model.eval()
for i in range(seq_length):
    output, hidden = model(caption_in, hidden)
    _, topi = output.data.topk(1)
    guess = topi.numpy().squeeze().T
    output_all.append(guess)
    caption_in = Variable(torch.from_numpy(guess).unsqueeze(0))

guess_seq = np.vstack((x for x in output_all)).T

print 'Check Two: '
for i in range(bsz):
    guess_str = decode_captions(guess_seq[i], mydict['idx_to_word'])
    print '='*80
    print 'No %d: ' % (i+1)
    print guess_str
    img, _ = cap[(masks_o/5)[i]]
    imshow2(img)
    plt.show()    


########################################################################################################
'''
check three, give my own picture
'''
########################################################################################################    
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

data_transforms = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_dir = '../mypic'
dset = datasets.ImageFolder(data_dir, data_transforms)
dset_loader = DataLoader(dset, batch_size = 5, num_workers=1)

dset_size = len(dset)
# print dset_size
dset_classes = dset.classes

import numpy as np
import matplotlib.pyplot as plt

def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    plt.imshow(inp)
    
inputs, classes = next(iter(dset_loader))

resnet18 = models.resnet18(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False
extractor = FeatureExtractor(resnet18, ['fc'])

hidden_0 = GetFeaturesInBatch(inputs, extractor, batch_size=5, isLSTM=True)
imageLen = hidden_0[0].size(1)   


seq_length = 17
bsz = imageLen if imageLen < 5 else 5
# print bsz
start_caption = Variable(torch.ones(1, bsz).long())
hidden = hidden_0
caption_in = start_caption

output_all = []
model.eval()
for i in range(seq_length):
    output, hidden = model(caption_in, hidden)
    _, topi = output.data.topk(1)
    guess = topi.numpy().squeeze().T
    output_all.append(guess)
    caption_in = Variable(torch.from_numpy(guess).unsqueeze(0))

guess_seq = np.vstack((x for x in output_all)).T
# print guess_seq.shape
print '='*80
print 'Check Three: '
print '='*80
for i in range(bsz):
    guess_str = decode_captions(guess_seq[i], mydict['idx_to_word'])
    print '='*80
    print 'No %d: ' % (i+1)
    print guess_str
    img = inputs[i] 
    imshow(img)
    plt.show() 