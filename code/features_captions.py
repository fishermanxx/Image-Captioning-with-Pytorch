from utils import *
import time
import torch
from torch import nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='Take a look at the data sample')
parser.add_argument('--data', type=str, default='train2014', 
    help='option: train2014, val2014')
parser.add_argument('--feature', type=str, default='TEST', 
    help='start extract features and difine the save file name, (should not be TEST)')
parser.add_argument('--caption', type=str, default='TEST', 
    help='start encode captions and difine the save file name, (should not be TEST)')
parser.add_argument('--check', action='store_true')
parser.add_argument('--check_name', type=str, default='train', 
    help='check for the extract features or the captions, option: train, val')
args = parser.parse_args()
########################################################################################################
'''
Load COCO Data
'''
########################################################################################################    
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

##Take a look of the data
visible = args.verbose
if visible:
	rand_num = random.randint(0, len(cap))
	img, targets = cap[rand_num]
	imshow(img, targets)
	plt.show()


########################################################################################################
'''
Extract the features of img with resnet18
'''
########################################################################################################    
resnet18 = models.resnet18(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False
extractor = FeatureExtractor(resnet18, ['fc'])


import time
StartFlag_feature = (args.feature != 'TEST')
if StartFlag_feature:
    startall = time.time()
    features_all = []
    features_save = []
    start = time.time()
    Image_all_Num = 4000
    print_every = 100
    for epoch in range(Image_all_Num):
        features_all.append(ExtFeature(cap=cap, extractor=extractor, startNum=epoch*10, BatchNum=10))
        if epoch%print_every == 0 and epoch>0:      
            print '%dth epoch | %d%% images is done | takes %ds' % (epoch/print_every, 100 * epoch / Image_all_Num, time.time() - start)
            start = time.time()

    features_save = np.vstack((x for x in features_all))
    print 'takes %ds' % (time.time() - startall)
    print 'features_save type: ', type(features_save)
    print 'features_save shape: ', features_save.shape

########################################################################################################
'''
Extract the features of img with resnet18
'''
########################################################################################################    

SaveFlag_feature = (args.feature != 'TEST')
if SaveFlag_feature:
    mydir = 'features_%s.h5' % args.feature
    h5f = h5py.File(mydir, 'w') 
    h5f.create_dataset('dataset_1', data=features_save)
    h5f.close()
    print 'Save Features successfully!'

########################################################################################################
'''
Load the dict that others make
'''
########################################################################################################    

import os, json
base_dir='../data'
dict_file = os.path.join(base_dir, 'coco2014_vocab.json')

mydict = {}
with open(dict_file, 'r') as f:
    dict_data = json.load(f)
    for k, v in dict_data.iteritems():
        mydict[k] = v

# print type(mydict['idx_to_word'])
# print len(mydict['idx_to_word'])
# print type(mydict['word_to_idx'])
# print len(mydict['word_to_idx'])
# print mydict['idx_to_word'][:10]

########################################################################################################
'''
Encode the sentence into number with the dict
'''
########################################################################################################    

StartFlag_caption = (args.caption != 'TEST')
if StartFlag_caption:
    start = time.time()
    epochs_all = 40000
    print_every = 100
    captions_num = []
    for epoch in range(epochs_all):
        _, captions = cap[epoch] 
        for i in range(5):
            captions_num.append(lineToNum(captions[i], mydict['word_to_idx'], 17))
        if epoch % print_every == 0 and epoch > 0:
            print '%dth epoch | %d%% captions | cost %ds' %(epoch/print_every, 100*epoch/epochs_all, time.time()-start)
            start = time.time()
            
    start = time.time()
    print 'transfer started!'
    captions_save = np.stack((x for x in captions_num), axis=0)
    captions_save = captions_save.astype('int32')
    print 'transfer cost %ds' % (time.time()-start)
    print type(captions_save), captions_save.shape, captions_save.dtype


########################################################################################################
'''
Save the captions in h5
'''
########################################################################################################    

SaveFlag_caption = (args.caption != 'TEST')
if SaveFlag_caption:
    mydir = 'captions_encode_%s.h5' % args.caption
    h5f = h5py.File(mydir, 'w')
    h5f.create_dataset('dataset_1', data=captions_save)
    h5f.close()
    print 'Save captions successfully!'


########################################################################################################
'''
Check for the order of the features and captions
'''
########################################################################################################
##Load features and captions

check_Flag = args.check
if check_Flag:
    LoadFlag = args.check_name
    if LoadFlag == 'train':
        h5f = h5py.File('../data/features_0_40000.h5','r')
        features1 = h5f['dataset_1'][:]
        h5f.close()

        h5f = h5py.File('../data/features_40000_80000.h5','r')
        features2 = h5f['dataset_1'][:]
        h5f.close()    

        h5f = h5py.File('../data/captions_encode_revised.h5','r')
        captions_all = h5f['dataset_1'][:]
        h5f.close()

        features_all = np.vstack((features1, features2))

    if LoadFlag == 'val':
        h5f = h5py.File('../data/features_val.h5','r')
        features = h5f['dataset_1'][:]
        h5f.close()    

        h5f = h5py.File('../data/captions_encode_val.h5','r')
        captions_all = h5f['dataset_1'][:]
        h5f.close()

        features_all = features

    print 'features: ', type(features_all), features_all.shape, features_all.dtype
    print 'captions: ', type(captions_all), captions_all.shape, captions_all.dtype
    print '='*60

    check_num = random.randint(0, len(features_all))
    check_batch = check_num/10
    batch_id = check_num % 10
    caption_id = check_num * 5
    print check_num, check_batch, batch_id, caption_id
    check_feature = ExtFeature(cap, extractor, check_batch*10, 10)

    print '='*60
    print '1:', features_all[check_num][:5]
    print '2:', check_feature[batch_id][:5]
    print '='*60

    img, targets = cap[check_num]

    for i in range(5):
        print '1:', captions_all[caption_id+i]
        target = targets[i]
        print '2:', lineToNum(target, mydict['word_to_idx'], 17).astype('int32')
        print decode_captions(captions_all[caption_id+i], mydict['idx_to_word'])

    print '='*60
    imshow(img, targets)
    plt.show()