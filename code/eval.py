from utils import *
from model import *
import argparse
import matplotlib.pyplot as plt
import os, json
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default='35000', help='load model name')
parser.add_argument('--save', type=str, default='test', help='save eval data name')
parser.add_argument('--num', type=int, default=500, help='eval num')
parser.add_argument('--printn', type=int, default=10, help='print_every')
parser.add_argument('--verbose', action='store_true', help='Take a look at the eval sample')
args = parser.parse_args()


########################################################################################################
'''
Load the dictionary and model
'''
########################################################################################################

## load the dictionary
vocab_path = '../data/vocab.pkl'
from build_vocab import Vocabulary
import pickle
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
mydict = {}
mydict['idx_to_word'] = vocab.idx2word
mydict['word_to_idx'] = vocab.word2idx
# print type(mydict['idx_to_word']), len(mydict['idx_to_word'])
# print type(mydict['word_to_idx']), len(mydict['word_to_idx'])

## load the model
mydir = '../model/model_LSTM_newdic_%s.pt' % args.load
ntoken = len(mydict['idx_to_word'])
model = RNN_caption(ntoken, rnn_type='LSTM', nlayers=1, dropout=0.2)
model.load_state_dict(torch.load(mydir))

########################################################################################################
'''
load the val-image and  correspoding id
'''
########################################################################################################
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('/home/xuxin/work/python/data/coco-master/PythonAPI')
from pycocotools.coco import COCO

dataDir = '/home/xuxin/work/python/data'
imageDir = '%s/val2014' % dataDir
capDir = '%s/Micro-coco/annotations/captions_val2014.json' % dataDir
transforms = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

coco = COCO(capDir)


########################################################################################################
'''
load the feature extractor
'''
########################################################################################################
import torchvision
from torchvision import models
resnet18 = models.resnet18(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False
extractor = FeatureExtractor(resnet18, ['fc'])
########################################################################################################
'''
get batches of the val image and find the caption and ID
'''
########################################################################################################
from PIL import Image
image_ids = coco.getImgIds()
# print type(image_ids), len(image_ids)
def get_val_batch(batch_size=10, start_num=0):
    img_id = image_ids[start_num:start_num + batch_size]
    img_all = []
    for i in range(len(img_id)):
        img_name = coco.loadImgs(img_id)[i]['file_name']
        img_path = os.path.join(imageDir, img_name)
        image = transforms(Image.open(img_path).convert('RGB'))
        img = image.unsqueeze(0)
        img_all.append(img.numpy())
    img_batch = torch.from_numpy(np.vstack((x for x in img_all)))
    return img_batch, img_id


def record_val_caption(extractor, start=0, batch_size=10, end=30, print_every = 10):
    epochs = int(math.ceil((end + 0.0) / batch_size))
#     print epochs
    id_record = []
    caption_record = []
    start_time = time.time()
    for epoch in range(epochs):
        if epoch + 1 != epochs:
            img_batch, img_id = get_val_batch(batch_size=batch_size, start_num=epoch*batch_size)
        if epoch + 1 == epochs:
            img_batch, img_id = get_val_batch(batch_size=end-epoch*batch_size, start_num=epoch*batch_size)
        id_record.append(img_id)
        hidden_0 = GetFeaturesInBatch(img_batch, extractor=extractor,batch_size=batch_size, isLSTM=True)
        imageLen = hidden_0[0].size(1)

        seq_length = 17
        bsz = imageLen
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

        for i in range(bsz):
            guess_str = decode_captions(guess_seq[i], mydict['idx_to_word'])
            caption_record.append(guess_str)
        
        if epoch % print_every == 0 and epoch > 0:
            cost = time.time()-start_time
            cost_min = int(cost / 60)
            cost_sec = int(cost - cost_min * 60)
            print 'epoch %d | process %3.2f%% | cost %dm %ds' % (epoch, 100*(epoch+0.0)/epochs, cost_min, cost_sec)
        
    id_record = np.hstack((x for x in id_record))
    
    result = []
    for i in range(len(caption_record)):
        dic={}
        dic[u'image_id'] = id_record[i].item()
        dic[u'caption'] = caption_record[i]
        result.append(dic)

    return id_record, caption_record, result

id_record, caption_record, result = record_val_caption(extractor, start=0, batch_size=10, end=args.num, print_every=args.printn)


########################################################################################################
'''
save the val result
'''
########################################################################################################
save_val = (args.save != 'test')
if save_val:
    val_dir = '../data/val/%s_model%s.json' % (args.save, args.load)
    with open(val_dir, 'w') as f:
        json.dump(result, f)

########################################################################################################
'''
some random check for the caption and images
'''
########################################################################################################
import random
def check_img_id(imgid):
    img_info = coco.loadImgs(imgid)
    img_name = coco.loadImgs(imgid)[0]['file_name']
    img_url = coco.loadImgs(imgid)[0]['flickr_url']
    img_path = os.path.join(imageDir, img_name)
    image = transforms(Image.open(img_path).convert('RGB'))
    
    # print 'img_id: ', imgid
    # print 'img_info: ', img_info
    print 'img_url: ', img_url
    print 'img_name: ', img_name
    imshow2(image)  
    plt.show()  

if args.verbose:
	for i in range(4):
	    rand_num = random.randint(0, len(result))
	    # print 'rand_num: ', rand_num
	    print '='*80
	    print 'caption: ', result[rand_num]['caption']
	    check_img_id(result[rand_num]['image_id'])