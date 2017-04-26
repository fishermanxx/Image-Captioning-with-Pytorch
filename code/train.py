from utils import *
from model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default='TEST', help='load file name')
parser.add_argument('--save', type=str, default='TEST', help='save file name')
parser.add_argument('--mode', type=str, default='LSTM', help='option is : LSTM, RNN_TANH')
args = parser.parse_args()

########################################################################################################
'''
Import data
'''
########################################################################################################
###import the extracted features and coded captions
import numpy as np
import h5py
import os, json

LoadFlag = True
if LoadFlag:
    h5f = h5py.File('../data/features_0_40000.h5','r')
    features1 = h5f['dataset_1'][:]
    h5f.close()

if LoadFlag:
    h5f = h5py.File('../data/features_40000_80000.h5','r')
    features2 = h5f['dataset_1'][:]
    h5f.close()    

# if LoadFlag:
#     h5f = h5py.File('./data/captions_encode_revised.h5','r')
#     captions_all = h5f['dataset_1'][:]
#     h5f.close()
if LoadFlag:
    h5f = h5py.File('../data/captions_encode_8000.h5','r')
    captions_all = h5f['dataset_1'][:]
    h5f.close()

features_all = np.vstack((features1, features2))

##import the dictionary
# base_dir='../data'
# dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
# mydict = {}
# with open(dict_file, 'r') as f:
#     dict_data = json.load(f)
#     for k, v in dict_data.iteritems():
#         mydict[k] = v
vocab_path = '../data/vocab.pkl'
from build_vocab import Vocabulary
import pickle
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
mydict = {}
mydict['idx_to_word'] = vocab.idx2word
mydict['word_to_idx'] = vocab.word2idx



########################################################################################################
'''
Build the model
'''
########################################################################################################
import torch.nn as nn

ntoken = len(mydict['idx_to_word'])
model = RNN_caption(ntoken, rnn_type=args.mode, nlayers=1, dropout=0.2)
criterion = nn.CrossEntropyLoss()

Loadflag = (args.load != 'TEST')
if Loadflag:
    mydir = '../model/model_LSTM_newdic_%s.pt' % args.load
    ntoken = len(mydict['idx_to_word'])
    model = RNN_caption(ntoken, rnn_type=args.mode, nlayers=1, dropout=0.2)
    model.load_state_dict(torch.load(mydir))
    print 'Load model successfully!'

########################################################################################################
'''
Training code
'''
########################################################################################################
import time

isLSTM = (args.mode == 'LSTM')
def train(model, lr=1, epochs=200, print_every=10):
    model.train()
    total_loss = 0
    loss_record = []
    start_time = time.time()
    start_all = time.time()
    ntokens = len(mydict['idx_to_word'])
    for epoch in range(epochs):
        caption_in, target, hidden_0, _, _= GetBatchData(features_all, captions_all, bsz = 100, isLSTM=isLSTM)
        
        ##TODO: put a mask to delete the loss behind the end word
        model.zero_grad()
        output, hidden = model(caption_in, hidden_0)
        loss = criterion(output.view(-1, ntokens), target)
        loss.backward()
        
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
            
        total_loss += loss.data
        
        if epoch % print_every == 0 and epoch > 0:
            cur_loss = total_loss / print_every
            cur_loss = cur_loss.numpy()
            elapsed = time.time() - start_time
            elapsed_all_min = int((time.time() - start_all) / 60)
            elapsed_all_sec = (time.time() - start_all) - elapsed_all_min * 60
            print('| epoch {:3d} | process {:3d}% |loss {:3.2f} | time {:5.2f}s | cost {:3d}m {:3.2f}s'.format(epoch, 
            	100*epoch/epochs, cur_loss[0], elapsed, elapsed_all_min, elapsed_all_sec))           
            ##TODO, save the model if loss is less than the min loss
            loss_record.append(cur_loss[0])
            total_loss = 0
            start_time = time.time()

        # if epoch % 2000 == 0 and epoch >0:
        #     mydir = '../model/model_LSTM_newdic_%d.pt' % (15000 + epoch)
        #     with open(mydir, 'wb') as f:
        #         torch.save(model.state_dict(), f)
        #     print 'save model successfully!'            
    return loss_record

loss_all = train(model, lr=1, epochs=5000, print_every=50)
loss_all = np.array(loss_all)
# print type(loss_all)
# print loss_all

Saveflag = (args.save != 'TEST')
if Saveflag:
    mydir = '../model/model_LSTM_newdic_%s.pt' % args.save
    with open(mydir, 'wb') as f:
        torch.save(model.state_dict(), f)
    print 'save model successfully!'

    mydir = '../loss/loss_%s.h5' % args.save
    h5f = h5py.File(mydir, 'w')
    h5f.create_dataset('dataset_1', data=loss_all)
    h5f.close()
    print 'save loss successfully!'

# import h5py

# h5f = h5py.File('./loss/loss_test.h5','r')
# loss = h5f['dataset_1'][:]
# h5f.close()

# print loss.shape
# print type(loss)
# print loss