#!/usr/bin/env python3
#coding=utf-8

import argparse,logging,random
import numpy as np
import torch
import json
import models
from time import time
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
import torch.nn as nn
import utils
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(asctime)s '
                                               '[INFO] %(message)s')
parser = argparse.ArgumentParser(description='Text Summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=10)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=2)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='data/train.json')
parser.add_argument('-val_dir',type=str,default='data/val.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=1500)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/test.json')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
parser.add_argument('-topk',type=int,default=3)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: you have a cuda device, should run with -device 0")

#set cuda and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

def eval(net,vocab,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        features,targets,_,doc_lens = vocab.make_features(batch)
        features,targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features,doc_lens)
        loss = criterion(probs,targets)
        total_loss += loss.item()
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss

def train():
    logging.info('Loading vocab,train and val dataset. wait a second, '
                 'please')
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    #待完成utils
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f][:200]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f][:200]
    val_dataset = utils.Dataset(examples)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # bulid mdel
    net = getattr(models, args.model)(args, embed)
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) /1e6
    print('Params:%.1fm' % (params))

    # optimizer
    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()

    t1 = time()
    for epoch in range(1, args.epochs+1):
        for i, batch in enumerate(train_iter):
            features, targets,_,doc_lens = vocab.make_features(batch)
            features, targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features,doc_lens)
            loss = criterion(probs,targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()
            if args.debug:
                logging.info('Batch ID:%d Loss:%f' % (i, loss.data[0]))
                continue
            if i % args.report_every == 0:
                cur_loss = eval(net, vocab, val_iter, criterion)
                if cur_loss<min_loss:
                    min_loss = cur_loss
                    best_path = net.save()
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                             % (epoch, min_loss, cur_loss))

        t2 = time()
        logging.info('Total Cost:%f h'%((t2-t1)/3600))
if __name__ == '__main__':
    train()