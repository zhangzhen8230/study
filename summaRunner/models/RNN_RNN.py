from models.BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse,logging,random
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
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=5)
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

class RNN_RNN(BasicModule):
    def __init__(self, args, embed=None):
        super(RNN_RNN, self).__init__(args)
        self.model_name = 'RNN_RNN'
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num
        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V, P_D)
        self.rel_pos_embed = nn.Embedding(S, P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(
                        input_size=D,
                        hidden_size=H,
                        batch_first=True,
                        bidirectional=True
                        )
        self.sent_RNN = nn.GRU(
                        input_size=2*H,
                        hidden_size=H,
                        batch_first=True,
                        bidirectional=True
                        )
        self.fc = nn.Linear(2*H, 2*H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D, 1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.Tensor(1).uniform_(-0.1,0.1))

    def max_pool1d(self,x,seq_lens):
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def avg_pool1d(self,x,seq_lens):
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out
    def forward(self,x,doc_lens):
        sent_lens = torch.sum(torch.sign(x),dim=1).data #can get the len?
        x = self.embed(x)
        H = self.args.hidden_size
        x = self.word_RNN(x)[0]
        word_out = self.max_pool1d(x, sent_lens)
        x = self.pad_doc(word_out, doc_lens)

        # sent level GRU
        sent_out = self.sent_RNN(x)[0]
        docs = self.max_pool1d(sent_out, doc_lens)
        probs = []
        for index, doc_len in enumerate(doc_lens):
            valid_hidden = sent_out[index,:doc_len,:]
            doc = torch.tanh(self.fc(docs[index])).unsqueeze(0)
            s = Variable(torch.zeros(1,2*H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1,-1)
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda
                abs_feature = self.abs_pos_embed(abs_index).squeeze(0)

                rel_index = int(round((position+1)*9.0/doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda
                rel_feature = self.rel_pos_embed(rel_index).squeeze(0)

                # classification layer
                content = self.content(h)
                salience = self.salience(h,doc)
                novelty = -1 * self.novelty(h, torch.tanh(s))
                abs_p = self.abs_pos(abs_feature)
                rel_p = self.rel_pos(rel_feature)
                prob = torch.sigmoid(content+salience+novelty+abs_p+rel_p+self.bias)
                probs.append(prob)
        return torch.cat(probs).squeeze()


if __name__ == '__main__':
    net = RNN_RNN(args)
    print(net)
