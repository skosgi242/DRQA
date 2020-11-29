#from rnnmodel import docreader
from drqamodel import docreader
#from fusionmodel import fusionreader
from fusiondrqa import fusionreader
import torch
import torch.nn.functional as Func
import torch.optim as optim
import pickle
import numpy as np
import re
import string
import collections
from torch.utils.tensorboard import SummaryWriter #to print to tensor board
import time
from srumodel import srureader

hidden_size = 128
num_layers = 3
embedding_length = 300
learning_rate =  0.002
use_qemb = True
dropout = 0.3
learning_rate_decay = 0.9
grad_clipping = 10
batch_size=32
in_dropout = 0.3
out_dropout = 0.3
model = "RNN"
#model = "FUSION"
#model = "SRU"
resume = True


class Train():
    def __init__(self,config,embedding,model):
        #Network and input params
        self.opt = {}
        self.opt['hidden_size'] = hidden_size
        self.opt['num_layers'] = num_layers
        self.opt['embedding_length'] = embedding_length
        self.opt['use_qemb'] = use_qemb
        self.opt['ent_size'] = config['ent_size']
        self.opt['tag_size'] = config['tag_size']
        self.opt['fea_size'] = 3
        self.opt['vocab_size'] = config['vocab_size']
        self.opt['dropout'] = dropout
        self.opt['in_dropout'] = in_dropout
        self.opt['out_dropout']= out_dropout
        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        if model == "RNN":
            self.network = docreader(self.opt, embedding)
        elif model == "FUSION":
            self.network = fusionreader(self.opt,embedding)
        else:
            self.network = srureader(self.opt,embedding)
        self.network = self.network.to(self.device)
    def train(self,epochs,batches,dev_mb,dev_y,resume,n_file):

        # if not resume:
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adamax(parameters, lr=learning_rate)
        e_start = 0
        self.losses = []
        if resume:
            state = torch.load(n_file,map_location=torch.device('cpu'))
            self.network.load_state_dict(state['state_dict'])
            self.optimizer= state['optimizer']
            e_start = state['epoch']
            self.losses = state['losses']

        predictions = []

        self.writer = SummaryWriter(f'runs/tensorboard')
        step = 0
        self.m_step = 0
        epochs+=e_start
        for e in range(e_start, epochs):
            j = 0
            st_time = time.time()
            self.e_loss = 0
            """for batch in batches:
                self.step(batch)
                if j % 100 == 0:
                    print("{} batches are done!!!!".format(j))
                j+=1"""
            print("Training for {} iteration is done!!!!".format(e))
            predictions.clear()
            for batch in dev_mb:
                predictions.extend(self.predict(batch))
            try:
                em, f1 = self.score(predictions, y=dev_y)
            except:
                print("Error in predicting!!!")
                continue
            print("Time taken for each iteration: ",time.time()-st_time)
            print("After {} iteration".format(e), self.e_loss, em, f1)
            self.losses.append(self.e_loss)

            self.writer.add_scalar('Training Loss', self.e_loss, global_step=step)
            self.writer.add_scalar('EM', em,global_step=step)
            self.writer.add_scalar('F1', f1,global_step=step)
            step += 1


            try:
                if e % 2 == 0:
                    self.save("network" + str(e), e, scores=[em, f1])
            except:
                print("Error while saving the model file")
                continue

    def step(self,batch):
        x_inputs = [inp.to(self.device) for inp in batch[:7]]
        ans_start = batch[8]
        ans_end = batch[9]
        ans_start  = ans_start.to(self.device)
        ans_end = ans_end.to(self.device)
        # Main network for docreader model
        # Values from the network
        self.network.train()
        start, end = self.network(*x_inputs)

        # Computing loss forward prop
        self.loss = Func.nll_loss(start, ans_start) + Func.nll_loss(end, ans_end)
        self.e_loss += self.loss.item()
        self.m_step += 1
        self.writer.add_scalar('Batch Loss', self.loss, global_step=self.m_step)
        # Backward prop
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clipping)
        self.optimizer.step()

    def predict(self,batch):
        self.network.eval()
        x_inputs = [inp.to(self.device) for inp in batch[:7]]
        text = batch[7]
        span = batch[8]
        with torch.no_grad():
            start,end = self.network(*x_inputs)
        preds = []
        max_len = start.size(1)
        for i in range(start.size(0)):
            scores = torch.ger(start[i], end[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = span[i][s_idx][0], span[i][e_idx][1]
            preds.append(text[i][s_offset:e_offset])

        return preds

    def score(self,pred,y):
        f1 = em = total = 0
        for p, act in zip(pred, y):
            total += 1
            em += self._exact_match(p, act)
            f1 += self._f1_score(p, act)
        em = 100. * em / total
        f1 = 100. * f1 / total
        return em, f1

    def __normalize_answer(self,s):
        punctuations = set(string.punctuation)
        text = str(s).lower()
        text = ''.join(word for word in text if word not in punctuations)
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ' '.join(text.split())
        return text

    def _exact_match(self,pred,y):
        #print(pred)
        if len(pred) == 0  or len(y) ==0:
            return 0
        pred = self.__normalize_answer(pred)
        for ans in y:
            if pred == self.__normalize_answer(ans):
                return True
        return False

    def _f1_score(self,pred,y):
        pred = self.__normalize_answer(pred).split()
        final_score = 0

        if len(pred) ==0  or len(y) ==0:
            return 0
        for ans in y:
            ans_t = self.__normalize_answer(ans).split()
            same_words = collections.Counter(pred) & collections.Counter(ans_t)
            n_same = sum(same_words.values())
            if n_same==0:
                return 0
            precision = 1.* n_same/len(pred)
            recall = 1.* n_same/len(ans_t)
            f1_score = 2*precision*recall/(precision+recall)
            if f1_score>=final_score:
                final_score = f1_score
        return final_score

    def loadandpredict(self, dev_mb, resume, n_file):

        # if not resume:
        if resume:
            state = torch.load(n_file, map_location=torch.device('cpu'))
            self.network.load_state_dict(state['state_dict'])
            self.optimizer = state['optimizer']
            e_start = state['epoch']
            self.losses = state['losses']
        predictions = []
        for batch in dev_mb:
            predictions.extend(self.predict(batch))
        return predictions[0]

    def save(self,filename,epoch,scores):

        model={
            'state_dict' : self.network.state_dict(),
            'optimizer': self.optimizer,
            'losses' : self.losses,
            'scores':scores,
            'epoch':epoch
        }

        try:
            torch.save(model,filename)
        except:
            print("Exception while saving the model to file")


def __get_samples(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def __get_vocab_set(path):
    with open(path,"rb") as f:
        return pickle.load(f)

def __generate_batches(data,batch_size,tag_size,ent_size,type):

    #data = sorted(data,key=lambda x: len(x[1]))
    data_arg = sorted(range(len(data)),key=lambda x: len(data[x][1]))
    data = [data[i] for i in data_arg]
    #print(len(data))
    chunks = [data[i:i+batch_size] for i in range(0,len(data),batch_size)]
    batches = len(chunks)
    mini_batches = []
    for batch in chunks:
        b_size = len(batch)
        #print([len(b[1]) for b in batch])
        batch = list(zip(*batch))
        context_max = max(len(sample) for sample in batch[1])
        t_context = torch.LongTensor(b_size,context_max).fill_(0)
        for i in range(b_size):
            t_context[i,:len(batch[1][i])] = torch.tensor(batch[1][i])
            #print(t_context[i])

        t_tag = torch.LongTensor(b_size,context_max,tag_size).fill_(0)
        for i in range(b_size):
            for w in range(len(batch[3][i])):
                t_tag[i,w,batch[3][i][w]] = 1

        t_ent = torch.LongTensor(b_size,context_max,ent_size).fill_(0)
        for i in range(b_size):
            for w in range(len(batch[2][i])):
                t_ent[i, w, batch[2][i][w]] = 1
        feature_len = len(batch[4][0])
        t_features = torch.LongTensor(b_size,context_max,feature_len).fill_(0)
        for i in range(b_size):
            for w in range(feature_len):
                #print("features",t_features[i,:,].size(),torch.tensor(batch[4][i][w]).size())
                t_features[i,:len(batch[4][i][w]),w] = torch.tensor(batch[4][i][w])

        q_len = max(len(sample) for sample in batch[5])
        t_que = torch.LongTensor(b_size,q_len).fill_(0)
        for i in range(b_size):
            t_que[i,:len(batch[5][i])] = torch.tensor(batch[5][i])

        c_mask = torch.eq(t_context,0)
        q_mask = torch.eq(t_que,0)
        text = batch[6]
        span = batch[7]
        if not type == 'eval':
            ans_start = batch[8]
            ans_start = np.array(ans_start,dtype=float)
            np.nan_to_num(ans_start, copy=False)
            ans_start = torch.LongTensor(ans_start)
            ans_end = batch[9]
            ans_end = np.array(ans_end,dtype=float)
            np.nan_to_num(ans_end, copy=False)
            ans_end = torch.LongTensor(ans_end)
            mini_batches.append((t_context,t_ent,t_tag,t_features,c_mask,t_que,q_mask,span,ans_start,ans_end))
        else:
            mini_batches.append((t_context, t_ent, t_tag, t_features, c_mask, t_que, q_mask,text,span))
    #print(len(mini_batches[0]),mini_batches[0][0].size())
    return mini_batches,data_arg

def load_batches():
    samples = __get_samples("train_samples")
    train = samples['train']
    dev = [sample[:-1] for sample in samples['val']]
    dev_y = [sample[-1] for sample in samples['val']]
    vocab_set = __get_vocab_set("vocab_set")

    config = {}
    embedding = torch.tensor(vocab_set['embeddings'])
    config['vocab_size'] = embedding.size(0)
    config['e_dim'] = embedding.size(1)
    config['tag_size'] = len(vocab_set['tag_vocab'])
    config['ent_size'] = len(vocab_set['ent_vocab'])

    """train_mb,_ = __generate_batches(train, batch_size, config['tag_size'], config['ent_size'], 'train')"""
    dev_mb,data_arg = __generate_batches(dev, batch_size, config['tag_size'], config['ent_size'], 'eval')
    dev_y = [dev_y[i] for i in data_arg]
    minibatches = open("minibatches", 'rb')
    samples = pickle.load(minibatches)
    train_mb = samples['train']
    #dev_mb = samples['dev_x']
    #dev_y = samples['dev_y']
    minibatches.close()
    #train_mb = [0]
    return train_mb,dev_mb,dev_y,embedding,config

def test_main():
    print("Training is starting.....")

    train_mb,dev_mb,dev_y,embedding,config = load_batches()
    print("Train and dev batches generated. No. of batches in train are {} and dev set are {}".format(len(train_mb),
                                                                                                  len(dev_mb)))
    embedding = embedding.float()
    tr = Train(config,embedding,model)
    tr.train(40,train_mb,dev_mb,dev_y,resume,'/Users/skosgi/Downloads/network26')

if __name__ == '__main__':
    test_main()
    print(torch.cuda.is_available())

