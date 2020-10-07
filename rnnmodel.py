

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np



class linearselfAttention(nn.Module):
    def __init__(self,input_size):
        super(linearselfAttention,self).__init__()
        self.attn = nn.Linear(input_size,1)
    
    def forward(self,x):
        scores = self.attn(x)
        scores = Func.softmax(scores,dim=1)
        return scores.squeeze()
        


class attention(nn.Module):
    def __init__(self,x_size,y_size):
        super(attention,self).__init__()
        #print(y_size,x_size)
        self.attn = nn.Linear(y_size,x_size)
        
        
    def forward(self,x,y):
        Wy = self.attn(y)

        x_Wy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        ##print("Wy",Wy.size())
        #x_Wy = x.bmm(y)
        
        if self.training:
            alpha = Func.softmax(x_Wy,dim=1)
            #print("alpha", alpha)
        else:
            alpha = Func.softmax(x_Wy,dim=1)
            #print("alpha", alpha)

        return alpha    


# In[7]:




class docreader(nn.Module):    
    def __init__(self,opt):
        super(docreader,self).__init__()
        self.input_size = opt['input_size']
        self.hidden_size = opt['hidden_size']
        self.num_layers = opt['num_layers']
        self.embedding_length = opt['embedding_length']
        self.training = False
        #print("inside docreader module",self.input_size,self.hidden_size,self.num_layers)
        self.queRNN = nn.LSTM(input_size=self.embedding_length,hidden_size=self.hidden_size,num_layers=self.num_layers,bidirectional = True, batch_first=True)
        doc_hidden_size = 2*self.hidden_size
        question_hidden_size = 2*self.hidden_size
        self.selfAttention = linearselfAttention(question_hidden_size)
        self.docRNN = nn.LSTM(self.embedding_length,self.hidden_size,num_layers=self.num_layers,bidirectional = True, batch_first=True)
        #print("doc hidden size:",doc_hidden_size)
        self.startAttention = attention(doc_hidden_size,question_hidden_size)
        self.endAttention = attention(doc_hidden_size,question_hidden_size)
        
    def forward(self,X_phrase,X_que):
        que_hidden,_ = self.queRNN(X_que)
        #print("que_hidden_size",que_hidden.size())
        que_weights = self.selfAttention(que_hidden)
        que_hiddens = weighted_average(que_hidden,que_weights)
        #print(que_hiddens.size(),"----------")
        doc_hidden,_ = self.docRNN(X_phrase)
        #print(doc_hidden.size(),"----doc")
        start = self.startAttention(doc_hidden,que_hiddens)
        end = self.endAttention(doc_hidden,que_hiddens)
        
        return start,end
        


def weighted_average(X,weights):
    w = weights.unsqueeze(1)
    return w.bmm(X).squeeze(1)

