
import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.autograd import Variable



class stackedrnns(nn.Module):
    def __init__(self,hidden_size,embedding_size,num_layers,dr_rate):
        super(stackedrnns,self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()
        self.dropout_rate = dr_rate
        self.out_drrate = dr_rate
        for i in range(num_layers):
            if i==0:
                self.rnns.append(nn.LSTM(embedding_size, self.hidden_size, num_layers=1, bidirectional=True,batch_first=True))
            else:
                self.rnns.append(nn.LSTM(2*self.hidden_size, self.hidden_size, num_layers=1, bidirectional=True,batch_first=True))
    def forward(self,X,X_mask):
        lens = X_mask.data.eq(False).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lens[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        x = X.index_select(0, idx_sort)

        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            dropout_input = nn.functional.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
            rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

    
        output = torch.cat(outputs[1:], 2)

        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        if output.size(1) != X_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  X_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        
        output = nn.functional.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

class linearselfAttention(nn.Module):
    def __init__(self ,input_size):
        super(linearselfAttention ,self).__init__()
        self.attn = nn.Linear(input_size ,1)

    def forward(self ,x ,x_mask):
        scores = self.attn(x).squeeze()
        # print(scores.size())
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        scores = Func.softmax(scores ,dim=1)
        return scores.squeeze()

class seqAttention(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
        * o_i = sum(alpha_j * y_j) for i in X
        * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self ,input_size):
        super(seqAttention ,self).__init__()
        self.linear = nn.Linear(input_size ,input_size)

    def forward(self ,X ,Y ,Y_mask):
        x_proj = self.linear(X.view(-1, X.size(2))).view(X.size())
        x_proj = Func.relu(x_proj)
        y_proj = self.linear(Y.view(-1, Y.size(2))).view(Y.size())
        y_proj = Func.relu(y_proj)

        scores = x_proj.bmm(y_proj.transpose(2 ,1))

        Y_mask = Y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill(Y_mask.data ,-float('inf'))
        alpha_flat = Func.softmax(scores.view(-1, Y.size(1)), dim=1)
        alpha = alpha_flat.view(-1, X.size(1), Y.size(1))

        # Take weighted average
        seq_ave = alpha.bmm(Y)
        return seq_ave


class attention(nn.Module):
    def __init__(self ,x_size ,y_size):
        super(attention ,self).__init__()
        # print(y_size,x_size)
        self.attn = nn.Linear(y_size ,x_size)


    def forward(self ,x ,y ,x_mask):
        Wy = self.attn(y)
        x_Wy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        # x_Wy.masked_fill(x_mask.data,-float('inf'))
        x_Wy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            alpha = Func.log_softmax(x_Wy ,dim=1)
        else:
            alpha = Func.softmax(x_Wy ,dim=1)
        return alpha


class docreader(nn.Module):
    def __init__(self ,opt ,embedding):
        super(docreader ,self).__init__()
        self.opt = opt
        self.hidden_size = self.opt['hidden_size']
        self.num_layers = self.opt['num_layers']
        self.embedding_length = self.opt['embedding_length']
        self.tag_size = self.opt['tag_size']
        self.ent_size = self.opt['ent_size']
        self.fea_size = self.opt['fea_size']
        self.dropout_rate = self.opt['dropout']
        self.in_drrate = self.opt['in_dropout']
        self.out_drrate = self.opt['out_dropout']
        """self.dropout = nn.Dropout(self.dropout_rate,training=self.training)
        self.in_dropout = nn.Dropout(self.in_drrate,training=self.training)
        self.out_dropout = nn.Dropout(self.out_drrate,training=self.training)"""
        self.training = False
        self.embedding = nn.Embedding.from_pretrained(embedding ,freeze=False)
        self.embedding.weight.requires_grad = False
        """self.queRNN = nn.LSTM(input_size=self.embedding_length ,hidden_size=self.hidden_size
                              ,num_layers=self.num_layers ,bidirectional = True, batch_first=True
                              ,dropout=self.dropout_rate)"""
        self.queRNN = stackedrnns(self.hidden_size,self.embedding_length,self.num_layers,self.in_drrate)
        question_hidden_size = 2*self.num_layers* self.hidden_size
        self.selfAttention = linearselfAttention(question_hidden_size)
        doc_hidden_size = 2*self.num_layers * self.hidden_size
        doc_embedding_size = self.embedding_length
        if self.opt['use_qemb']:
            self.qemb = seqAttention(self.embedding_length)
            doc_embedding_size += self.embedding_length
        if self.tag_size:
            doc_embedding_size += self.tag_size
        if self.ent_size:
            doc_embedding_size += self.ent_size
        doc_embedding_size += self.fea_size
        print(doc_embedding_size, self.fea_size, self.ent_size, self.embedding_length, self.tag_size,self.in_drrate)
        """self.docRNN = nn.LSTM(doc_embedding_size, self.hidden_size, num_layers=self.num_layers, bidirectional=True,
                              batch_first=True, dropout=self.dropout_rate)"""
        self.docRNN = stackedrnns(self.hidden_size,doc_embedding_size,self.num_layers,self.in_drrate)
        # print("doc hidden size:",doc_hidden_size)
        self.startAttention = attention(doc_hidden_size, question_hidden_size)
        self.endAttention = attention(doc_hidden_size, question_hidden_size)

    def forward(self, X_context, X_ent, X_tag, X_f, X_cmask, X_que, X_qmask):

        X_que = self.embedding(X_que)
        X_que = nn.functional.dropout(X_que, self.dropout_rate, training=self.training)
        X_context = self.embedding(X_context)
        X_context = nn.functional.dropout(X_context, self.dropout_rate, training=self.training)
        """que_hiddens = self.que_rnn(X_que, X_qmask)
        doc_hiddens = self.doc_rnn(X_context, X_cmask, X_ent, X_tag, X_f, X_que, X_qmask)"""
        que_hiddens = self.queRNN(X_que,X_qmask)
        que_weights = self.selfAttention(que_hiddens, X_qmask)
        que_hiddens = weighted_average(que_hiddens, que_weights)
        X_phrase_list = []
        X_phrase_list.append(X_context)
        X_phrase_list.append(X_ent)
        X_phrase_list.append(X_tag)
        X_phrase_list.append(X_f)
        if self.opt['use_qemb']:
            que_emb = self.qemb(X_context, X_que, X_qmask)
            X_phrase_list.append(que_emb)
        X_context = torch.cat(X_phrase_list, 2)
        doc_hiddens = self.docRNN(X_context,X_cmask)
        start = self.startAttention(doc_hiddens, que_hiddens, X_cmask)
        end = self.endAttention(doc_hiddens, que_hiddens, X_cmask)
        return start, end

def weighted_average(X, weights):
    w = weights.unsqueeze(1)
    return w.bmm(X).squeeze(1)

