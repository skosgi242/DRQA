
import torch
import torch.nn as nn
import torch.nn.functional as Func
from sru import SRU, SRUCell


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

class fullAttention(nn.Module):
    def __init__(self,full_size,h_size):
        super(fullAttention, self).__init__()
        self.full_size = full_size
        self.h_size = h_size
        self.linear = nn.Linear(full_size,full_size)
        self.D = nn.Parameter(torch.ones(1, full_size), requires_grad = True)
    def forward(self,xh1,xh2,x2,x2_mask):

        len1 = xh1.size(1)
        len2 = xh2.size(1)


        x1_key = Func.relu(self.linear(xh1.view(-1, xh1.size(2))).view(xh1.size()))
        x2_key = Func.relu(self.linear(xh2.view(-1, xh2.size(2))).view(xh2.size()))
        final_v = self.D.expand_as(x2_key)
        x2_key = final_v * x2_key

        scores = x1_key.bmm(x2_key.transpose(2, 1))


        x2_mask = x2_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(x2_mask.data, -float('inf'))

        alpha_flat = Func.softmax(scores.view(-1, len2), dim=1)
        alpha = alpha_flat.view(-1, len1, len2)
        # alpha = F.softmax(scores, dim=2)
        seq_ave = alpha.bmm(x2)

        return seq_ave




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


class srureader(nn.Module):
    def __init__(self ,opt ,embedding):
        super(srureader ,self).__init__()
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
        self.queRNN = SRU(input_size=self.embedding_length ,hidden_size=self.hidden_size
                              ,num_layers=self.num_layers ,bidirectional = True, dropout=self.dropout_rate,layer_norm=True)

        question_hidden_size = 2* self.hidden_size
        self.selfAttention = linearselfAttention(question_hidden_size)

        doc_hidden_size = 2 * self.hidden_size
        doc_embedding_size = self.embedding_length
        if self.opt['use_qemb']:
            self.qemb = seqAttention(self.embedding_length)
            doc_embedding_size += self.embedding_length
        if self.tag_size:
            doc_embedding_size += self.tag_size
        if self.ent_size:
            doc_embedding_size += self.ent_size
        doc_embedding_size += self.fea_size
        self.docRNN = SRU(doc_embedding_size, self.hidden_size, num_layers=self.num_layers, bidirectional=True,
                              layer_norm=True)
        # print("doc hidden size:",doc_hidden_size)
        self.startAttention = attention(doc_hidden_size, question_hidden_size)
        self.endAttention = attention(doc_hidden_size, question_hidden_size)



        self.ql = SRU(input_size=self.embedding_length ,hidden_size=self.hidden_size
                              ,num_layers=self.num_layers ,bidirectional = True
                              ,dropout=self.dropout_rate,layer_norm=True)

        self.qh = SRU(input_size=self.hidden_size*2,hidden_size=self.hidden_size
                              ,num_layers=self.num_layers ,bidirectional = True
                              ,dropout=self.dropout_rate,layer_norm=True)

        self.cl = SRU(doc_embedding_size, self.hidden_size, num_layers=self.num_layers, bidirectional=True,
                              layer_norm=True)
        self.ch = SRU(self.hidden_size*2, self.hidden_size, num_layers=self.num_layers, bidirectional=True,
                              layer_norm=True,dropout=self.dropout_rate)

        self.q_und = SRU(self.hidden_size*2*2, self.hidden_size,num_layers=self.num_layers, bidirectional=True,
                              layer_norm=True,dropout=self.dropout_rate)

        self.c_und =  SRU(2*self.hidden_size*5,self.hidden_size,num_layers=self.num_layers,bidirectional=True,
                                layer_norm=True,dropout=self.dropout_rate)

        self.low_att = fullAttention(self.embedding_length+4*self.hidden_size,self.hidden_size)
        self.high_att = fullAttention(self.embedding_length+4*self.hidden_size,self.hidden_size)
        self.u_att = fullAttention(self.embedding_length+4*self.hidden_size,self.hidden_size)



    def forward(self, X_context, X_ent, X_tag, X_f, X_cmask, X_que, X_qmask):

        X_que = self.embedding(X_que)
        X_que = nn.functional.dropout(X_que, self.dropout_rate, training=self.training)
        X_context = self.embedding(X_context)
        X_context = nn.functional.dropout(X_context, self.dropout_rate, training=self.training)



        q_low =  self.rnn_encoding(rnnlevel="q_low",X_que=X_que,X_qmask=X_qmask)
        q_high = self.rnn_encoding("q_high",q_low,X_qmask)

        q_hl = torch.cat([q_low,q_high],2)

        q_und = self.rnn_encoding("q_und",q_hl,X_qmask)

        X_phrase_list = []
        X_phrase_list.append(X_context)
        X_phrase_list.append(X_ent)
        X_phrase_list.append(X_tag)
        X_phrase_list.append(X_f)
        if self.opt['use_qemb']:
            que_emb = self.qemb(X_context, X_que, X_qmask)
            X_phrase_list.append(que_emb)
        X_c = torch.cat(X_phrase_list, 2)

        c_low = self.rnn_encoding("c_low",X_c,X_cmask)
        c_high = self.rnn_encoding("c_high",c_low,X_cmask)

        how_c = torch.cat([X_context,c_low,c_high],2)
        how_q = torch.cat([X_que,q_low,q_high],2)

        fus_l = self.low_att(how_c,how_q,q_low,X_qmask)
        fus_h = self.high_att(how_c,how_q,q_high,X_qmask)
        fus_u = self.u_att(how_c,how_q,q_und,X_qmask)

        tot_v = torch.cat([c_low,c_high,fus_l,fus_h,fus_u],2)
        tot_v = self.rnn_encoding("c_und",tot_v,X_cmask)

        que_weights = self.selfAttention(q_und, X_qmask)
        que_hiddens = weighted_average(q_und, que_weights)



        start = self.startAttention(tot_v, que_hiddens, X_cmask)
        end = self.endAttention(tot_v, que_hiddens, X_cmask)
        return start, end

    def rnn_encoding(self,rnnlevel, X_que, X_qmask):
        q_lens = X_qmask.eq(0).long().sum(1)
        q_lens = q_lens.to("cpu")
        X_que = nn.utils.rnn.pack_padded_sequence(X_que, q_lens, batch_first=True, enforce_sorted=False)
        if rnnlevel=="q_low":
            que_hidden, _ = self.ql(X_que)
        elif rnnlevel=="q_high":
            que_hidden, _ = self.qh(X_que)
        elif rnnlevel=="c_low":
            que_hidden, _ = self.cl(X_que)
        elif rnnlevel=="c_high":
            que_hidden, _ = self.ch(X_que)
        elif rnnlevel=="q_und":
            que_hidden, _ = self.q_und(X_que)
        elif  rnnlevel=="c_und":
            que_hidden,_ = self.c_und(X_que)
        que_hidden, _ = nn.utils.rnn.pad_packed_sequence(que_hidden, batch_first=True)
        que_hidden = nn.functional.dropout(que_hidden, self.out_drrate, training=self.training)
        return que_hidden

def weighted_average(X, weights):
    w = weights.unsqueeze(1)
    return w.bmm(X).squeeze(1)

