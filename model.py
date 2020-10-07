from rnnmodel import docreader
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim


x = torch.ones(2,3)

q_input_size = 30
a_input_size = 400
hidden_size = 10
num_layers = 1
embedding_length = 300
batch_size = 5
learning_rate = 0.01



class Train():
    def __init__(self,epochs,mini_batches):

        #Network and input params
        opt = {}

        opt['input_size'] = q_input_size
        opt['hidden_size'] = hidden_size
        opt['num_layers'] = num_layers
        opt['embedding_length'] = embedding_length
        self.network = docreader(opt)

        #criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        print(self.network.parameters())

        parameters = [p for p in self.network.parameters()]
        self.optimizer = optim.Adamax(parameters,lr = learning_rate)
        Q_tensor = torch.randn(batch_size, q_input_size, embedding_length)
        P_tensor = torch.randn(batch_size, a_input_size, embedding_length)
        #print(P_tensor)
        ans_start = torch.zeros(batch_size,dtype=torch.long)
        ans_start[3] = 1
        # print(ans_start)
        ans_end = torch.zeros(batch_size,dtype=torch.long)
        ans_end[4] = 1
        for e in range(epochs):
            for batch in range(mini_batches):
                #Main network for docreader model

                #Values from the network
                start,end = self.network(P_tensor,Q_tensor)

                #Computing loss forward prop

                print(start[0].size(),ans_start[0].size())
                self.loss = self.criterion(start,ans_start)
                self.loss += self.criterion(end,ans_end)
                print("After {} iteration".format(e),self.loss)
                #Backward prop
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()


if __name__ == '__main__':
    Train(50,1)