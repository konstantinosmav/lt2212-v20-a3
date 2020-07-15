
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import random
from sklearn.metrics import classification_report

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, activation):
        super().__init__()
        self.hidden_size = hidden_size        
        self.activation = activation
        if hidden_size > 0:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, 1)
        else:
            self.linear = nn.Linear(input_size, 1)
        if activation != " ":
            if self.activation == "relu":
                self.non_linearity = nn.ReLU()
            else:
                self.non_linearity = nn.Tanh()
            
        else:
            self.non_linearity = 0
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        if self.hidden_size == 0:
            n = self.linear(x)
        else:
            if self.non_linearity:
                l = self.linear1(x)
                m = self.non_linearity(l)
                n = self.linear2(m)
            else:
                m = self.linear1(x)
                n = self.linear2(m)
        o = self.sigmoid(n)
        return o     

    
    

class TNT:
    def __init__(self, lr =0.01):
        self.lr = lr
    def train_data(self, df, hidden_size, activation, samplesize, epoch): 
        self.model = FFNN((df.shape[1]-2)*2, hidden_size, activation)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train_test_label = df.iloc[:, 1:2]
        vector_data = df.drop(df.columns[[0,1]], axis=1)
        

        for epoch in range(epoch):
            for i in range(samplesize):
                idx1 = random.randint(0, len(vector_data)-1)
                author = train_test_label.iloc[idx1, 0]
                same_author = train_test_label[train_test_label.iloc[:, 0] == author].index
                other_author = train_test_label[train_test_label.iloc[:, 0] != author].index
                random_choice = random.randint(0, 1)
                if random_choice == 0:
                    num2 = random.choice(other_author)
                if random_choice == 1:
                    num2 = random.choice(same_author)
                docs = vector_data.to_numpy()
                tens1 = torch.Tensor(docs[idx1])
                tens2 = torch.Tensor(docs[num2])
                conc_tensor = torch.cat((tens1, tens2), 0)
                output = self.model(conc_tensor)
                label = torch.Tensor([random_choice])
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test_data(self, df, samplesize):
        predictions = []
        labels = []
        vector_data = df.drop(df.columns[[0,1]], axis=1)
        train_test_label = df.iloc[:, 1:2]
        for i in range(samplesize):
            idx1 = random.randint(0, len(vector_data)-1)
            author = train_test_label.iloc[idx1, 0]
            same_author = train_test_label[train_test_label.iloc[:, 0] == author].index
            other_author = train_test_label[train_test_label.iloc[:, 0] != author].index
            random_choice = random.randint(0, 1)
            if random_choice == 1:
                num2 = random.choice(same_author)
            if random_choice == 0:
                num2 = random.choice(other_author)
            labels.append(random_choice)
            docs = vector_data.to_numpy()
            tens1 = torch.Tensor(docs[idx1])
            tens2 = torch.Tensor(docs[num2])
            conc_tensor = torch.cat((tens1, tens2), 0)

            output = self.model(conc_tensor)
            if output > 0.5:
                prediction = 1
            else:
                prediction = 0
            predictions.append(prediction)


        print(classification_report(labels, predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--trainsize", dest="train_size", default = 210, type = int, help="Size of data used as training sample for the Neural Network.")
    parser.add_argument("--testsize", dest="test_size", default = 70, type = int, help="size of data used as testing samples for the Neural Network.")    
    parser.add_argument("--epoch", dest ="epoch", default = 3, type = int, help = "Total number of epochs the data will be passed through the Neural Network.")
    parser.add_argument("--hidden", dest="hidden_size", default = 0, type = int, help = "Hidden layer size")
    parser.add_argument("--activation", dest ="activation", default = " ", type = str, help = "Name of the activity function")
    
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    df = pd.read_csv(args.featurefile, header = None)
    train = df[df[0] == "train"]
    train.reset_index(inplace=True, drop=True)
    test = df[df[0] == "test"]
    test.reset_index(inplace=True, drop=True)

    net = TNT()
    net.train_data(train, args.hidden_size, args.activation, args.train_size, args.epoch)
    net.test_data(test, args.test_size)