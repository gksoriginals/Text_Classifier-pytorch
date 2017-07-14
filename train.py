#!/usr/bin/env python3
import torch.nn as nn
from torch.autograd import Variable
import torch
import random
from pathlib import Path


class ANN(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ANN,self).__init__()
        self.i2h = nn.Linear(input_size,hidden_size)
        self.h2o = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        # forward pass of the network
        hidden = self.i2h(input)   # input to hidden layer
        output = self.h2o(hidden)  # hidden to output layer
        output = self.softmax(output)   # softmax layer
        return output
learning_rate = .005
hidden_size = 128  # size of hidden layer
# initializing dataset
# dataset contain the input sentence and corresponding intent
training_data = list()
training_data.append({"intent":"greeting", "sentence":"how are you?"})
training_data.append({"intent":"greeting", "sentence":"how is your day?"})
training_data.append({"intent":"greeting", "sentence":"hi there hello"})
training_data.append({"intent":"greeting", "sentence":"good morning"})
training_data.append({"intent":"greeting", "sentence":"good day"})
training_data.append({"intent":"greeting", "sentence":"how is it going today?"})

training_data.append({"intent":"goodbye", "sentence":"have a nice day"})
training_data.append({"intent":"goodbye", "sentence":"see you later"})
training_data.append({"intent":"goodbye", "sentence":"good bye"})
training_data.append({"intent":"goodbye", "sentence":"talk to you soon"})
training_data.append({"intent":"goodbye", "sentence":"i have to go"})
training_data.append({"intent":"goodbye", "sentence":"i am going"})

training_data.append({"intent":"alert", "sentence":"help me"})
training_data.append({"intent":"alert", "sentence":"save me"})
training_data.append({"intent":"alert", "sentence":"i am in danger"})


training_data.append({"intent":"sandwich", "sentence":"get me sandwich"})
training_data.append({"intent":"sandwich", "sentence":"i want sandwich"})
training_data.append({"intent":"sandwich", "sentence":"where can i get sandwich"})
training_data.append({"intent":"sandwich", "sentence":"do you like sandwich"})
training_data.append({"intent":"sandwich", "sentence":"buy me sandwich"})



def dataclean(training_data):
    all_categories = list()  # list to store categories of intent
    all_words = list()  # for storing all words to convert input sentence into bag of words
    for data in training_data:
        if data["intent"] not in all_categories:
           all_categories.append(data["intent"])
        for word in data["sentence"].split(" "):

            #  storing words in each sentence
            if word not in all_words:
               all_words.append(word)
    return all_categories, all_words


all_categories, all_words = dataclean(training_data)

# We need to convert sentence into a vector with size is total number of distinct words in dataset,initialized to zero.
# If a word is in the sentence its position is made one. Thus a bag of word is created.


n_words = len(all_words)  # input size
n_categories = len(all_categories)  # output size
input_size = n_words
output_size = n_categories
hidden_size = 128

def wordToIndex(word):
    # finding indx of a word from all_words
    return all_words.index(word)


def sentencetotensor(sentence):
    # input tensor initialized with zeros
    tensor=torch.zeros(1,n_words)
    for word in sentence.split(" "):
        if word not in all_words:
            # to deal with words not in dataset in evaluation stage
            continue
        tensor[0][wordToIndex(word)] = 1  # making found word's position 1
    return tensor


def randomchoice(l):
    # random function for shuffling dataset
    return l[random.randint(0,len(l)-1)]


def randomtrainingexample():
    # produce random training data
    data = randomchoice(training_data)
    category = data["intent"]  # intent
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))  # creating target Tensor
    sentence = data["sentence"]  # input
    line_tensor = Variable(sentencetotensor(sentence))  # input tensor
    return category,sentence,category_tensor,line_tensor


ann = ANN(input_size, hidden_size, output_size)  # will initialize the computation graph
criterion = nn.NLLLoss()  #


def train(output,input,ann):

    # function for training the neural net
    ann.zero_grad()  # initializing gradients with zeros
    # predicting the output
    output_p = ann(input)  # input --> hidden_layer --> output
    loss = criterion(output_p,output)  # comparing the guessed output with actual output
    loss.backward()  # backpropagating to compute gradients with respect to loss

    for p in ann.parameters():
        # adding learning rate to slow down the network
        p.data.add_(-learning_rate,p.grad.data)
    return output,loss.data[0]  # returning predicted output and loss


def training(n_iters,ann):
    current_loss = 0
    for iter in range(1,n_iters+1):
        # training the network for n_iteration
        category,sentence,category_tensor,line_tensor = randomtrainingexample()  # fetching random training data
        output,loss = train(category_tensor,line_tensor,ann)  # training the neural network to predict the intent accuratly
        current_loss += loss  # updating the error
        if iter%100 == 0:
           # for each 50 iteration print the error,input,actual intent,guessed intent
           k = 0
           output_index = output.data.numpy()[0]
           out_index = category_tensor.data.numpy()  # converting tensor datatype to integer
           accuracy = 100-(loss*100)
           if accuracy < 0:
              accuracy = 0
           print('accuracy=',round(accuracy),'%','input=',sentence,'actual=',all_categories[out_index[0]],
                  'guess=',all_categories[output_index])
# testing


def evaluate(line_tensor,ann):
    # output evaluating function
    output = ann(line_tensor)
    return output


def predict(sentence,ann):
    # function for evaluating user input sentence
    # print("input=",sentence)
    output = evaluate(Variable(sentencetotensor(sentence)),ann)
    top_v,top_i = output.data.topk(1)
    output_index = top_i[0][0]
    return all_categories[output_index],top_v[0][0],output_index

model_file = Path('ann.pt')
if not model_file.is_file():
   training(10000,ann)
   torch.save(ann,'ann.pt')


# predicting sentence the model didn't seen before
# predict("talk to you soon")
# predict("where can i get sandwich")


