# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.autograd import Variable
import torch
import random
class ANN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(ANN,self).__init__()
        self.i2h=nn.Linear(input_size,hidden_size) 
        self.h2o=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax()
    def forward(self,input):
        #forward pass of the network
        hidden=self.i2h(input) #input to hidden layer
        output=self.h2o(hidden) #hidden to output layer
        output=self.softmax(output) #softmax layer
        return output
learning_rate=.005
hidden_size=128 #size of hidden layer
#initializing dataset
#dataset contain the input sentence and corresponding intent
training_data=[]
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

training_data.append({"intent":"sandwich", "sentence":"find me a sandwich shop"})
training_data.append({"intent":"sandwich", "sentence":"is there a sandwich shop"})
training_data.append({"intent":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"intent":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"intent":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"intent":"sandwich", "sentence":"what's for lunch?"})
all_categories=[] #list to store categories of intent
all_words=[] #for storing all words to convert input sentence into bag of words
for data in training_data:
    #storing categories into list
    if data["intent"] not in all_categories:
        all_categories.append(data["intent"])
    for word in data["sentence"].split(" "):
        #storing words in each sentence
        if word not in all_words:
            all_words.append(word)
#We need to convert a sentence into a vector with size is total number of distinct words in dataset,initialized to zero.
#If a word is in the sentence its position is made one. Thus a bag of word is created.


n_words=len(all_words) #input size
n_categories=len(all_categories) #output size   
input_size=n_words
output_size=n_categories
def wordToIndex(word):
    #finding indx of a word from all_words
    return all_words.index(word)

def sentenceToTensor(sentence):
    #input tensor initialized with zeros
    tensor=torch.zeros(1,n_words)
    for word in sentence.split(" "):
        if word not in all_words:
            #to deal with words not in dataset in evaluation stage
            continue
        tensor[0][wordToIndex(word)]=1 #making found word's position 1
    return tensor

def randomChoice(l):
    #random function for shuffling dataset
    return l[random.randint(0,len(l)-1)]

def randomTrainingExample():
    #produce random training data
    data=randomChoice(training_data)
    category=data["intent"] #intent
    category_tensor=Variable(torch.LongTensor([all_categories.index(category)])) #creating target Tensor
    sentence=data["sentence"] #input
    line_tensor=Variable(sentenceToTensor(sentence)) #input tensor
    return category,sentence,category_tensor,line_tensor

n_iters=1000 #number of iteration

ann=ANN(input_size,hidden_size,output_size) #will initialize the computation graph
criterion=nn.NLLLoss() #loss function 



def train(category_tensor,line_tensor):
    #function for training the neural net
    ann.zero_grad() #initializing gradients with zeros
    #predicting the output
    output=ann(line_tensor) #input --> hidden_layer --> output
    
        
    loss=criterion(output,category_tensor) #comparing the guessed output with actual output
    loss.backward()  #backpropagating to compute gradients with respect to loss
    
    for p in ann.parameters():
        #adding learning rate to slow down the network
        p.data.add_(-learning_rate,p.grad.data)
    return output,loss.data[0] #returning predicted output and loss

current_loss=0
for iter in range(1,n_iters+1):
    #training thenetwork for n_itersiteration
    category,sentence,category_tensor,line_tensor=randomTrainingExample() #fetching random training data
    output,loss=train(category_tensor,line_tensor) #training the neural network to predict the intent accuratly 
    current_loss+=loss #updating the error
    if iter%50==0:
       #for each 50 iteration print the error,input,actual intent,guessed intent
       top_n,top_i=output.data.topk(1)
       output_index=top_i[0][0] #converting output tensor to integer
       out_index=category_tensor.data.numpy() #converting tensor datatype to integer
       accuracy=100-(loss*100)
       if accuracy<0:
           accuracy=0
       print('accuracy=',round(accuracy),'%','input=',sentence,'actual=',all_categories[out_index[0]],'guess=',all_categories[output_index])
"""
accuracy= 0 % input= having a sandwich today? actual= sandwich guess= goodbye
accuracy= 0 % input= what's for lunch? actual= sandwich guess= sandwich
accuracy= 15 % input= can you make a sandwich? actual= sandwich guess= sandwich
accuracy= 10 % input= good day actual= greeting guess= greeting
accuracy= 36 % input= how is your day? actual= greeting guess= greeting
accuracy= 24 % input= can you make a sandwich? actual= sandwich guess= sandwich
accuracy= 49 % input= how is it going today? actual= greeting guess= greeting
accuracy= 24 % input= good day actual= greeting guess= greeting
accuracy= 20 % input= have a nice day actual= goodbye guess= goodbye
accuracy= 60 % input= is there a sandwich shop actual= sandwich guess= sandwich
accuracy= 46 % input= good morning actual= greeting guess= greeting
accuracy= 53 % input= can you make a sandwich? actual= sandwich guess= sandwich
accuracy= 71 % input= how is it going today? actual= greeting guess= greeting
accuracy= 66 % input= how are you? actual= greeting guess= greeting
accuracy= 91 % input= find me a sandwich shop actual= sandwich guess= sandwich
accuracy= 46 % input= hi there hello actual= greeting guess= greeting
accuracy= 52 % input= hi there hello actual= greeting guess= greeting
accuracy= 50 % input= have a nice day actual= goodbye guess= goodbye
accuracy= 79 % input= talk to you soon actual= goodbye guess= goodbye
accuracy= 78 % input= can you make a sandwich? actual= sandwich guess= sandwich

"""
    
    
#testing       
def evaluate(line_tensor):
    #output evaluating function
    output=ann(line_tensor)
    return output
def predict(sentence):
    #function for evaluating user input sentence
    print("input=",sentence)
    output=evaluate(Variable(sentenceToTensor(sentence)))
    top_v,top_i=output.data.topk(1)
    output_index=top_i[0][0]
    print("intent=",all_categories[output_index])
#predicting sentence the model didn't seen before
#predict("good bye")
#predict("where can i get sandwich")

"""
input= good bye
intent= greeting
input= where can i get sandwich
intent= sandwich

"""

