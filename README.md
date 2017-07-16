# Intent Recognition using pytorch

[PyTorch](http://pytorch.org/) is a deep learning framework that puts Python first. This project is an intent recognisor using pytorch. Intent recognition is a natural language processing task for finding what are the actions specfied in a sentence.

Used in building virtual assistants

## Get the code

Run this command in terminal / command prompt

```
git clone https://github.com/GopikrishnanSasikumar/Text_Classifier-pytorch.git
```

## You need:

* python >= 3.0

  Install [python3](https://www.python.org/download/releases/3.0/)

* pip

  ### For linux:

  ```
  sudo apt-get install python3-pip
  ```

  ### For Mac

  To install or upgrade pip, download get-pip.py from the official site. Then run the following command:

  ```
  sudo python get-pip.py
  ```

* pytorch

  ### For mac:

  ```
  pip install http://download.pytorch.org/whl/torch-0.1.12.post2-cp35-cp35m-macosx_10_7_x86_64.whl
  ```

  ```
  pip install torchvision
  ```

  ### For Linux

  ```
  pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl
  ```

  ```
  pip install torchvision
  ```

## Running

Run

```
python3 train.py
```
for training the neural network model it Will create and store the model in ann.pt

## Testing

```
python3 test.py
```
will initiate an interactive section like this

```
User:
```
Enter the sentence and see the output.


## What's going on ?

A neural network in pytorch can be implemented like this.

```
class ANN(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(ANN,self).__init__()
        self.i2h = nn.Linear(input_size,hidden_size) #input to hidden layer
        self.h2o = nn.Linear(hidden_size,output_size) #hidden to output layer
        self.softmax = nn.LogSoftmax() #softmax layer

    def forward(self, input):
        # forward pass of the network
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output
```

Dataset used for training the neural network is initialized in a list like this,

```
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
```

Initializing pytorch model

```
ann = ANN(input_size, hidden_size, output_size)
```
Loss function

```
criterion = nn.NLLLoss()
```
Model is trained in pytorch with the iteration of these lines, last line backpropagate the loss.

```
output_p = ann(input)
loss = criterion(output_p,output)
loss.backward()
```
After some iterations model learned the task and stored what it learned in a file, 'ann.pt'. 

## It worked !
The model is tested with texts it didnt seen before. The 'test.py' use the learned model stored in 'ann.pt' to predict the intent.
```
ann=torch.load('ann.pt') #importing trained model
```
Here goes the prediction !
```
User:hello
greeting
User:good afternoon
greeting
User:see you
goodbye
User:
```
### Try out with your own dataset

Delete the 'ann.pt' file and make changes in dataset like this

```
training_data.append({"intent":"your_intent","sentence:"corresponding sentence "})
```
Run the train and test program again

## License

This project is licensed under the MIT License - see the [LICENSE.md](Text_Classifier-pytorch/LICENSE) file for details



## Built with :heart: by [Gopi](https://github.com/GopikrishnanSasikumar)













