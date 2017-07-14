from train import predict
<<<<<<< HEAD
import random
import torch
ann=torch.load('ann.pt')
while True:
      k=input("User:")
      intent,top_value,top_index = predict(str(k),ann)
      print(intent,top_value,top_index)
      #if intent=="greeting":
          #print("sakhi:",random.choice(["hello","hi there","greetings","hello it is a pleasure tto meet you"]))
      #elif intent=="goodbye":
          #print("sakhi:",random.choice(["see you soon","ok bye","stay safe","have a nice time"]))
      #elif intent=="bot":
          #print("sakhi:",random.choice(["my name is sakhi,i am from sango","hi i am sakhi how can i help you?"]))
      #elif intent=="sango":
          #print("sakhi",random.choice(["we are sango,we work for women security",
                                       #"we are a group of engineers called sango. we are trying to reduce crimes using technology","sango,a non profitable organisation working for women safety"]))
      #elif intent=="alert":
          #print("sakhi",random.choice(["sending help","triggering panic message","finding nearby safe zone"]))
      #else:
          #print("error")
=======
while True:
      k=input("User:") 
      predict(str(k))
>>>>>>> 56d8ee61f556e25b5c52adb07f03111913ed1f94
