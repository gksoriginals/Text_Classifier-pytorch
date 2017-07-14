from train import predict
import random
import torch
ann=torch.load('ann.pt') #importing trained model
while True:
      k=input("User:")
      intent,top_value,top_index = predict(str(k),ann)
      print(intent)

