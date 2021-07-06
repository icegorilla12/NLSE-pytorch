# import modules
import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from scipy import integrate
from torch.optim.lr_scheduler import ReduceLROnPlateau

#defining derivatives of the sigmoid function, useful in calculating the hamiltonian
def sigmoid_derivative(input):
  return torch.mul(torch.sigmoid(input),(1-torch.sigmoid(input)))

def sigmoid_second_derivative(input):
  a= torch.mul(torch.sigmoid(input),torch.sigmoid(input))
  b = torch.mul((1-torch.sigmoid(input)),(1-2*torch.sigmoid(input)))
  return torch.mul(a,b)
 
# Defining the architecture of the neural net
class Net(nn.Module):
  def __init__(self,hidden_units):
      super(Net, self).__init__()
      self.hidden=nn.Linear(1,hidden_units)
      self.fc1=nn.Linear(hidden_units,2)
      # self.fc2=nn.Linear(hidden_units,1)
      self.sigmoid=nn.Sigmoid()
      self.tanh = nn.Tanh()
      # self.sigmoid1 = diff_sigmoid()
      # self.sigmoid2 = second_diff_sigmoid()
  
  def forward(self,x):
    y = self.hidden(x)
    z = self.sigmoid(y)
    output = self.fc1(z)
    output = self.tanh(output)
    # S = self.fc2(y)
    # diff = self.sigmoid1(y)
    # doublediff = self.sigmoid2(y)
    return y, output
	
#defining loss function

def loss_func(w11,w21,w22,E,diff,doublediff,A, S,C1,C2):
  A1 = torch.matmul(w21,torch.mul(w11,diff))
  S1 = torch.matmul(w22,torch.mul(w11,diff))
  A2 = torch.matmul(w21,torch.mul(torch.mul(w11,w11),doublediff))
  S2 = torch.matmul(w22,torch.mul(torch.mul(w11,w11),doublediff))
  V = torch.mul(E,torch.mul(A,S))
  return torch.square(torch.abs(((torch.mul(A2,S)+torch.mul(2,torch.mul(A1,S1))+torch.mul(S2,A)+V)))) + torch.square((C1+C2))
  
#finding the eigen function for a particular eigen value.

from torch.optim.lr_scheduler import ReduceLROnPlateau
net=Net(50)
x = np.linspace(0,4,200)
np.random.shuffle(x)
x=torch.from_numpy(x)
x=x.float()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5) 
for i in range(200):
  print('Epoch: ',i+1)
  train_loss=0
  for j in range(len(x)):
          temp = x[j].resize_((1))
          y, output = net(temp)
          A = output[0]
          S = output[1]
          diff = sigmoid_derivative(y)
          doublediff = sigmoid_second_derivative(y)
          for name, param in net.named_parameters():
              # print (name, param.data)
              if 'fc1.weight' in name:
                fc_weights = param.detach()
                w21 = fc_weights[0]
                w22 = fc_weights[1]
              if 'hidden.weight' in name:
                w11 = param.detach()
          w21=w21.reshape((1,50))
          w22=w22.reshape((1,50))
          diff=diff.reshape((50,1))
          doublediff=doublediff.reshape((50,1))
          loss = loss_func(w11,w21,w22,2.45,diff,doublediff,A,S,torch.tensor(0),torch.tensor(0))
          train_loss += loss.item()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
        
  scheduler.step(train_loss)
  print(train_loss)

       
