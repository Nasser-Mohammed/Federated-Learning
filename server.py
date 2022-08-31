
from torch import nn
import copy
from sklearn.metrics import confusion_matrix
import random
import torch
from neuralnet import create_model
import client


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Server:
  def __init__(self, num_clients, num_rounds, client_list, data, device):
    self.num_clients = num_clients
    self.num_rounds = num_rounds
    self.global_model = create_model().to(device)
    self.client_list = client_list
    self.lf = nn.CrossEntropyLoss
    self.test_dl = data
    self.global_dict = None


  def initialize_clients(self):
    for i in range(self.num_clients):
      string = "client " + str(i)
      cpy = client.Client(copy.deepcopy(self.global_model), i)
      self.client_list[string] = cpy
    print(self.client_list)

  def push_new_model(self):
    for index, client in enumerate(self.client_list):
      self.client_list[client].model.load_state_dict(self.global_dict)

  def server_merge(self, nameList):
    self.global_dict = copy.deepcopy(self.global_model.state_dict())
    for layer in self.global_dict:
      summation = 0
      for client in nameList:
        summation += copy.deepcopy(self.client_list[client].model.state_dict()[layer])
      self.global_dict[layer] = summation/len(nameList)

    self.global_model.load_state_dict(self.global_dict)
    self.push_new_model()
    # for person in self.client_list:
    #   self.client_list[person].model.load_state_dict(self.global_dict)

  
  def test(self):
    self.global_model.eval()
    total = 0
    correct = 0
    index = random.randint(0, self.num_clients-1)
    with torch.no_grad():
      total = 0
      correct = 0
      for i, (image, label) in enumerate(self.test_dl[index]):
        (image, label) = (image.to(device), label.to(device))
        prediction = self.global_model(image)
        total += prediction.size(0)
        correct += (prediction.argmax(1) == label).sum()
    return correct*100./total 
  






  
