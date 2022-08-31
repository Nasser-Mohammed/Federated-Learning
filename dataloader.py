import torch, torchvision

import copy



def load_data(num_clients):
  BATCH_SIZE = 4
  train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((.5, .5), (.5, .5))
  ]) 

  test_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      #torchvision.transforms.Normalize((.5,), (.5,))
  ])
  num_clientsCopy = copy.deepcopy(num_clients)

  train_data = torchvision.datasets.MNIST('mnist_data', train = True, download = True, transform = train_transform)

  test_data = torchvision.datasets.MNIST('mnist_data', train = False, download = True, transform = test_transform)

  train_cpy = copy.deepcopy(num_clients)

  if train_data.data.shape[0]%train_cpy != 0:
    for _ in range(train_data.data.shape[0]):
      train_cpy += 1
      if train_data.data.shape[0]%train_cpy == 0:
        print(f"Train set is divided into {train_cpy} parts")
        break
      else:
        continue
    if train_data.data.shape[0]%train_cpy != 0:
      print("Sorry we couldn't divide your data properly")
  if train_cpy == num_clients:
    print(f"No change to number of clients for train set")


  if test_data.data.shape[0]%num_clientsCopy != 0:
    for _ in range(test_data.data.shape[0]):
      num_clientsCopy += 1
      if test_data.data.shape[0]%num_clientsCopy == 0:
        print(f"Test set is divided into {num_clientsCopy} parts")
        break
      else:
        continue
    if test_data.data.shape[0]%num_clientsCopy != 0:
      print("Sorry we couldn't divide your data properly")
  if num_clientsCopy == num_clients:
    print("No change to number of clients for test set")

  traindata_split = torch.utils.data.random_split(train_data, [int(train_data.data.shape[0]/train_cpy) for _ in range(train_cpy)])

  testdata_split = torch.utils.data.random_split(test_data, [int(test_data.data.shape[0]/num_clientsCopy) for _ in range(num_clientsCopy)])

  train_dl = [torch.utils.data.DataLoader(x, batch_size = BATCH_SIZE, shuffle = True) for x in traindata_split]

  test_dl = [torch.utils.data.DataLoader(y, batch_size = BATCH_SIZE, shuffle = True) for y in testdata_split]

  return train_dl, test_dl