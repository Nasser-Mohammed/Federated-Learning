
import torch
import matplotlib.pyplot as plt

import random
import numpy as np 
torch.backends.cudnn.benchmark = True
from dataloader import load_data
import server


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    client_list = {}

    # print("Enter how many clients you want connected to your server")
    # num_clients = int(input())
    # print("Enter number of rounds")
    # num_rounds = int(input())

    num_clients = 30
    num_rounds = 3


    if num_clients < 1 or num_rounds < 1:
        exit()
    
    data = load_data(num_clients)
    train_dl = data[0]
    test_dl = data[1]
    serv = server.Server(num_clients, num_rounds, client_list, data[1], device)
    serv.initialize_clients()

    z = 0
    for client in range(num_clients):
        cpyStr = "client " + str(z)
        serv.client_list[cpyStr].train_dl = train_dl[client]
        serv.client_list[cpyStr].test_dl = test_dl[client]
        print(f"Gathering training data for {cpyStr} at location {serv.client_list[cpyStr]}")
        print(f"We gave {cpyStr} the data loader: {train_dl[client]}, and {test_dl[client]}")
        z+=1

    print("-------------------------------------------------------------------------------------------------")

    #modelDict = {}
    acc = {}
    for round in range(serv.num_rounds):
        print(f"ROUND: {round+1}/{serv.num_rounds}")
        print("-------------------------------------------------------------------------------------------------")
        C = random.random()
        print(f"Fraction of clients is: {C}")
        #num_selected = random.randint(1, int(float(serv.num_clients)*C))
        nameList = []
        num_selected = max(int(float(serv.num_clients)*C), 1)
        client_index = np.random.permutation(serv.num_clients)[:num_selected]
        print(f"Order is: {client_index}")
        for index, client in enumerate(client_index):
            key = 'client ' + str(client)
            print(f"Training {key} at address: {serv.client_list[key]}, Round: {round+1}")
            serv.client_list[key].client_training()
            nameList.append(key)
        serv.server_merge(nameList)
        tmp = serv.test()
        print(f"Accuracy of new global model is {tmp}")
        acc[round] = tmp

    newAcc = sorted(acc.items())
    x,y = zip(*newAcc)
    plt.plot(x,y)
    plt.xlabel("Number of Rounds")
    plt.ylabel("Accuracy of Global Model")
    plt.title("Communication Rounds Effect on Accuracy")
    plt.savefig("figure2.png")
    plt.show()

if __name__ == "__main__":
    main()