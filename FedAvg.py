import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import copy
import random 

#========================
#Configuaration
#========================

Num_clients=10
Fraction=0.5
Local_epochs=2
Batch_size=64
Learning_rate=0.01
Rounds=20
Device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#========================
#Model
#========================

class MLP(nn.Module):
        def __init__(self):
                super().__init__()
                self.net=nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(28*28,200),
                        nn.ReLU(),
                        nn.Linear(200,10)
                )
        def forward(self,x):
                return self.net(x)

#========================
#Dataset
#========================

transform=transforms.Compose([
        transforms.ToTensor()
])

train_dataset=datasets.MNIST(root="./data",
                             train=True,
                             transform=transform,
                             download=True)

test_dataset=datasets.MNIST(root="./data",
                            train=False,
                            download=True,
                            transform=transform)

test_loader=DataLoader(test_dataset,batch_size=Batch_size,shuffle=False)

#========================
#Create_Client_Datasets
#========================

def create_iid_clients(dataset, num_clients):
        num_items=len(dataset)//num_clients
        all_indices=np.random.permutation(len(dataset))

        clients=[]

        for i in range(num_clients):
                start=i*num_items
                end=start+num_items

                subset_indices=all_indices[start:end]
                clients.append(Subset(dataset,subset_indices))
        return clients

def create_non_iid_clients(dataset,num_clients):
        targets=np.array(dataset.targets)
        digit_indices={}

        for digit in range(10):
                digit_indices[digit]=np.where(targets==digit)[0]
        
        clients=[]

        for i in range(num_clients):
                chosen_digits=np.random.choice(range(10),2)
                indices=[]
                for d in chosen_digits:
                        idx=np.random.choice(
                                digit_indices[d],
                                3000,
                                replace=False
                        )
                        indices.extend(idx)

                clients.append(Subset(dataset,indices))

        return clients

clients=create_iid_clients(train_dataset,Num_clients)    




#=========================
#Local_Training
#=========================

def local_train(model,dataset):
        model.train()

        loader=DataLoader(
                dataset,
                batch_size=Batch_size,
                shuffle=True
        )
        optimizer=optim.SGD(
                model.parameters(),
                lr=Learning_rate
        )
        criterion=nn.CrossEntropyLoss()

        for epoch in range(Local_epochs):
                for images,labels in loader:
                        images,labels=images.to(Device),labels.to(Device)

                        optimizer.zero_grad()
                        outputs=model(images)
                        loss=criterion(outputs,labels)
                        loss.backward()
                        optimizer.step()
        return model.state_dict()

#=========================
#Federated_Averaging
#=========================

def fedavg(global_model,client_weights,client_sizes):
        global_dict=global_model.state_dict()
        total_sample=sum(client_sizes)

        for key in global_dict.keys():
                global_dict[key]=torch.zeros_like(global_dict[key])
                for i in range(len(client_weights)):
                        weight=client_sizes[i]/total_sample
                        global_dict[key]+=client_weights[i][key]*weight

        global_model.load_state_dict(global_dict)
        return global_model

#=========================
#Evaluation
#=========================

def evaluate(model):
        model.eval()

        correct=0
        total=0

        with torch.no_grad():
                for images,labels in test_loader:
                        images,labels=images.to(Device),labels.to(Device)
                        outputs=model(images)
                        _,predicted=torch.max(outputs.data,1)
                        total+=labels.size(0)
                        correct+=(predicted==labels).sum().item()
        accuracy=correct/total*100
        return accuracy 
#=========================
#Main_Federated_Learning_Loop
#=========================

global_model=MLP().to(Device)

accuracies=[]
for round_num in range(Rounds):
        print(f"\n--- Round {round_num+1} ---")

        num_selected=int(Fraction*Num_clients)
        selected_clients=random.sample(
                range(Num_clients),
                num_selected
        )

        client_weights=[]
        client_sizes=[]

        for client_id in selected_clients:
                local_model=copy.deepcopy(global_model)
                weights=local_train(
                        local_model,
                        clients[client_id]
                )
                client_weights.append(weights)
                client_sizes.append(len(clients[client_id]))

        global_model=fedavg(global_model,client_weights,client_sizes)   

        accuracy=evaluate(global_model)
        accuracies.append(accuracy)
        print(f"Test Accuracy: {accuracy:.2f}%")

# =========================
# PLOT
# =========================

plt.plot(accuracies)
plt.xlabel("Communication Round")
plt.ylabel("Test Accuracy")
plt.title("FedAvg Performance")
plt.grid(True)
plt.show()        
