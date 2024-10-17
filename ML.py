import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    P=np.zeros(shape=(model.M,model.N,4,model.M,model.N))
    for i in range(model.M):
        for j in range(model.N):
            for k in range(4):
                if (model.T[i, j]) != True:
                    if k==3:
                        if j >= model.N-1 or model.W[i, j+1]:
                            P[i, j,3, i, j] += model.D[i, j, 1]
                        else:
                            P[i, j, 3, i, j+1] += model.D[i, j, 1]
                        if i >= model.M-1 or model.W[i+1, j]:
                            P[i, j, 3, i, j] += model.D[i, j, 0]
                        else:
                            P[i, j, 3, i+1, j] += model.D[i, j, 0]
                        if j < 1 or model.W[i, j-1]:
                            P[i, j, 3, i, j] += model.D[i, j, 2]
                        else:
                            P[i,j, 3, i, j-1] += model.D[i, j, 2]
                    elif k == 2:
                        if i < 1 or model.W[i-1, j]:
                            P[i, j, 2, i, j] += model.D[i, j, 1]
                        else:
                            P[i, j, 2, i-1, j] += model.D[i, j, 1]
                        if j >= model.N -1 or model.W[i, j+1]:
                            P[i, j, 2, i, j] += model.D[i, j, 0]
                        else:
                            P[i, j, 2, i, j+1] += model.D[i, j, 0]
                        if i >= model.M-1 or model.W[i+1, j]:
                            P[i, j, 2, i, j] += model.D[i, j, 2]
                        else:
                            P[i,j, 2, i+1, j] += model.D[i, j, 2]
                    elif k == 1:
                        if j < 1 or model.W[i, j-1]:
                            P[i, j, 1, i,j] += model.D[i, j, 1]
                        else:
                            P[i, j, 1, i, j-1] += model.D[i, j, 1]
                        if i < 1 or model.W[i-1, j]:
                            P[i, j, 1, i, j] += model.D[i, j, 0]
                        else:
                            P[i, j, 1, i-1, j] += model.D[i, j, 0]
                        if j >= model.N-1 or model.W[i, j+1]:
                            P[i, j, 1, i, j] += model.D[i, j, 2]
                        else:
                            P[i,j, 1, i, j+1] += model.D[i, j, 2]
                    elif k == 0:
                        if i >= model.M-1 or model.W[i+1, j]:
                            P[i, j, 0, i, j] += model.D[i, j, 1]
                        else:
                            P[i, j, 0, i+1, j] += model.D[i, j, 1]
                        if j < 1 or model.W[i, j-1]:
                            P[i, j, 0, i, j] += model.D[i, j, 0]
                        else:
                            P[i, j, 0, i, j-1] += model.D[i, j, 0]
                        if i < 1 or model.W[i-1,j]:
                            P[i, j, 0, i, j] += model.D[i, j, 2]
                        else:
                            P[i,j, 0, i-1, j] += model.D[i, j, 2]
    return P


def update_utility(model, P, U_current):
    U_next = np.zeros(shape= U_current.shape)
    for i in range(model.M):
        for j in range(model.N):
            val = -99999999999999999999999999999999999
            for k in range(4):
                counter = 0
                for z in range(model.M):
                    for h in range(model.N):
                        counter += P[i, j,k, z, h] *U_current[z, h]
                if counter > val:
                    val = counter
            U_next[i,j] = model.R[i,j] + model.gamma *val

    return U_next

def value_iteration(model):
    P = compute_transition_matrix(model)
    U_current = np.zeros(shape=(model.M, model.N))
    U_next = update_utility(model, P, U_current)
    for i in range(100):
        if np.all(U_next -U_current < epsilon):
            return U_next
        U_current = U_next
        U_next = update_utility(model, P, U_current)

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class q_learner():
    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst
        self.state_cardinality = state_cardinality

        totalNumOfStates = state_cardinality[0] * state_cardinality[1] * state_cardinality[2] * state_cardinality[3] * state_cardinality[4]
        self.Q =np.zeros(totalNumOfStates *3)
        self.N =np.zeros(totalNumOfStates *3)

    def index(self, state, action):
        index = state[0] * self.state_cardinality[1] * self.state_cardinality[2] *  self.state_cardinality[3] * self.state_cardinality[4] + \
                    state[1] * self.state_cardinality[2] * self.state_cardinality[3] * self.state_cardinality[4] +state[2] * self.state_cardinality[3] * self.state_cardinality[4] + \
                    state[3] * self.state_cardinality[4] + state[4]
        return index *3 + action + 1

    def report_exploration_counts(self, state):
        num = [0, 0,0]
        naction = [-1, 0,1]
        for i in range(3):
            action = naction[i]
            num[i] = int(self.N[self.index(state,action)])
        return num

    def choose_unexplored_action(self, state):
        action_counts = self.report_exploration_counts(state)
        movesp = []
        for i in range(3):
            if action_counts[i] <self.nfirst:
                movesp.append(i)
        for j in range(len(movesp)):
              movesp[j]= movesp[j]-1
        if not movesp:
          return None

        else:
          choice = np.random.choice(movesp)
          self.N[self.index(state, choice)] = self.N[self.index(state, choice)]+1
          return choice

    def report_q(self, state):
        empty= []
        for action in [-1,0, 1]:
            empty.append(self.Q[self.index(state,action)])
        return empty

    def q_local(self, reward, newstate):
        return reward + self.gamma * max(self.Q[self.index(newstate, -1)], self.Q[self.index(newstate, 0)],self.Q[self.index(newstate, 1)])

    def learn(self, state, action, reward, newstate):
        Q_local = self.q_local(reward, newstate)
        self.Q[self.index(state, action)] += self.alpha * (Q_local - self.Q[self.index(state, action)])

    def save(self, filename):
        np.savez(filename,self.Q, self.N)

    def load(self, filename):
        npzfile = np.load(filename)
        self.Q = npzfile['Q']
        self.N = npzfile['N']

    def exploit(self, state):
        return np.argmax([self.Q[self.index(state, -1)],self.Q[self.index(state, 0)],self.Q[self.index(state, 1)]]) - 1, np.max([self.Q[self.index(state, -1)],self.Q[self.index(state, 0)],self.Q[self.index(state, 1)]])

    def act(self, state):
        ue =self.choose_unexplored_action(state)
        if ue is None:
          random_number = np.random.uniform(0, 1)
          if random_number <self.epsilon:
              return np.random.choice([-1,0, 1])  
          mpve, nothin= self.exploit(state)
          return mpve
        return ue


import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import resnet18

num_classes = 8
num_epochs = 1
torch.set_printoptions(linewidth=200)


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


"""
1.  Define and build a PyTorch Dataset
"""


class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):

        dlist = [unpickle(data_file) for data_file in data_files]
        self.images = [image for data in dlist for image in data[b"data"]]
        self.labels = [label for data in dlist for label in data[b"labels"]]
        self.transform = transform if transform else lambda x: x
        self.target_transform = (target_transform if target_transform else lambda x: x)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).reshape(3,32,32).float()
        return image, self.labels[idx]


def get_preprocess_transform(mode):

    return lambda x: x


def build_dataset(data_files, transform=None):
    return CIFAR10(data_files, transform=transform)


"""
2.  Build a PyTorch DataLoader
"""


def build_dataloader(dataset, loader_params={"batch_size": 32, "shuffle": True}):
    return DataLoader(dataset, **loader_params)


"""
3. (a) Build a neural network class.
"""


class FinetuneNet(torch.nn.Module):
    def __init__(self, pretrained=False, pretrained_path="resnet18.pt"):
        super().__init__()
        self.model = resnet18()
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained_path))
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):

        return self.model.forward(x)


"""
3. (b)  Build a model
"""


def build_model(trained=False):
    return FinetuneNet(trained)


"""
4.  Build a PyTorch optimizer
"""


def build_optimizer(optim_type, model_params, hparams):

    if optim_type== "SGD":
        return torch.optim.SGD(params=model_params, **hparams)
    if optim_type == "Adam":
        return torch.optim.Adam(params=model_params, **hparams)


"""
5. Training loop for model
"""


def train(train_dataloader, model, loss_fn, optimizer):

    for images, labels in train_dataloader:
        output = model.forward(images)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


"""
6. Testing loop for model
"""


def test(test_dataloader, model):

    with torch.no_grad():
        c = 0
        suma = 0
        for i, j in test_dataloader:
            outputs = model.forward(i)
            pred = torch.argmax(outputs, 1)
            c += (pred==j).sum().item()
            suma += j.size(0)
        return c /suma


"""
7. Full model training and testing
"""


def run_model():

    train_dataloader = build_dataloader(
        build_dataset(
            [
                "cifar10_batches/data_batch_1",
                "cifar10_batches/data_batch_2",
                "cifar10_batches/data_batch_3",
                "cifar10_batches/data_batch_4",
                "cifar10_batches/data_batch_5",
            ],
        )
    )
    test_dataloader = build_dataloader(
        build_dataset(["cifar10_batches/test_batch"])
    )
    model = build_model(True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer("Adam", model.model.parameters(), {"lr": 0.0005})

    for epoch in range(num_epochs):
        print(f"========== Epoch {epoch} ==========")
        train(train_dataloader, model, loss_fn, optimizer)
        print(f"Accuracy: {test(test_dataloader, model)}")
    return model


if __name__ == "__main__":

    def check_data():
        files = [
            "cifar10_batches/data_batch_1",
            "cifar10_batches/data_batch_2",
        ]
        dataset = build_dataset(files)
        print("length of dataset: {}".format(len(dataset)))
        image, label = dataset[0]
        print("image type: {}".format(type(image)))
        print("label type: {}".format(type(label)))
        print("image shape: {}".format(image.shape))
        print("label: {}".format(label))
        dataloader = build_dataloader(dataset)
        for image, label in dataloader:
            print("image type (dataloader): {}".format(type(image)))
            print("label type (dataloader): {}".format(type(label)))
            print("image shape (dataloader): {}".format(image.shape))
            print("label (dataloader): {}".format(label))
            break

    def check_model():
        model = run_model()
        test_dataloader = build_dataloader(
            build_dataset(
                ["cifar10_batches/test_batch"],
                transform=get_preprocess_transform("train"),
            ),
        )
        test(test_dataloader, model)

    check_model()
