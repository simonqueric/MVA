import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from torchvision import models
import tqdm 
import glob
import time
from datasets import *
from resnet import *
from efficientnet import *
import argparse

def train(epoch, print_every, name, model, optimizer, criterion, trainloader, valloader, scheduler=None, save=True) :
    if epoch % print_every == 0 :
        print('\nEpoch: %d' % epoch)
    

    # if "resnet" in name and epoch==20:
    #     optimizerCNN = torch.optim.SGD(params = model.parameters(), lr=parameters.lr_CNN, momentum=0.9) 
    global best_acc
    global loss_train # list storing the loss evolution
    global loss_val
    global accuracies
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(trainloader) :
        if "resnet" or "efficientnet" in name:
            image, clinical = x 
            clinical = clinical.to(device)
            image = image.squeeze(0).to(device)
            y = y.squeeze(0).to(device)
            optimizer.zero_grad()
            y_pred = model(image, clinical)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        else : 
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    if scheduler is not None : 
        scheduler.step()

    model.eval()
    total_loss /= i+1   
    loss_train.append(total_loss)
    if epoch % print_every == 0:
        print("Loss on training set : {:.3f}".format(total_loss))

    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad() :
        for i, (x, y) in enumerate(valloader) :
            if "resnet" in name:
                image, clinical = x 
                clinical = clinical.to(device)
                image = image.squeeze(0).to(device)
                y = y.squeeze(0).to(device)
                y_pred = model(image, clinical)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
            else : 
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
            predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
            true_labels += list(y.cpu().numpy())
        balanced_acc = balanced_accuracy_score(true_labels, predictions)      
    
    total_loss /= i+1   
    loss_val.append(total_loss)
    if epoch % print_every == 0 :
        print('\nEpoch: %d' % epoch, 'finished')
        print("Loss on validation set : {:.3f}".format(total_loss))
        print("balanced accuracy : {:.2f}".format(balanced_acc))
        
    accuracies.append(balanced_acc)
    if save :
        torch.save(model.state_dict(), "checkpoints/" + name  + "_" + str(balanced_acc)[:4]  +  "_" + str(epoch) + ".pth")
    if balanced_acc > best_acc :
        print("Accuracy improved")
        best_acc = balanced_acc

## DEFAULT PARAMETERS ##

lr_CNN = 1e-4 #5e-4 # 1e-4, 5e-4, 1e-3
epochs_CNN = 20
device = "cuda:2"

## Config ##

config = argparse.ArgumentParser(description="Configuration")
config.add_argument("--lr_CNN", type=float, default=lr_CNN)
config.add_argument("--epochs_CNN", type=int, default=epochs_CNN)
config.add_argument("--device", type=str, default=device)

parameters = config.parse_args()

print(parameters)

## DEVICE ##

device = torch.device(parameters.device if torch.cuda.is_available() else "cpu")

print("device :", device)

# images
images_dir = "trainset"

# Load clinical data
data = pd.read_csv("clinical_annotation.csv")
data_train_val = data[data["LABEL"]!=-1]
data_test = data[data["LABEL"]==-1]

# print(data_train_val.values)
# Analysis of the data

def statistics(data):
    P = len(data[data["LABEL"]==1]["ID"].values)
    N = len(data[data["LABEL"]==0]["ID"].values)
    
    p = P/(P+N) 
    n = N/(P+N)
    print("Proportion of negative samples : {:.2f}".format(p))
    print("Proportion of positive samples : {:.2f}".format(n))


# print("Size training set :", len(data_train))
# print("Size validation set :", len(data_val))
# print("Size test set :", len(data_test))
# print("Size test set :", len(glob.glob("testset/*")) - 1)


# image_train = PatientDataset(data_train.values, images_dir)
# image_val = PatientDataset(data_val.values, images_dir)

# train_loader_CNN = DataLoader(image_train, batch_size=1, shuffle=True)
# val_loader_CNN = DataLoader(image_val, batch_size=1, shuffle=True)


resnet = models.resnet18(pretrained=False)
cnn = ModifiedResNet(resnet).to(device)

# optimizerCNN = torch.optim.Adam(params = cnn.parameters(), lr=parameters.lr_CNN)#
optimizerCNN = torch.optim.SGD(params = cnn.parameters(), lr=parameters.lr_CNN, momentum=0.9)
criterionCNN = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerCNN, gamma=0.95)

# Training CNN ##

n_epochs = parameters.epochs_CNN
losses = []
best_loss = np.inf
best_acc = 0
print_every = 1

## K fold cross validation ##

k_fold = KFold(n_splits=5, shuffle = True, random_state=42)

print(data_train_val.values.shape)

for i, (train_index, val_index) in enumerate(k_fold.split(data_train_val.values)):
    train_set = PatientDataset(data_train_val.values[train_index], images_dir)
    val_set = PatientDataset(data_train_val.values[val_index], images_dir)
    train_loader = DataLoader(train_set, batch_size=1)
    val_loader = DataLoader(val_set, batch_size=1)
    
    resnet = models.resnet18(pretrained=False)
    cnn = ModifiedResNet(resnet).to(device)
    optimizerCNN = torch.optim.SGD(params = cnn.parameters(), lr=1e-4, momentum=0.9)
    criterionCNN = nn.BCELoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerCNN, gamma=0.95)
    best_acc = 0
    loss_train = []
    loss_val = []
    accuracies = []
    name = "resnet18_fold" + str(i)
    print(name)
    for epoch in tqdm.tqdm(range(n_epochs)):
        train(epoch, print_every, "resnet18_fold" + str(i) , cnn, optimizerCNN, criterionCNN, train_loader, val_loader, scheduler = scheduler, save=True)

    np.save("fold" + str(i) + "_loss_train.npy", loss_train)
    np.save("fold" + str(i) + "_loss_val.npy", loss_val)
    np.save("fold" + str(i) + "_accuracies.npy", accuracies)
