import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
from torchvision import models
import glob
import time
from models import *
from datasets import *
from resnet import *
import argparse


def plot_roc_curve(fpr, tpr):
    """
    Plots a ROC curve given the false positve rate (fpr) and 
    true postive rate (tpr) of a classifier.
    """
    # Plot ROC curve
    plt.plot(fpr, tpr, color='orange', label='ROC')
    # Plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("ROC.png")
    plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device :", device)

data = pd.read_csv("clinical_annotation.csv")
data_test = data[data["LABEL"]==-1]
data_rest = data[data["LABEL"]!=-1]
data_train, data_val = train_test_split(data_rest, test_size=0.2, random_state=42)  # 0.3 ? 0.4 ?, random_state=42) # test_size = 0.25
k_fold = KFold(n_splits=5, shuffle = True, random_state=42)
print(data_rest.values.shape)

images_dir = "trainset"

resnet = models.resnet18(pretrained=False) #.to(device)
# resnet = models.resnet34(pretrained=False) #.to(device)

#resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
cnn = ModifiedResNet(resnet).to(device)
cnn.load_state_dict(torch.load("checkpoints/resnet18_fold0_0.88_17.pth"))



k_fold = KFold(n_splits=5, shuffle = True, random_state=42)

for i, (train_index, val_index) in enumerate(k_fold.split(data_rest.values)):
    if i==0:
        val_set = PatientDataset(data_rest.values[val_index], images_dir)
        val_loader = DataLoader(val_set, batch_size=1)
        predictions = []
        true_labels = []

        with torch.no_grad() :
            for i, (x, y) in (enumerate(val_loader)) :
                image, clinical = x 
                clinical = clinical.to(device)
                image = image.squeeze(0).to(device)
                y = y.squeeze(0).to(device)
                y_pred = cnn(image, clinical)
                predictions += list((y_pred>0.5).detach().cpu().numpy().astype(float))
                true_labels += list(y.cpu().numpy())

        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        plot_roc_curve(fpr, tpr)

        print(roc_auc_score(true_labels, predictions))


cnn.eval()

patients = []
true_labels = []
predictions = []
for patient in tqdm(data_rest["ID"].values) :
    patients.append(patient)
    label, dob, count = data_rest[data_rest["ID"]==patient]["LABEL"].values[0], data_rest[data_rest["ID"]==patient]["DOB"].values[0], data_rest[data_rest["ID"]==patient]["LYMPH_COUNT"].values[0]
    true_labels.append(label)
    
    age = 2024 - int(dob[-4:])
    clinical_features = torch.Tensor([count, age]).to(device)
    #y_MLP = mlp(clinical_features)
    y_CNN = 0
    N = len(glob.glob(images_dir+"/"+patient+"/*"))
    X = torch.zeros((N, 3, 224, 224))
    for i, file_image in enumerate(glob.glob(images_dir+"/"+patient+"/*")) :
        image = plt.imread(file_image)
        image = image / 255
        X[i,:,:,:] = torch.Tensor(image.transpose(-1, 0, 1))
    X = X.to(device)        
    y_CNN = cnn(X, clinical_features).to(device)
        
    #y_pred =1.*((y_CNN+y_MLP) / 2 > 0.5 )
    y_pred =1.*(y_CNN>0.5)
    predictions.append(int(y_pred.cpu().numpy()[0]))

fpr, tpr, thresholds = roc_curve(true_labels, predictions)
plot_roc_curve(fpr, tpr)

print("Balanced accuracy on trainining and validation data :", balanced_accuracy_score(true_labels, predictions))

images_dir = "testset"
patients = []
predictions = []
for patient in tqdm(data_test["ID"].values) :
    patients.append(patient)
    _, dob, count = data_test[data_test["ID"]==patient]["LABEL"].values[0], data_test[data_test["ID"]==patient]["DOB"].values[0], data_test[data_test["ID"]==patient]["LYMPH_COUNT"].values[0]
    age = 2024 - int(dob[-4:])
    clinical_features = torch.Tensor([count, age]).to(device)
    y_CNN = 0
    N = len(glob.glob(images_dir+"/"+patient+"/*"))
    X = torch.zeros((N, 3, 224, 224))
    for i, file_image in enumerate(glob.glob(images_dir+"/"+patient+"/*")) :
        image = plt.imread(file_image)
        image = image / 255
        X[i,:,:,:] = torch.Tensor(image.transpose(-1, 0, 1))
    X = X.to(device)        
    y_CNN = cnn(X, clinical_features).to(device)
    y_pred =1.*(y_CNN>0.5)
    predictions.append(int(y_pred.cpu().numpy()[0]))

submission = pd.DataFrame()
submission["Id"] = patients
submission["Predicted"] = predictions
submission.to_csv("submission.csv", sep=",", index=False)

print("Submission finished")
