import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import glob

class PatientDataset(Dataset):
    def __init__(self, data, root_dir):
        self.id = data[:,1]        
        self.label = data[:,2] 
        self.dir = root_dir
        self.counts = data[:,-1]
        self.ages = []
        for bod in data[:,-2]:
            self.ages.append(2024 - int(bod[-4:]))
        
        self.features = torch.Tensor(list(zip(self.counts, self.ages)))

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        ID, y = self.id[idx], self.label[idx]
        x = self.features[idx]
        files = glob.glob(self.dir + "/" + ID + "/*")
        b = len(files)
        X = torch.zeros((b, 3, 224, 224))
        for i in range(b):
            image = Image.open(files[i])
            image = np.array(image)
            image = image / 255 
            X[i,0,:,:] = torch.Tensor(image[:,:,0])
            X[i,1,:,:] = torch.Tensor(image[:,:,1])
            X[i,2,:,:] = torch.Tensor(image[:,:,2])            
            
        return (X, x), torch.Tensor([y])
