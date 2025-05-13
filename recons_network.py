# IMPORT LIBRARIES
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import math as m
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch.optim as optim
from torch import nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Define neural network's parameters
device = torch.device('cuda')
nz = 80
ngf = 64
nc = 1

img_size = 28

# Convert data to torch tensors
class Data(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(Data, self).__init__()
        # Convert input data and labels to PyTorch tensors with float32 precision
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Store the number of samples in the dataset
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len

# Define network
class Net(nn.Module):
    def __init__(self, second_dim):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()      # Flatten input to a 1D tensor per sample
        
        # Fully connected layer for the first two input features
        self.fc11 = nn.Linear(2, 16)

        # Fully connected layer for the remaining input features
        self.fc1_ = nn.Linear(second_dim, 64)

        # Convolutional transpose network
        self.nnet = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)         # Flatten input tensor
        x1 = self.fc11(x[:, :2])    # Process the first two input features
        x2 = self.fc1_(x[:, 2:])    # Process the remaining features

        # Concatenate processed features and reshape to match ConvTranspose2d input
        x = self.nnet(torch.cat((x1, x2),1).reshape((-1, nz, 1, 1)))
        x = x.view(x.size(0), -1).unsqueeze(1)  # Reshape to [batch_size, 1, 784]
        return x
    
def train_model(model, trainloader, valloader):
    loss1 = nn.MSELoss()    # Mean Squared Error loss
    loss2 = nn.L1Loss()     # Mean Absolute Error loss
    optimizer = optim.Adam(model.parameters(), lr=1e-5)   # Adam optimizer with a small learning rate
    best_loss = np.inf      # Initialize best validation loss to infinity
    patience = 5            # Number of epochs to wait for improving before early stopping

    for epoch in range(500):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            X, y = data
            X = X.to(device)
            y = y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Make predictions for this batch
            outputs = model(X)
            outputs = outputs.view(outputs.size(0), -1) # Flatten outputs
            
            # Compute the total loss (combining MSE and MAE)
            total_loss = loss1(outputs, y) + loss2(outputs, y)
            total_loss.backward()    # Backpropagation
            
            # Update model weights
            optimizer.step()
            
            # Accumulate training loss
            running_loss += total_loss.item()   
        
        with torch.no_grad():
            vrunning_loss = 0
            for i, val_data in enumerate(valloader):
                val_input, val_label = val_data
                val_input = val_input.to(device)
                val_label = val_label.unsqueeze(1).to(device)
                voutputs = model(val_input)
                vloss = loss1(voutputs, val_label) + loss2(voutputs, val_label)
                vrunning_loss += vloss.item()

        # Early stopping logic
        if vloss < best_loss:
                best_loss = vloss       # Update best validation loss
                best_model_weights = model  # Save current best model (shallow copy)
                patience = 5    # reset patience counter
        else:
             patience -= 1      # Decrease patience
             if patience == 0:
                  print(f'Stop training at epoch {epoch} !')
                  return best_model_weights    # Return best model after early stopping
        
    print('Finished Training')
    return model # Return model after training completes if not early stopped

def create_dataloader(X_train, y_train):
    # Build custom dataset with flattened image
    dataset = Data(X_train, y_train.reshape(-1, img_size*img_size))

    # Randomly split dataset into training (4000 samples) and validation (500 samples)
    train_set, val_set = torch.utils.data.random_split(dataset, [4000, 500])

    # Create DataLoader for training set with shuffling
    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)

    # Create DataLoader for validation set without shuffling
    valloader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    return trainloader, valloader

def process_data(X, y):
    scaler = MinMaxScaler() # Normalize features to the [0, 1] range
    
    # Split data into training and test sets
    # Scale original data (y) by dividing by 255 to transform to range [0, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y/255, test_size=.1, random_state=123)
    
    # Fit scaler on training features and trnasform both training and test features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
