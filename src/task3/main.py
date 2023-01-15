import sys
sys.path.append('src')
import random

from common import *

logging.info('Loading data.')
with multiprocessing.pool.ThreadPool(len(names)) as pool:
	globals().update(dict(zip(names, pool.map(pandas.read_pickle, paths_for_cleaned))))
del pool


import optuna
from optuna.visualization import plot_contour, plot_edf, plot_optimization_history,\
  plot_parallel_coordinate, plot_param_importances, plot_slice  

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define the objective function for Optuna
def objective(trial):
    # Sample the hyperparameters
    input_size = trial.suggest_int("input_size", 50, 300)
    hidden_size = trial.suggest_int("hidden_size", 50, 300)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    num_classes = 2
    batch_size = trial.suggest_int("batch_size", 8, 64)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    num_epochs = trial.suggest_int("num_epochs", 10, 100)
    
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model and move it to the device
    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define a DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

			trial.report(loss, i+1)

    		if trial.should_prune():
      			raise optuna.TrialPruned()

  	return loss
           
study = optuna.create_study(study_name="RidgeRegression")
study.optimize(objective, n_trials=15)