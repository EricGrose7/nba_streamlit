# Import dependencies
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import torch
from torch.autograd import Variable
import joblib

################
# Prepare Data #
################
data = pd.read_csv('Seasons_Stats.csv')
df = data[data['Year'] >= 1990]

#######################
# Feature Engineering #
#######################

# Shooting Efficiency
df['Shooting_Efficiency'] = df['FG'] / df['FGA']

# 3P Efficiency
df['3P_Efficiency'] = df['3P'] / df['3PA']

# FT Efficiency
df['FT_Efficiency'] = df['FT'] / df['FTA']

# Rebound Rate
df['RB_Rate'] = df['TRB'] / df['G']

# Assist Rate
df['AST_Rate'] = df['AST'] / df['G']

# Turnover Rate
df['TO_Rate'] = df['TOV'] / df['G']

# Dropped ~3k rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df[df['PER'].notna()]

############
# Modeling #
############

# List of the features to be included in the model
model_cols = ['Shooting_Efficiency', '3P_Efficiency', 'FT_Efficiency', 'RB_Rate', 'AST_Rate', 'TO_Rate']

# Split the data into training and testing sets
x = df[model_cols].values
y = df['PER'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Save your StandardScaler instance
joblib.dump(sc, 'scaler.pkl') 

# Convert data to PyTorch tensors
x_train = Variable(torch.from_numpy(x_train)).float()
y_train = Variable(torch.from_numpy(y_train)).float()
x_test = Variable(torch.from_numpy(x_test)).float()

# Define model
model = torch.nn.Linear(x_train.shape[1], 1)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train model
epochs = 1000
for epoch in range(epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.data))

# Test model
y_pred = model(x_test)

# Convert predictions back to numpy array
y_pred = y_pred.detach().numpy()

# Save model
torch.save(model.state_dict(), 'final_model.pt')

