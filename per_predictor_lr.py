# Import dependencies
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pycaret.regression import *

###############
# Import data #
###############
data = pd.read_csv('Seasons_Stats.csv')

# Create subset of data >= 1990
df = data[data['Year'] >= 1990]

# Convert columns to numeric
cols = ['Year', 'Age', 'G', 'GS', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
df[cols] = df[cols].astype(int) 

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

#############
# Data Prep #
#############

# Dropped ~3k rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df[df['PER'].notna()]

############
# Modeling #
############

# Initialize setup
exp_reg = setup(data = df, target = 'PER', train_size = 0.8)

# Compare models
compare_models()

# Select best model(lr)
best_model = create_model('lr')

# Tune lr model
tuned_best = tune_model(best_model)

# Make predictions on test set
predictions = predict_model(tuned_best)

# Finalize model
final_model = finalize_model(tuned_best)

# Save model
save_model(final_model, 'final_model')