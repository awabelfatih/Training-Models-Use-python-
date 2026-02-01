from sklearn. linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn. preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as warnings
warnings.filterwarnings("ignore")
# load data set 
Coursera
df = pd.read_csv('car_data.csv')
chunk_df = pd.read_csv('car_data.csv', chu)