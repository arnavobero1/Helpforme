import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('train.csv')
df.head()

null = df.isnull().sum()

average_1 = df[df['Pclass'] == 1]['Age'].mean()
average_2 = df[df['Pclass'] == 2]['Age'].mean()
average_3 = df[df['Pclass'] == 3]['Age'].mean()

