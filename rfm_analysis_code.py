import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# convert cause dataset is xlsx (not so good for analysis)
# excel = pd.read_excel('./online_retail.xlsx')
# excel.to_csv('./online_retail.csv')


# loading csv dataset
df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

# always important to see first lines
print(df.head())

# general information about dataset
print(df.info())

# statistical description 
print(df.describe())


# uniques customers and country distribution
print(f"\n Unique Customers: {df['Customer ID'].nunique()} \n")
print(f"Country Distribution: {df['Country'].value_counts()} \n")

# data treatment




