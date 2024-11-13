import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# data cleaning
df = df[df['Customer ID'].notnull()] # remove nulls
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Price'] = df['Quantity'] * df['Price'] # calculate total spend
df.drop_duplicates(inplace=True)

#rfm

reference_date = df['InvoiceDate'].max()
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate' : lambda x:(reference_date - x.max()).days,
    'Invoice' : 'nunique',
    'Price' : 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.reset_index(inplace=True)
rfm.head()

# print(rfm)

# standardization
rfm = rfm[rfm['Monetary'] > 0]
rfm[['Recency', 'Frequency', 'Monetary']] = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# k means clustering
kmeans = KMeans(n_clusters = 4, random_state = 42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
rfm.head()

# analyze and visualize
plt.figure(figsize = (10, 6))
sns.scatterplot(data = rfm, x = 'Recency', y = 'Frequency', hue = 'Cluster', palette = 'viridis')
plt.title('Customer Segments by RFM')
plt.show()

    
