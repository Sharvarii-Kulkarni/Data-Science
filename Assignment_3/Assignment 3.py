#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import the pandas library
import pandas as pd
import numpy as np
#Creating a DataFrame with Missing Values
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f','h'], columns=['C01', 'C02', 'C03'])
df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print("\n Reindexed Data Values")
print("-------------------------")
print(df)
#Method 1 - Filling Every Missing Values with 0
print("\n\n Every Missing Value Replaced with '0':")
print("--------------------------------------------")
print(df.fillna(0))
#Method 2 - Dropping Rows Having Missing Values
print("\n\n Dropping Rows with Missing Values:")
print("----------------------------------------")
print(df.dropna())
#Method 3 - Replacing missing values with the Median
Valuemedian = df['C01'].median()
df['C01'].fillna(Valuemedian, inplace=True)
print("\n\n Missing Values for Column 1 Replaced with Median Value:")
print("--------------------------------------------------")
print(df)


# In[7]:


#2.2 Program for Data Transformation
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.stats as s
#Creating a DataFrame
d = {'C01':[1,3,7,4],'C02':[12,2,7,1],'C03':[22,34,-11,9]}
df2 = pd.DataFrame(d)
print("\n ORIGINAL DATA VALUES")
print("------------------------")
print(df2)
#Method 1: Rescaling Data
print("\n\n Data Scaled Between 0 to 1")
data_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
data_scaled = data_scaler.fit_transform(df2)
print("\n Min Max Scaled Data ")
print("-----------------------")
print(data_scaled.round(2))
#Method 2: Normalization rescales such that sum of each row is 1.
dn = preprocessing.normalize(df2, norm = 'l1')
print("\n L1 Normalized Data ")
print(" ----------------------")
print(dn.round(2))
#Method 3: Binarize Data (Make Binary)
data_binarized = preprocessing.Binarizer(threshold=5).transform(df2)
print("\n Binarized data ")
print(" -----------------")
print(data_binarized)
#Method 4: Standardizing Data
print("\n Standardizing Data ")
print("----------------------")
X_train = np.array([[ 1., -1., 2.],[ 2., 0., 0.],[ 0., 1., -1.]])
print(" Orginal Data \n", X_train)
print("\n Initial Mean : ", s.tmean(X_train).round(2))
print(" Initial Standard Deviation : ",round(X_train.std(),2))
X_scaled = preprocessing.scale(X_train)
X_scaled.mean(axis=0)
X_scaled.std(axis=0)
print("\n Standardized Data \n", X_scaled.round(2))
print("\n Scaled Mean : ",s.tmean(X_scaled).round(2))
print(" Scaled Standard Deviation : ",round(X_scaled.std(),2))


# In[10]:


#Program for Equal Width Binning
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#Create a Dataframe
d={'item':['Shirt','Sweater','BodyWarmer','Baby_Napkin'],
'price':[ 1250,1160,2842,1661]}
#print the Dataframe
df = pd.DataFrame(d)
print("\n ORIGINAL DATASET")
print(" ----------------")
print(df)
#Creating bins
m1=min(df["price"])
m2=max(df["price"])
bins=np.linspace(m1,m2,4)
names=["low", "medium", "high"]
df["price_bin"]=pd.cut(df["price"],bins,labels=names,include_lowest=True)
print("\n BINNED DATASET")
print(" ----------------")
print(df)


# In[61]:


import pandas as pd
url = 'https://github.com/suneet10/DataPreprocessing/blob/main/Data.csv?raw=true';
df = pd.read_csv(url, index_col=0)
df


# In[19]:


df.describe()


# In[20]:


df.shape


# In[4]:


import pandas as pd
url = 'https://github.com/suneet10/DataPreprocessing/blob/main/Data.csv?raw=true';
df = pd.read_csv(url, index_col=0)
#c) Display first 3 rows from dataset
df.head(3)


# In[23]:


#2. Handling Missing Value: a) Replace missing value of salary,age column with mean of that column.
ValuemeanAge = df['Age'].mean()
df['Age'].fillna(ValuemeanAge, inplace=True)
df


# In[24]:


#2. Handling Missing Value: a) Replace missing value of salary,age column with mean of that column.
ValuemeanAge = df['Salary'].mean()
df['Salary'].fillna(ValuemeanAge, inplace=True)
df


# In[35]:



#b. Apply Label encoding on purchased column
 
from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
df['Purchased'] = labelencoder.fit_transform(df['Purchased'])
#labelencoder = LabelEncoder() 
#df['Feedback'] = labelencoder.fit_transform(df['Feedback'])
cols = ['Purchased']
df[cols] = df[cols].apply(LabelEncoder().fit_transform)


# In[67]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
df = pd.DataFrame(ct.fit_transform(df))
df


# In[1]:


import pandas as pd
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv?raw=true";
df = pd.read_csv(url,sep = ";")
df


# In[2]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df = min_max_scaler.fit_transform(df)
#df = df.apply(lambda x: 0 if x.strip()=='N' else 1)
df


# In[3]:


#2. Rescaling: Normalised the dataset using MinMaxScaler class
import pandas, scipy, numpy
from sklearn.preprocessing import MinMaxScaler
df=pandas.read_csv( 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv ',sep=';')
array=df.values
#Separating data into input and output components
x=array[:,0:8]
y=array[:,8]
scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(x)
numpy.set_printoptions(precision=3) #Setting precision for the output
rescaledX[0:5,:]


# In[4]:


#3. Standardizing Data (transform them into a standard Gaussian distribution with a mean of 0 and a standard deviation of 1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x)
rescaledX=scaler.transform(x)
rescaledX[0:5,:]


# In[5]:


# 4. Normalizing Data ( rescale each observation to a length of 1 (a unit norm). For this, use the Normalizer class.)
from sklearn.preprocessing import Normalizer
scaler=Normalizer().fit(x)
normalizedX=scaler.transform(x)
normalizedX[0:5,:]


# In[9]:


#5. Binarizing Data using we use the Binarizer class (Using a binary threshold, it is possible to transform our data by marking the values above it 1 and those equal to or below it, 0)
from sklearn.preprocessing import Binarizer
binarizer=Binarizer(threshold=0.0).fit(x)
binaryX=binarizer.transform(x)
binaryX[0:5,:]


# In[10]:


#Set C
#Import dataset and perform Discretization of Continuous Data
#Dataset name: Student_bucketing.csv
#Dataset link: https://github.com/TrainingByPackt/Data-Science-with-Python/blob/master/Chapter01/Data/Student_bucketing.csv

# 1 Write python code to import the required libraries and load the dataset into a pandas dataframe.
import pandas as pd
df = pd.read_csv('https://github.com/TrainingByPackt/Data-Science-with-Python/blob/master/Chapter01/Data/Student_bucketing.csv?raw=true')
df


# In[11]:


#2) Display the first five rows of the dataframe.
df.head()


# In[12]:


# 3) Discretized the marks column into five discrete buckets, the labels need to be populated accordingly with five values: Poor, Below_average, Average, Above_average, and Excellent. 
#Perform bucketing using the cut () function on the marks column and display the top 10 columns.
df['bucket']=pd.cut(df['marks'],5,labels=['Poor','Below_average','Average','Above_Average','Excellent'])
df.head(10)


# In[7]:


import pandas as pd
df = pd.read_csv('https://github.com/TrainingByPackt/Data-Science-with-Python/blob/master/Chapter01/Data/Student_bucketing.csv?raw=true')
df
df.head(10)


# In[ ]:




