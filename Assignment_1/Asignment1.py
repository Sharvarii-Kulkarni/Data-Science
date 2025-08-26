#!/usr/bin/env python
# coding: utf-8

# # Assihnment-1

# Set A
# 1. Write a Python program to create a dataframe containing columns name, age and percentage. Add 10 rows to the dataframe. View the dataframe.
# 2. Write a Python program to print the shape, number of rows-columns, data types, feature names and the description of the data
# 3. Write a Python program to view basic statistical details of the data.
# 4. Write a Python program to Add 5 rows with duplicate values and missing values. Add a column ‘remarks’ with empty values. Display the data
# 5. Write a Python program to get the number of observations, missing values and duplicate values.
# 6. Write a Python program to drop ‘remarks’ column from the dataframe. Also drop all null and empty values. Print the modified data.
# 7. Write a Python program to generate a line plot of name vs percentage
# 8. Write a Python program to generate a scatter plot of name vs percentage

# 1. Write a Python program to create a dataframe containing columns name, age and percentage. Add 10 rows to the dataframe. View the dataframe.

# In[15]:


import pandas as pd
data = {'NAME':['Abhishek','Isha','Anil','Sunil','Kshama','shanti','Rama','Radha','Arpita','Adwaita'],
       'AGE':[15,16,17,18,19,15,16,17,18,19],
       'PERCENTAGE':[78.89,90.89,67.98,55,88.99,78.99,67.77,66,90,77]}
df = pd.DataFrame(data)
df


# 2. Write a Python program to print the shape, number of rows-columns, data types, feature names and the description of the data

# In[5]:


print(df.shape)


# In[6]:


total_rows=len(df.axes[0])
total_cols=len(df.axes[1])
print("Number of Rows: "+str(total_rows))
print("Number of Columns: "+str(total_cols))


# In[13]:


print("Feature Names : ")
df.keys()


# 3. Write a Python program to view basic statistical details of the data.

# In[12]:


df.describe()


# 4. Write a Python program to Add 5 rows with duplicate values and missing values. Add a column ‘remarks’ with empty values. Display the data

# In[20]:


df.loc[len(df.index)] = ['Amy', 89, 93] 
df
df2 = {'NAME': 'ANIL', 'AGE': 89, 'PERCENTAGE': 93}
df = df.append(df2, ignore_index = True)
df


# 5. Write a Python program to get the number of observations, missing values and duplicate values.

# In[24]:


df.info()
df.shape


# In[33]:


df.isnull() 


# In[26]:


df.duplicated() 


# 6. Write a Python program to drop ‘remarks’ column from the dataframe. Also drop all null and empty values. Print the modified data.

# In[31]:


df["remarks"] = None
df


# In[32]:


df.drop(columns='remarks', axis=1, inplace=True)
df


# 7. Write a Python program to generate a line plot of name vs percentage

# In[35]:


import matplotlib.pyplot as plt
plt.plot(df['NAME'], df['PERCENTAGE'], "r--")
plt.show


# 8. Write a Python program to generate a scatter plot of name vs percentage

# In[41]:


df.plot(kind ="scatter", x='AGE', y='PERCENTAGE')


# # Set B
# 1. Download the heights and weights dataset and load the dataset from a given csv file into a dataframe.Print the first, last 10 rows and random 20 rows. (https://www.kaggle.com/burnoutminer/heightsand-weights-dataset)
# 2. Write a Python program to find the shape, size, datatypes of the dataframe object.
# 3. Write a Python program to view basic statistical details of the data.
# 4. Write a Python program to get the number of observations, missing values and nan values.
# 5. Write a Python program to add a column to the dataframe “BMI” which is calculated as :weight/height2
# 6. Write a Python program to find the maximum and minimum BMI.
# 7. Write a Python program to generate a scatter plot of height vs weight
# 

# 1. Download the heights and weights dataset and load the dataset from a given csv file into a dataframe.Print the first, last 10 rows and random 20 rows. (https://www.kaggle.com/burnoutminer/heightsand-weights-dataset)

# In[57]:


import pandas as pd
df=pd.read_csv('C:\\Users\\khadi\\Downloads\\SOCR-HeightWeight.csv')
df


# In[59]:


df.head(10)


# In[60]:


df.tail(20)


# 2. Write a Python program to find the shape, size, datatypes of the dataframe object.

# In[62]:


df.shape


# In[64]:


df.size


# In[69]:


df.dtypes


# 3. Write a Python program to view basic statistical details of the data.

# In[70]:


df.describe()


# 4. Write a Python program to get the number of observations, missing values and nan values.

# In[72]:


df.shape


# In[73]:


df.isnull()


# In[74]:


df.duplicated() 


# In[75]:


pd.isna(df)


# 5. Write a Python program to add a column to the dataframe “BMI” which is calculated as :weight/height2

# In[84]:


df["BMI"] = None
df.rename(columns = {'Height(Inches)':'height'}, inplace = True)
df.rename(columns = {'Weight(Pounds)':'weight'}, inplace = True)
df["BMI"] = df['weight']/df['height']
df


# 6. Write a Python program to find the maximum and minimum BMI.

# In[87]:


df.min()


# In[88]:


df.max()


# 7. Write a Python program to generate a scatter plot of height vs weight

# In[89]:


plt.scatter(df['height'],df['weight'])


# In[90]:


df.plot(kind ="scatter", x='height', y='weight')


# In[ ]:




