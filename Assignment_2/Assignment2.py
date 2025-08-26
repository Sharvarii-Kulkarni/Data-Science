#!/usr/bin/env python
# coding: utf-8

# # SET A:
# 1. Write a Python program to find the maximum and minimum value of a given flattened array.
# Expected Output:Original flattened array:
# [[0 1]
# [2 3]]
# Maximum value of the above flattened array:3
# Minimum value of the above flattened array:0
# 2. Write a python program to compute Euclidian Distance between two data points in a dataset. [Hint: Use linalgo.norm function from NumPy]
# 3. Create one dataframe of data values. Find out mean, range and IQR for this data.
# 4. Write a python program to compute sum of Manhattan distance between all pairs of points.
# 
# 5. Write a NumPy program to compute the histogram of nums against the bins. Sample Output:
# nums: [0.5 0.7 1. 1.2 1.3 2.1]
# bins: [0 1 2 3]
# Result: (array([2, 3, 1], dtype=int64), array([0, 1, 2, 3]))
# 6. Create a dataframe for students’ information such name, graduation percentage and age.
# Display average age of students, average of graduation percentage. And, also describe all basic statistics of data. (Hint: use describe()). 

# 1. Write a Python program to find the maximum and minimum value of a given flattened array.

# In[3]:


import numpy as np
a = np.arange(4).reshape((2,2))
print("Original flattened array:")
print(a)
print("Maximum value of the above flattened array:")
print(np.amax(a))
print("Minimum value of the above flattened array:")
print(np.amin(a))


# Write a python program to compute Euclidian Distance between two data points in a dataset. [Hint: Use linalgo.norm function from NumPy]

# In[4]:


import numpy as np
 
# initializing points in
# numpy arrays
point1 = np.array((1, 2, 3))
point2 = np.array((1, 1, 1))
 
# calculating Euclidean distance
# using linalg.norm()
dist = np.linalg.norm(point1 - point2)
 
# printing Euclidean distance
print(dist)


# 3. Create one dataframe of data values. Find out mean, range and IQR for this data.

# In[13]:


import numpy as np
import pandas as pd
#define array of data
data = np.array([14, 19, 20, 22, 24, 26, 27, 30, 30, 31, 36, 38, 44, 47])

#calculate interquartile range 
q3, q1 = np.percentile(data, [75 ,25])
iqr = q3 - q1

#display interquartile range 
iqr
12.25

df = pd.DataFrame(data)
df.mean(axis=0)


# 4. Write a python program to compute sum of Manhattan distance between all pairs of points.

# In[4]:


from math import sqrt

#create function to calculate Manhattan distance 
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))
 
#define vectors
A = [2, 4, 4, 6]
B = [5, 5, 7, 8]

#calculate Manhattan distance between vectors
manhattan(A, B)


# 5. Write a NumPy program to compute the histogram of nums against the bins. Sample Output:
# nums: [0.5 0.7 1. 1.2 1.3 2.1]
# bins: [0 1 2 3]
# Result: (array([2, 3, 1], dtype=int64), array([0, 1, 2, 3]))

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
nums = np.array([0.5, 0.7, 1.0, 1.2, 1.3, 2.1])
bins = np.array([0, 1, 2, 3])
print("nums: ",nums)
print("bins: ",bins)
print("Result:", np.histogram(nums, bins))
plt.hist(nums, bins=bins)
plt.show()


# 6. Create a dataframe for students’ information such name, graduation percentage and age.
# Display average age of students, average of graduation percentage. And, also describe all basic statistics of data. (Hint: use describe()). 

# In[22]:


import pandas as pd
import numpy as np
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print("\nMean score for each different student in data frame:")
print(df['score'].mean)
df.describe()


# SET B:
# 1. Download iris dataset file. Read this csv file using read_csv() function. Take samples from entire dataset. Display maximum and minimum values of all numeric attributes.
# 2. Continue with above dataset, find number of records for each distinct value of class attribute. Consider entire dataset and not the samples.
# 3. Display column-wise mean, and median for iris dataset from Q.4 (Hint: Use mean() and median() functions of pandas dataframe. 

# In[29]:


import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
print(df.head())
df.max()


# In[28]:


df.min()


# In[38]:


n = len(pd.unique(df['SepalLengthCm']))
print("No.of.unique SepalLengthCm values :",n)
n1 = len(pd.unique(df['SepalWidthCm']))
print("No.of.unique SepalWidthCm values :",n1)
n2 = len(pd.unique(df['PetalLengthCm']))
print("No.of.unique PetalLengthCm values :",n2)
n3 = len(pd.unique(df['PetalWidthCm']))
print("No.of.unique PetalWidthCm values :",n3)
n4 = len(pd.unique(df['Species']))
print("No.of.unique Species values :",n4)


# In[44]:


df.mean()


# In[45]:


df.median()


# # SET C:
# 1. Write a python program to find Minkowskii Distance between two points.
# 2. Write a Python NumPy program to compute the weighted average along the specified
# axis of a given flattened array.
# From Wikipedia: The weighted arithmetic mean is similar to an ordinary arithmetic
# mean (the most common type of average), except that instead of each of the data points
# contributing equally to the final average, some data points contribute more than others.
# The notion of weighted mean plays a role in descriptive statistics and also occurs in a
# more general form in several other areas of mathematics.
# Sample output:
# Original flattened array:
# [[0 1 2]
# [3 4 5]
# [6 7 8]]
# Weighted average along the specified axis of the above flattened array:
# [1.2 4.2 7.2]
# 3. Write a NumPy program to compute cross-correlation of two given arrays.
# Sample Output:
# Original array1:
# [0 1 3]
# Original array2:
# [2 4 5]
# Cross-correlation of the said arrays:
# [[2.33333333 2.16666667]
# [2.16666667 2.33333333]]
# 4. Download any dataset from UCI (do not repeat it from set B). Read this csv file using
# read_csv() function. Describe the dataset using appropriate function. Display mean
# value of numeric attribute. Check any data values are missing or not.
# 5. Download nursery dataset from UCI. Split dataset on any one categorical attribute.
# Compare the means of each split. (Use groupby)
# 6. Create one dataframe with 5 subjects and marks of 10 students for each subject. Find
# arithmetic mean, geometric mean, and harmonic mean.
# 7. Download any csv file of your choice and display details about data using pandas
# profiling. Show stats in HTML form

# 1. Write a python program to find Minkowskii Distance between two points.

# In[46]:


from math import *
from decimal import Decimal
def my_p_root(value, root):
   my_root_value = 1 / float(root)
   return round (Decimal(value) **
   Decimal(my_root_value), 3)
def my_minkowski_distance(x, y, p_value):
   return (my_p_root(sum(pow(abs(a-b), p_value)
      for a, b in zip(x, y)), p_value))
# Driver Code
vector1 = [0, 2, 3, 4]
vector2 = [2, 4, 3, 7]
my_position = 5
print("The Distance is::",my_minkowski_distance(vector1, vector2, my_position))


# 2. Write a Python NumPy program to compute the weighted average along the specified
# axis of a given flattened array.
# From Wikipedia: The weighted arithmetic mean is similar to an ordinary arithmetic
# mean (the most common type of average), except that instead of each of the data points
# contributing equally to the final average, some data points contribute more than others.
# The notion of weighted mean plays a role in descriptive statistics and also occurs in a
# more general form in several other areas of mathematics.
# Sample output:
# Original flattened array:
# [[0 1 2]
# [3 4 5]
# [6 7 8]]
# Weighted average along the specified axis of the above flattened array:
# [1.2 4.2 7.2]

# In[47]:


import numpy as np
a = np.arange(9).reshape((3,3))
print("Original flattened array:")
print(a)
print("Weighted average along the specified axis of the above flattened array:")
print(np.average(a, axis=1, weights=[1./4, 2./4, 2./4]))


# 3. Write a NumPy program to compute cross-correlation of two given arrays.
# Sample Output:
# Original array1:
# [0 1 3]
# Original array2:
# [2 4 5]
# Cross-correlation of the said arrays:
# [[2.33333333 2.16666667]
# [2.16666667 2.33333333]]

# In[48]:


import numpy as np
x = np.array([0, 1, 3])
y = np.array([2, 4, 5])
print("\nOriginal array1:")
print(x)
print("\nOriginal array1:")
print(y)
print("\nCross-correlation of the said arrays:\n",np.cov(x, y))


# 4. Download any dataset from UCI (do not repeat it from set B). Read this csv file using
# read_csv() function. Describe the dataset using appropriate function. Display mean
# value of numeric attribute. Check any data values are missing or not.

# In[3]:


import pandas as pd
df = pd.read_csv("C:\\Users\\khadi\\Downloads\\Coursera.csv")
df


# In[4]:


df.isnull()


# In[10]:


df.describe()


# 5. Download nursery dataset from UCI. Split dataset on any one categorical attribute. Compare the means of each split. (Use groupby)

# In[2]:


import pandas as pd
df = pd.read_csv("C:\\Users\\khadi\\Downloads\\nursery.data")
df


# In[8]:


df.groupby('usual')
df


# In[12]:


grouped = df.groupby("nonprob", axis="columns")


# 6. Create one dataframe with 5 subjects and marks of 10 students for each subject. Find arithmetic mean, geometric mean, and harmonic mean. 

# In[8]:


from scipy.stats import hmean
import pandas as pd
# define the dataset

data ={'StudentName': ['rama','shama','radha','dhara','arti','priti','sapna','komal','rima','neha'],
       'Maths': [56,67,78,89,90,56,67,78,89,90],
      'English': [77,88,77,88,99,66,45,88,77,77],
      'Science' : [66,88,99,88,77,66,88,99,88,77],
       'History' : [77,88,77,88,99,66,45,88,77,77],
       'Geography' : [77,88,77,88,99,66,45,88,77,77]
      }
df = pd.DataFrame(data)
df
df.mean()


# In[24]:


hmean(data['Maths'])


# In[25]:


from scipy.stats import gmean
gmean(data['Maths'])


# 7. Download any csv file of your choice and display details about data using pandas profiling. Show stats in HTML form.
# pip install pandas-profiling
# It can also be installed via Conda package manager too:
# 
# conda env create -n pandas-profiling
# conda activate pandas-profiling
# conda install -c conda-forge pandas-profiling

# In[27]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df)
profile


# In[ ]:





# In[ ]:




