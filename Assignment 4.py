#!/usr/bin/env python
# coding: utf-8

# Set A
# 1.Generate a random array of 50 integers and display them using a line chart, scatter
# plot, histogram and box plot. Apply appropriate color, labels and styling options.
# 2.Add two outliers to the above data and display the box plot.
# 3.Create two lists, one representing subject names and the other representing marks
# obtained in those subjects. Display the data in a pie chart and bar chart.
# 4.Write a Python program to create a Bar plot to get the frequency of the three species
# of the Iris data.
# 5.Write a Python program to create a Pie plot to get the frequency of the three species of
# the Iris data.
# 6.Write a Python program to create a histogram of the three species of the Iris data.

# In[4]:


#1.Generate a random array of 50 integers and display them using a line chart, scatter plot, histogram and box plot. Apply appropriate color, labels and styling options.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
 
# Creating dataset
np.random.seed(23685752)
N_points = 50
n_bins = 20
 
# Creating distribution
x = np.random.randn(N_points)
y = .8 ** x + np.random.randn(50) + 25
 
# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
 
axs.hist(x, bins = n_bins,color ='b')
 
# Show plot
plt.show()
plt.scatter(x,y,color = 'r')
plt.show()
plt.plot(y, linestyle = 'dotted', color = 'r')
plt.show()
plt.plot(ypoints, linestyle = 'dotted')
plt.show()


# In[5]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
 
# Creating dataset
np.random.seed(23685752)
N_points = 50
n_bins = 20

x = np.random.randn(N_points)
y = .8 ** x + np.random.randn(50) + 25
data = {'x': x, 'y' : y}
df = pd.DataFrame(data)
size = 50
df.boxplot( column =['x'], grid = False)


# In[7]:


#2.Add two outliers to the above data and display the box plot
df.loc[51] = 20
df.loc[52] =-11
df.boxplot( column =['x'], grid = False)


# In[9]:


#3.Create two lists, one representing subject names and the other representing marks
#obtained in those subjects. Display the data in a pie chart and bar chart
import matplotlib.pyplot as plt
import numpy as np
subject = ['Operating System','Java','Web Technology','Python','Data Science','Block Chain']
Marks = [5,46,37,45,26,47]
plt.pie(Marks, labels = subject)
plt.show() 
plt.barh(subject,Marks,color="red")
plt.show()


# In[38]:


#4.Write a Python program to create a Bar plot to get the frequency of the three species
#of the Iris data.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\khadi\\Downloads\\iris.csv")
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('Species',data=df)
plt.title("Iris Species Count")
plt.show()


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
iris  = pd.read_csv("C:\\Users\\khadi\\Downloads\\iris.csv")
ax=plt.subplots(1,1,figsize=(10,8))
iris['Species'].value_counts().plot.pie()#explode=[0.1,0.1,0.1])#,autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.title("Iris Species %")
plt.show()


# In[42]:


#6.Write a Python program to create a histogram of the three species of the Iris data.
plt.figure(figsize = (10, 7))
x = iris["SepalLengthCm"]
  
plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")


# Set B
# 1.Write a Python program to create a graph to find relationship between the petal length
# and petal width.
# 2.Write a Python program to draw scatter plots to compare two features of the iris
# dataset. 
# 3.Write a Python program to create box plots to see how each feature i.e. Sepal Length,
# Sepal Width, Petal Length, Petal Width are distributed across the three species.

# In[43]:


#1.Write a Python program to create a graph to find relationship between the petal length and petal width.
import pandas as pd
import matplotlib.pyplot as plt
iris  = pd.read_csv("C:\\Users\\khadi\\Downloads\\iris.csv")
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()


# In[69]:


#2.Write a Python program to draw scatter plots to compare two features of the iris dataset. 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

iris  = pd.read_csv("C:\\Users\\khadi\\Downloads\\iris.csv")
#Drop id column
iris = iris.drop('Id',axis=1)
#Convert Species columns in a numerical column of the iris dataframe
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
iris.Species = le.fit_transform(iris.Species)
x = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values
plt.scatter(x[:,0], x[:, 3], c=y, cmap ='flag')
plt.xlabel('Sepal Length cm')
plt.ylabel('Petal Width cm')
plt.show()


# In[16]:


#3.Write a Python program to create box plots to see how each feature i.e. Sepal Length,
#Sepal Width, Petal Length, Petal Width are distributed across the three species.
import pandas as pd
import matplotlib.pyplot as par
iris  = pd.read_csv("C:\\Users\\khadi\\Downloads\\iris.csv")
iris.columns = ['id','sepal length', 'sepal width', 'petal length', 'petal width', 'class']

iris.head()
iris.boxplot('sepal length')


# Set C
# 1.Write a Python program to create a pairplot of the iris data set and check which flower
# species seems to be the most separable.
# 2.Write a Python program to generate a box plot to show the Interquartile range and
# outliers for the three species for each feature.
# 3. Write a Python program to create a join plot using "kde" to describe individual
# distributions on the same plot between Sepal length and Sepal width. Note: The kernel
# density estimation (kde) procedure visualizes a bivariate distribution. In seaborn, this
# kind of plot is shown with a contour plot and is available as a style in joint plot(

# In[77]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris  = pd.read_csv("C:\\Users\\khadi\\Downloads\\iris.csv")
g = sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$SepalLength(Cm)$", "$SepalWidth(Cm)$") 
plt.show()


# In[88]:


#2.Write a Python program to generate a box plot to show the Interquartile range and
#outliers for the three species for each feature. 
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
# Load Iris dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
iris  = pd.read_csv("C:\\Users\\khadi\\Downloads\\iris.csv")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x='Species',y='PetalLengthCm',data=iris,order=['Iris-virginica','Iris-versicolor','Iris-setosa'],linewidth=2.5,orient='v',dodge=False)


# In[90]:


#3. Write a Python program to create a join plot using "kde" to describe individual
#distributions on the same plot between Sepal length and Sepal width. Note: The kernel
#density estimation (kde) procedure visualizes a bivariate distribution. In seaborn, this
#kind of plot is shown with a contour plot and is available as a style in joint plot().
sns.pairplot(iris,hue='Species');


# In[ ]:





# In[ ]:




