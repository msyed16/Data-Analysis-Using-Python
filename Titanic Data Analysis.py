
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


titanic_df=pd.read_csv('train.csv')


# In[3]:


## Preview of the Data


# In[4]:


titanic_df.head()


# In[5]:


#overview of info


# In[6]:


titanic_df.info()


# In[18]:


sns.catplot('Sex', data=titanic_df, kind='count')


# In[ ]:


sns.catplot('Sex', data=titanic_df, kind='count')


# In[20]:


sns.catplot('Pclass',data=titanic_df,kind='count', hue='Sex')


# In[26]:


# how do we know about children?

def typeofperson(passenger):
    age,sex = passenger
    if age<16:
        return 'child'
    else:
        return sex
    
#to implement this into the DataFrame
titanic_df['person']= titanic_df[['Age', 'Sex']].apply(typeofperson, axis=1)

#to view 
titanic_df[:10]


# In[27]:


sns.catplot('Pclass',data=titanic_df, kind='count', hue='person')


# In[31]:


titanic_df['Age'].max()


# In[32]:


## given that the ,ax age is 80 I'll set 80 columns for age
titanic_df['Age'].hist(bins=80)


# In[33]:


#How many Women Men and Children were on the Titanic????
titanic_df['person'].value_counts()


# In[39]:


fig = sns.FacetGrid(titanic_df, hue="Sex", aspect=5)

fig.map(sns.kdeplot,'Age',shade= True)

# Set the x max limit by the oldest passenger
oldest = titanic_df['Age'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()


# In[40]:


fig = sns.FacetGrid(titanic_df, hue="person", aspect=5)

fig.map(sns.kdeplot,'Age',shade= True)

# Set the x max limit by the oldest passenger
oldest = titanic_df['Age'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()


# In[41]:


fig = sns.FacetGrid(titanic_df, hue="Pclass", aspect=5)

fig.map(sns.kdeplot,'Age',shade= True)

# Set the x max limit by the oldest passenger
oldest = titanic_df['Age'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()


# In[44]:


#there are many unknown cabin values  
#so I'll temporalily drop them and place them in a new category
deck= titanic_df['Cabin'].dropna()
deck.head()


# In[54]:


#to remove excess and useless info
levels=[]
for level in deck:
    levels.append(level[0])
    #takes the first letter value in the data
    
levels=levels.sort()


# In[56]:


#create a new DF for cabin
cabin_df=DataFrame(levels)
cabin_df.columns = ['Cabin']
#remove the anomaly
cabin_df = cabin_df[cabin_df.Cabin != 'T']
#graph it
sns.catplot('Cabin',data=cabin_df, kind='count')


# In[57]:


# where did the passengers depart from?

sns.catplot('Embarked', data=titanic_df, hue='Pclass', kind='count')


# In[65]:


#factors in survival rate

titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

sns.catplot('Survivor', data=titanic_df, kind='count', )


# In[72]:


sns.catplot('Pclass', 'Survived', data=titanic_df,kind='point')


# In[73]:


sns.catplot('Pclass', 'Survived', data=titanic_df, hue='person',kind='point')


# In[83]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[84]:


sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='summer')


# In[85]:


#the graph above looks kinda messy
#time to clean it up 
decade=[10,20,30,40,50,60,70,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=decade)


# In[86]:


#lets swtich class with sex and see what happens

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=decade)

