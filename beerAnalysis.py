#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


# ## Reading Dataset

# In[2]:


recipes = pd.read_csv('recipeData.csv', index_col='BeerID', encoding='latin1')
recipes


# ### Filtering the dataset by Irish Beer Types

# In[3]:


Irish_Styles = np.array(['Irish Dry Stout', 'Irish Extra Stout', 'Irish Red Ale', 'Irish Pale Ale', 'Irish Cream Ale', 'Irish Lager'])


# In[4]:


is_irish = recipes.Style.isin(Irish_Styles)
irish_recipes = recipes[is_irish]
irish_recipes


# ## Cleaning Data
# 
# ### Converting 'N/A' values to null values

# In[5]:


irish_recipes.replace('N/A', 'NA')
irish_recipes.info()


# ### Summary Statistics

# In[6]:


irish_recipes.describe().T


# ### % of null values in each column

# In[7]:


irish_recipes.isnull().sum()/3381*100


# ### Dropping coulmns with >25% null values

# In[8]:


irish_recipes = pd.DataFrame(irish_recipes)
irish_recipes = irish_recipes.drop(columns=['MashThickness', 'PitchRate', 'PrimaryTemp', 'PrimingMethod', 'PrimingAmount', 'UserId'])

# also dropping other unnecessary columns
irish_recipes = irish_recipes.drop(columns=['URL', 'StyleID'])


# In[9]:


irish_recipes.info()


# ### Replacing null values 

# In[10]:


replace_value = np.nanmedian(irish_recipes['BoilGravity'])
irish_recipes['BoilGravity'].fillna(replace_value, inplace = True)
irish_recipes.info()


# In[11]:


np.count_nonzero(irish_recipes['SugarScale'] == 'Plato')


# ### Extracting rows with Specific Gravity units

# In[12]:


is_sg = (irish_recipes['SugarScale'] == 'Specific Gravity')
irish_recipes = irish_recipes[is_sg]


# In[13]:


irish_recipes.shape


# In[14]:


irish_recipes.describe().T


# ### Storing the summary to a csv file using NumPy

# In[15]:


summary = irish_recipes.describe().T
#summary.insert(0, "category", summary.index)
summary.index
np.savetxt('summary.csv', summary, fmt='%.4f', delimiter=',', newline='\n', header=str(summary.columns), footer='', comments='# ', encoding='latin1')


# ### Storing the cleaned data to a csv file using Pandas

# In[16]:


irish_recipes.to_csv('irishRecipes.csv')


# ## Visualizations
# 
# ### Brew Methods Pie Chart

# In[17]:


irish_recipes.BrewMethod.value_counts().plot(kind="pie")
plt.title("Brew method distribution")


# ### Regression Plot for BoilSize vs. BoilTime

# In[18]:


df_boil = irish_recipes[(irish_recipes['BoilSize']<=100) & (irish_recipes['BoilTime'])]

plt.figure(figsize=(16,8))
sn.regplot(df_boil['BoilSize'],df_boil['BoilTime'])
plt.title('BoilSize and BoilTime relation', fontsize=20)
plt.xlabel('BoilSize', fontsize=15)
plt.ylabel('BoilTime', fontsize=15)
plt.show()


# ### Histograms

# In[19]:


plt.figure(figsize=(18,6))
count=0
for col, color in zip(['Efficiency', 'Color'],['c','y']):
    count+=1
    plt.subplot(1,2,count)
    sn.distplot(irish_recipes[col], bins=25, label=col, color=color)
    plt.title('{} Distribution'.format(col), fontsize=15)
    plt.legend()
    plt.ylabel('Normed Frequency', fontsize=15)
    plt.xlabel('{}'.format(col), fontsize=15)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# ### Logarithmic Histograms

# In[20]:


plt.figure(figsize=(12,12))
count=0
for col, color in zip(['OG', 'FG', 'BoilGravity','IBU','ABV'],['b','m','g','r','k']):
    count+=1
    if(count==5):
        plt.subplot(3,2,(5,6))
    else:
        plt.subplot(3,2,count)
    sn.distplot(np.log1p(irish_recipes[col]), bins=100, label=col, color=color)
    plt.title('Log(1 + {}) Distribution'.format(col), fontsize=15)
    plt.legend()
    plt.ylabel('Normed Frequency', fontsize=15)
    plt.xlabel('Log(1 + {})'.format(col), fontsize=15)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# ### Correlations with Heat Map

# In[21]:


f,ax = plt.subplots(figsize=(12, 8))
sn.heatmap(irish_recipes.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)


# ### Violinplots on %ABV by Beer Styles

# In[22]:


abv_df = irish_recipes[irish_recipes['ABV'] <= 25]

fig, ax = plt.subplots(1, 1, figsize=[20, 10])
sn.violinplot(x='Style',y='ABV', data=abv_df, ax=ax)
ax.set_xlabel('Beer Style')
ax.set_title('% Alcohol by Vol')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.show()

