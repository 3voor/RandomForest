#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
# get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint

from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions
from utils import train_test_split, calculate_accuracy


# ## Load and Prepare Data

# ### Format of data
# - last column of data frame must contain the label and ir must also be called label
# - the should be no missing values in the data frame

# In[28]:


df = pd.read_csv("datasets/winequality-white.csv", sep=';')
df["label"] = df.quality
df = df.drop("quality", axis =1)

column_names = []
for column in df.columns:
    name = column.replace(" ", "_")
    column_names.append(name)

df.columns = column_names
# df.head()


# In[29]:


wine_quality = df.label.value_counts(normalize=True)
wine_quality = wine_quality.sort_index()
wine_quality.plot(kind="bar")


# In[30]:


def transform_label(value):
    if value <= 5:
        return "bad"
    else:
        return "good"
df["label"] = df.label.apply(transform_label)


# In[31]:


wine_quality = df.label.value_counts(normalize=True)
wine_quality[["bad", "good"]].plot(kind="bar")
# wine_quality


# In[32]:


random.seed(0)
train_df, test_df  = train_test_split(df, test_size=0.2)


# ## Random Forest

# In[12]:


def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped


# In[13]:


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df=train_df, n_bootstrap=n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    return forest


# In[52]:


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree {}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions
    
    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions


# In[61]:


forest = random_forest_algorithm(train_df, 4, 800, 2, 4)
predictions = random_forest_predictions(test_df, forest)
accuracy = calculate_accuracy(predictions, test_df.label)
print(accuracy)

# In[55]:


# accuracy


# # In[59]:


# list(forest[0].keys())[0].split(' ')


# # In[36]:


# decision_tree_predictions(test_df, forest[0])


# # In[47]:


# def predict_example_v2(example, tree):
#     question = list(tree.keys())[0]
#     feature_name, comparison_operator, value = question.split(' ')
    
# #     ask question
#     if comparison_operator == "<=":
#         if example[feature_name] <= float(value):
#             answer = tree[question][0]
#         else:
#             answer = tree[question][1]
    
# #     feature is categorical
#     else:
#         if str(example[feature_name]) == value:
#             answer = tree[question][0]
#         else:
#             answer = tree[question][1]
        
# #     base case
#     if not isinstance(answer, dict):
#         return answer
    
# #     recursive part
#     else:
#         residual_tree = answer
#         return predict_example_v2(example, residual_tree)


# # In[45]:


# def decision_tree_predictions_v2(test_df, tree):
#     predictions = test_df.apply(predict_example_v2, args=(tree,), axis=1)
#     return predictions


# # In[48]:


# decision_tree_predictions_v2(test_df, forest[0])

