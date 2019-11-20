#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import determine_type_of_feature
import numpy as np
import random
import pandas as pd


# ## 1. Decision Tree Helper functions

# ### 1.1 Data Pure
# Check if data is pure

# In[73]:


# check if target has more than two values
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)
    
    if len(unique_classes) == 1:
        return True
    else: 
        return False


# ### 1.2 Classify Data
# 
# Get the most common value of the target values

# In[79]:


# 
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# ### 1.3 Potentital Splits

# In[84]:


def get_potential_splits(data, random_subspace):
    #subspaces are the group of attributes or columns that can be splitted
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1)) #excluding the last columns which is the label
#     print(column_indices)
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)
#         print(column_indices)
    
    for column_index in column_indices:
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


# ### 1.4 Lowest Overall Entropy

# In[83]:


def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)
    
    probabilities = counts/counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    
    return entropy


# In[64]:


def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    
    overall_entropy = (p_data_below * calculate_entropy(data_below) + 
                       p_data_above * calculate_entropy(data_above))
    
    return overall_entropy


# In[65]:


def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy =  calculate_overall_entropy(data_below, data_above)
            
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    return best_split_column, best_split_value


# ### 1.5 Split Data

# In[82]:


def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]
    
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
#   feature is categorical
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    return data_below, data_above


# ## 2. Decision Tree Algorithm

# In[104]:


def decision_tree_algorithm(df, counter=0, min_samples = 2, max_depth = 5, random_subspace=None):
#     data preparation
    if counter==0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df
    
#     base cases
    if(check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
    
        return classification
    
#     recursive part
    else:
        counter += 1
        
#       helper functions
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
#       check for empty data
        if len(data_below) == 0 or len(data_above) ==0:
            classification = classify_data(data)
            return classification
        
#         determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
#       feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
#         instantiate sub-tree
        sub_tree = {question: []}
        
#       find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, random_subspace)
        
#         if the answers are the same, then there is no point asking the question.
#         this could happen when the data is classified even though it si not pure
#         yes (min_samples or max_depth base case)
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        return sub_tree


# ## 3. Make predictions

# ### 3.1 Make one prediction

# In[109]:


def predict_example(example, tree):
    # print(tree)
    question = list(tree.keys())[0]
    # print(question)
    feature_name, comparison_operator, value = question.split(' ')
    
#     ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
#     feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]
        
#     base case
    if not isinstance(answer, dict):
        return answer
    
#     recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)


# ### 3.2 All examples of the test data

# In[110]:


def decision_tree_predictions(test_df, tree):
    # print(tree)
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions

