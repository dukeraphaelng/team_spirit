#!/usr/bin/env python
# coding: utf-8

# ## Chopping text and sampling
# Notice:
# We had already chopped the text,
# the original dataset and the sample dataset both had been uploaded to One drive,
# you don't have to run this code

# In[5]:


INPUT_FILE = "preprocessed.csv"
AUTHOR_NUM = 50
NUM_BOOKS_AUTHOR_WRITE = 20
SAMPLE_FILE_NAME = "top_30_authors_1000.csv"


# In[6]:


import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import ast


# In[3]:


filename = INPUT_FILE
preprocessed_df = pd.read_csv(filename)


# Filter

# In[4]:


preprocessed_df['author'].unique().size


# In[37]:


df = preprocessed_df[preprocessed_df.num_tokens > 0]
author_count = df['author'].value_counts()
many_works_author = author_count.sort_values()
author_count[0:30]


# In[29]:


many_works_author = many_works_author.unique()[0:30]


# In[31]:


many_works_author.size


# In[99]:


#many_works_author = many_works_author.sample(n=AUTHOR_NUM)
#many_works_author


# In[39]:


df = df[df.author.isin(author_count[0:30].index.to_numpy())].reset_index()
len(df)


#                                                             One hot

# In[50]:


df['authors_counts'] = df['author'].map(author_count[0:30])
df


# In[51]:


def onehot_to_vertor(x):
    values = list(x.values)
    return values





INPUT_FILE = "preprocessed.csv"
AUTHOR_NUM = 50
NUM_BOOKS_AUTHOR_WRITE = 20
SAMPLE_FILE_NAME = "top_30_authors_1000.csv"





filename = INPUT_FILE
preprocessed_df = pd.read_csv(filename)


# Filter




preprocessed_df['author'].unique().size





df = preprocessed_df[preprocessed_df.num_tokens > 0]
author_count = df['author'].value_counts()
many_works_author = author_count.sort_values()
print(author_count[0:30])





many_works_author = many_works_author.unique()[0:30]





many_works_author.size





#many_works_author = many_works_author.sample(n=AUTHOR_NUM)
#many_works_author





df = df[df.author.isin(author_count[0:30].index.to_numpy())].reset_index()
len(df)


#                                                             One hot




df['authors_counts'] = df['author'].map(author_count[0:30])
df


# In[10]:


def onehot_to_vertor(x):
    values = list(x.values)
    return values


# In[11]:


def save_author_order(x):
    x.to_csv('order.csv', index=False)



one_hot = pd.get_dummies(df['author'])
save_author_order(one_hot)
# convert true/false to 1/0
one_hot = one_hot.astype(int)
df['label'] = one_hot.apply(onehot_to_vertor, axis=1)
df['label']


#                                     Take the first 512 tokens
# 
#                                     Take the last 512 tokens
# 
#                                     Take the first 128 and last 384 tokens




len(df['label'].apply(str).unique())


# In[15]:


# split to 512
chunk_size = 512
sample_size = 400
chunk_df = pd.DataFrame(columns=df.columns)
count = 0
chunk_l = []
for index, row in tqdm(df.iterrows()):
    doc = row['text'].split()
    chunk_list = []
    sublist_length = 5
    if row['authors_counts'] >= 200:
        sublist_length = 5
    if 150 <= row['authors_counts'] < 200:
        sublist_length = 7
    if 100 <= row['authors_counts'] < 150:
        sublist_length = 10
    if 40 < row['authors_counts'] < 100:
        sublist_length = 25
    for i in range(0, len(doc), chunk_size):
        chunk_list.append(i)
    if len(chunk_list) < sublist_length:
        #print(row)
        count += 1
        random_chunk = chunk_list
    else:
        random_chunk = random.sample(chunk_list, sublist_length)
    for i in random_chunk:
        sub_doc = ' '.join(doc[i: i + chunk_size])
        new_row = {'label': row['label'], 'text': sub_doc, 'author': row['author']}
        chunk_l.append(new_row)
        #chunk_df = pd.concat([chunk_df,pd.DataFrame(new_row)],ignore_index=True)
print(count)
# sample 400 randomly


# In[74]:


chunk_df = pd.DataFrame(chunk_l)
chunk_df


# In[76]:


chunk_df['author'].value_counts()


# In[82]:


def sample(x):
    if x.shape[0] > 1000:
        return x.sample(n=1000, random_state=42)
    else:
        return x


#lambda x: x.sample(n=1000, random_state=42)

sample_df = chunk_df.groupby('author', group_keys=False).apply(sample)
sample_df.groupby('author', group_keys=False).count()
sample_df = sample_df.drop('author', axis=1)



# In[83]:


sample_df.to_csv('top_30_authors_1000.csv', index=False)


#                                                 Sampling

# In[26]:





# In[89]:


df = pd.read_csv(SAMPLE_FILE_NAME)


# In[90]:


def get_y(x):
    try:
        return np.asarray(ast.literal_eval(x), dtype=int)
    except ValueError:
        print(f"Error parsing: {x}")
        # You can return a default value, or re-raise the exception
        # return np.array([], dtype=int)
        raise


# In[91]:


df


# In[92]:


def sampling(df):
    #raw_train_data, raw_test_data = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['label'], random_state=42)
    raw_train_data, test_data = train_test_split(df, test_size=0.1, shuffle=True, stratify=df['label'], random_state=42)
    train_ds, val_ds = train_test_split(raw_train_data, test_size=0.111111, shuffle=True,
                                        stratify=raw_train_data['label'], random_state=42)
    """
    skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = np.asarray(raw_train_data['text'])
    y = raw_train_data['label'].apply(get_y)
    y1 = []
    for i in y:
        y1.append(i)
    y1 = np.asarray(y1)
    train_ds = pd.DataFrame(columns=['text', 'label']).reset_index(drop=True)
    val_ds = pd.DataFrame(columns=['text', 'label']).reset_index(drop=True)
    df = raw_train_data
    for train_idxs, val_idxs in skf.split(X, y1):
        for i in train_idxs:
            train_ds = train_ds.append(df.iloc[i], ignore_index=True)
            #train_ds = pd.concat([train_ds, df.iloc[i]], ignore_index=True)
        for i in val_idxs:
            val_ds = val_ds.append(df.iloc[i], ignore_index=True)
            #val_ds = pd.concat([val_ds, df.iloc[i]], ignore_index=True)
        break
    """
    return train_ds, val_ds, test_data


# 

# In[93]:


train_ds, val_ds, test_data = sampling(df)


# In[94]:


test_data['label']


# Check the number of author in each dataset

# In[95]:


def check_author_in_dataset(df):
    print("Size of Test DataSet: " + str(len(test_data)))
    print("Size of Train DataSet: " + str(len(train_ds)))
    print("Size of Validation DataSet: " + str(len(val_ds)))
    authors_in_test = test_data['label'].tolist()
    authors_in_test = [str(x) for x in authors_in_test]

    authors_in_train = train_ds['label'].tolist()
    authors_in_train = [str(x) for x in authors_in_train]

    authors_in_val = val_ds['label'].tolist()
    authors_in_val = [str(x) for x in authors_in_val]

    print("Num of authors in test: " + str(len(set(authors_in_test))))
    print("Num of authors in tran: " + str(len(set(authors_in_train))))
    print("Num of authors in validation: " + str(len(set(authors_in_val))))


# In[96]:


check_author_in_dataset(df)



# In[99]:


train_ds['label'].value_counts()


# In[100]:


test_data['label'].value_counts()


# In[101]:


val_ds['label'].value_counts()


# In[103]:


prefix = SAMPLE_FILE_NAME.split('.')[0]

test_data.to_csv(f'{prefix}_test.csv', index=False)
train_ds.to_csv(f'{prefix}_train.csv', index=False)
val_ds.to_csv(f'{prefix}_val.csv', index=False)


# In[19]:


svm_test_df = pd.read_csv('../dataset/svm_data/test.csv')
svm_train_df = pd.read_csv('../dataset/svm_data/train.csv')
svm_val_df = pd.read_csv('../dataset/svm_data/val.csv')
#svm_val_df['text_len']=svm_val_df['text'].apply(len)
#svm_val_df.sort_values(by=['text_len','author'])
#svm_data = pd.concat([svm_test_df, svm_train_df, svm_val_df], ignore_index=True)
svm_data = pd.concat([svm_test_df,svm_val_df,svm_train_df], ignore_index=True)
svm_data


# In[20]:


def get_x(x):
    return len(x.split())


svm_data['text_len'] = svm_data['text'].apply(get_x)
svm_data.groupby(by='author', group_keys=False).agg({'text_len': 'sum'}).sort_values(by='text_len')


# In[21]:


one_hot = pd.get_dummies(svm_data['author'])
save_author_order(one_hot)
# convert true/false to 1/0
one_hot = one_hot.astype(int)
svm_data['label'] = one_hot.apply(onehot_to_vertor, axis=1)
svm_data.sort_values(by='text_len')


# In[22]:


chunk_l = []
for index, row in tqdm(svm_data.iterrows()):
    doc = row['text'].split()
    chunk_list = []
    #sublist_length = 20
    for i in range(0, len(doc), chunk_size):
        sub_doc = ' '.join(doc[i: i + chunk_size])
        new_row = {'label': row['label'], 'text': sub_doc, 'author': row['author']}
        chunk_l.append(new_row)
    # random_chunk = random.sample(chunk_list, sublist_length)
    # for i in random_chunk:
    #     sub_doc = ' '.join(doc[i: i + chunk_size])
    #     new_row = {'label': row['label'], 'text': sub_doc, 'author': row['author']}
    #     chunk_l.append(new_row)
    #chunk_df = pd.concat([chunk_df,pd.DataFrame(new_row)],ignore_index=True)
svm_data2 = pd.DataFrame(chunk_l)


# In[23]:


svm_data2.value_counts('author')
#svm_data2.to_csv('svm_chopped_for_eval.csv')


# In[24]:


svm_data3 = svm_data2.groupby('author').sample(n=2000, random_state=42)
svm_data3


# In[27]:


raw_train_data, test_data = train_test_split(svm_data3, test_size=0.1, shuffle=True, stratify=svm_data3['label'], random_state=42)
train_ds, val_ds = train_test_split(raw_train_data, test_size=0.111111, shuffle=True,
                                        stratify=raw_train_data['label'], random_state=42)

print("Size of Test DataSet: " + str(len(test_data)))
print("Size of Train DataSet: " + str(len(train_ds)))
print("Size of Validation DataSet: " + str(len(val_ds)))
authors_in_test = test_data['label'].tolist()
authors_in_test = [str(x) for x in authors_in_test]

authors_in_train = train_ds['label'].tolist()
authors_in_train = [str(x) for x in authors_in_train]

authors_in_val = val_ds['label'].tolist()
authors_in_val = [str(x) for x in authors_in_val]

print("Num of authors in test: " + str(len(set(authors_in_test))))
print("Num of authors in tran: " + str(len(set(authors_in_train))))
print("Num of authors in validation: " + str(len(set(authors_in_val))))


test_data.to_csv(f'svm_2000_test.csv', index=False)
train_ds.to_csv(f'svm_2000_train.csv', index=False)
val_ds.to_csv(f'svm_2000_val.csv', index=False)







