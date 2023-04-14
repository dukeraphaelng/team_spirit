# %%

import pandas as pd
import numpy as np

from pathlib import Path
import os
from data_io import get_book

# %%
metadata_filename = 'metadata.csv'
counts_dirname = 'counts'
tokens_dirname = 'tokens'

metadata_df = pd.read_csv(metadata_filename)

filtered_df = metadata_df[(metadata_df.language == "['en']") & (metadata_df.type == 'Text')]

SELECTED_COLUMNS = ['id', 'title', 'author', 'authoryearofbirth', 'authoryearofdeath']
filtered_df = filtered_df.dropna(subset=SELECTED_COLUMNS)
filtered_df = filtered_df[SELECTED_COLUMNS]
filtered_df = filtered_df.reset_index(drop=True)

author_count = filtered_df['author'].value_counts()
many_works_author = author_count[author_count >= 50]
filtered_df = filtered_df[filtered_df.author.isin(many_works_author.index.to_numpy())].reset_index()

'PG8700' in filtered_df.id

# %%
#filtered_df = filtered_df.sample(n=50, random_state=2).reset_index()

sampled_authors = filtered_df.author.sample(n=50, random_state=1)
# Some author names are duplicated, the set contains 30 authors
print(len(list(sampled_authors)))
print(len(set(sampled_authors)))
train_ids = []
test_ids = []
val_ids = []

# Split train, test, validation, total 50 books per author, total 30 authors
# 60% = 30 books into training set
# 30% = 15 books into test set
# 10% = 5 books into validation set
print(len(set(sampled_authors)))
for author in set(sampled_authors):
    works = filtered_df[filtered_df.author == author].sample(n=50, random_state=1)
    works_list = list(works.id)
    train_id, test_id, val_id = works_list[:30], works_list[30:45], works_list[45:]
    
    # Does not check if this file exists and is valid
    train_ids.extend(train_id)
    test_ids.extend(test_id)
    val_ids.extend(val_id)

train_df = filtered_df[filtered_df.id.isin(train_ids)]
test_df = filtered_df[filtered_df.id.isin(test_ids)]
val_df = filtered_df[filtered_df.id.isin(val_ids)]

df_arrs = []
for df in [train_df, test_df, val_df]:
    df = df[['id', 'author']]

    docs = []
    docs_unavail_pg_ids = []
    for pg_id in df.id:    
        try:
            doc = ' '.join(get_book(pg_id, os.path.join(tokens_dirname), level='tokens'))
            docs.append(doc)
        except:
            docs_unavail_pg_ids.append(pg_id)
    
    df = df[~df.id.isin(docs_unavail_pg_ids)].reset_index()
    df.insert(2, 'text', docs, True)
    
    df_arrs.append(df)

train_df, test_df, val_df = df_arrs


