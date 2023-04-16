# %% [markdown]
# # Explorative Data Analysis

# %% [markdown]
# ## 1. Preprocessing

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from pathlib import Path
import os
from data_io import get_book
import sys
import math

# %% [markdown]
# Change the filenames here based on what you have downloaded

# %%
# metadata_filename = 'SPGC-metadata-2018-07-18.csv'
# counts_dirname = 'SPGC-counts-2018-07-18'
# tokens_dirname = 'SPGC-tokens-2018-07-18'

metadata_filename = 'metadata.csv'
counts_dirname = 'counts'
tokens_dirname = 'tokens'

metadata_df = pd.read_csv(metadata_filename).set_index('id')

metadata_df

# %%
metadata_df.info()

# %%
var = 'type'

sns.countplot(data=metadata_df, x=var, order=metadata_df[var].value_counts().index)
plt.title(f'Frequency of {var}')
plt.xticks(rotation=100000000)
plt.show()

# %%
metadata_df.type.unique()

# %% [markdown]
# We can see that there are other data types than 'Text', we only want 'Text'

# %% [markdown]
# We want to see how many documents are there in each language. First we filter out all languages with too few documents

# %%
s = metadata_df['language'].value_counts()
language_df = metadata_df[metadata_df.language.isin(s[s > 10].index)]

# %%
var = 'language'
sns.countplot(data=language_df, x=var, order=language_df[var].value_counts().index)
plt.title(f'Frequency of {var}')
plt.xticks(rotation=100000000)
plt.show()

# %% [markdown]
# We can see that there are a lot of different languages for the books. This project will only utilise the English documents. But it could be well adapted to a multi-lingual problems, or for a language other than English

# %%
metadata_df = metadata_df[(metadata_df.language == "['en']") & (metadata_df.type == 'Text')]

# %%
metadata_df[(metadata_df.author == 'Nietzsche, Friedrich Wilhelm') & (metadata_df.title == 'Thus Spake Zarathustra: A Book for All and None')]

# %% [markdown]
# We also notice that this work by Nietzsche is originally in German, and it is a translated work. However, the name of the translator is not visible. This is a limitation for our data, that we have to take into consideration

# %% [markdown]
# Another limitation that can also be observed is that there are multiple works whose authors are numerous but only one of which is given authorship. For example, 'The Federalist Papers' has three authors as recorded by Gutenberg. However, only John Jay is given authorship. Another example is that 'The Communist Manifesto' has two authors: Karl Marx and Friedrich Engels, but only Engels' name is recorded. This is another limitation of the SGPC. https://www.gutenberg.org/ebooks/1404

# %%
metadata_df[metadata_df['title'].isin(['The Federalist Papers', 'The Communist Manifesto'])]

# %% [markdown]
# We now need to keep only the entries that have both the counts and the tokens file

# %%
file_not_exist = set()

for id_ in metadata_df.index:
    try:
        open(f'{counts_dirname}/{id_}_counts.txt')
        open(f'{tokens_dirname}/{id_}_tokens.txt')
    except FileNotFoundError:
        file_not_exist.add(id_)

print(f'Number of ids that do not have a corresponding counts or tokens file is: {len(file_not_exist)}')

# %% [markdown]
# Now we can drop all those entries

# %%
metadata_df = metadata_df[~metadata_df.index.isin(file_not_exist)]

# %% [markdown]
# We can now remove the columns (language, downloads, subjects, type) as they are not relevant to our modelling, and drop all rows that have a NULL values for any of the remaining columns 

# %%
SELECTED_COLUMNS = ['title', 'author', 'authoryearofbirth', 'authoryearofdeath']
metadata_df = metadata_df.dropna(subset=SELECTED_COLUMNS)
metadata_df = metadata_df[SELECTED_COLUMNS]
metadata_df['idx'] = range(0, len(metadata_df))

metadata_df

# %% [markdown]
# Now we will put the counts, the total number of tokens, and all the tokens into a dictionary for lookup in further EDA analysis and training later on. We will also remove anything that cannot be parsed correctly

# %%
num_indices = len(metadata_df)

error_ids = {'int_idx': set(), 'str_idx': set()}
word_freq = {}
tokens_doc = {'id': metadata_df.index.to_numpy(dtype=object), 'num_tokens': np.zeros(num_indices, dtype=np.ushort), 'num_unique': np.zeros(num_indices, dtype=np.ushort), 'text': np.empty(num_indices, dtype=object)}

def get_counts_num_tokens(df):
    for int_idx , str_idx in enumerate(df.index):
        try:
            cur_counts = get_book(str_idx, os.path.join(counts_dirname), level='counts')

            if not cur_counts:
                raise Exception('cur_counts is empty')

            word_freq.update(cur_counts)

            tokens_doc['num_unique'][int_idx] = len(cur_counts.keys())
            tokens_doc['num_tokens'][int_idx] = np.sum(np.fromiter(cur_counts.values(), dtype=int))

            tokens_doc['text'][int_idx] = ' '.join(get_book(str_idx, os.path.join(tokens_dirname), level='tokens'))
            
        except Exception as e:
            print(f'Error at {str_idx}: ', e)
            error_ids['int_idx'].add(int_idx)
            error_ids['str_idx'].add(str_idx)

# %%
get_counts_num_tokens(metadata_df[0:10000])

# %%
get_counts_num_tokens(metadata_df[10000:20000])

# %%
get_counts_num_tokens(metadata_df[20000:])

# %%
for k, v in tokens_doc.items():
    tokens_doc[k] = np.delete(v, list(error_ids['int_idx']), 0)
metadata_df = metadata_df[~metadata_df.index.isin(error_ids['str_idx'])]

tokens_df = pd.DataFrame.from_dict(tokens_doc).set_index('id')
metadata_df = pd.merge(metadata_df, tokens_df, left_index=True, right_index=True)

# %% [markdown]
# Now we try to get all of the duplicated books which have the same title and author, and remove the ones with less number of tokens

# %%
# https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python
titles = metadata_df["title"]
titles_authors = metadata_df[["title", "author"]]
duplicated_entries = metadata_df[titles.isin(titles[titles_authors.duplicated()])].sort_values(["title", "num_tokens"])
duplicated_entries

# %%
unique_highest_num_token = duplicated_entries[['title', 'author', 'num_tokens']].groupby(['title', 'author']).idxmax()
unique_highest_num_token

# %%
non_unique_highest_num_token = duplicated_entries[~duplicated_entries.index.isin(unique_highest_num_token.num_tokens)]
non_unique_highest_num_token

# %%
metadata_df = metadata_df[~metadata_df.index.isin(non_unique_highest_num_token.index)]

# %% [markdown]
# We confirm that we have selected the correct rows

# %%
metadata_df[metadata_df.title =='"Captains Courageous": A Story of the Grand Banks']

# %%
metadata_df

# %% [markdown]
# This is the entry with the PGID which has the highest number of tokens

# %% [markdown]
# **Export preprocessed dataframe**

# %%
metadata_df.to_csv('preprocessed.csv')

# %% [markdown]
# ## 2. Data Analysis

# %% [markdown]
# ### a. Text

# %%
plt.hist(metadata_df.num_tokens, bins=10)
plt.title('Histogram of No. of Tokens Per Document Over No. of Docs')
plt.show()

# %% [markdown]
# We can see that most documents have the number of tokens around 10^5, or otherwise 10^4

# %%
plt.hist(metadata_df.num_unique, bins=10)
plt.title('Histogram of No. of Unique Tokens Per Document Over No. of Docs')
plt.show()

# %% [markdown]
# We can see that most documents have the number of unique tokens a little bit less than 10^4

# %%
# https://stackoverflow.com/questions/43145199/create-wordcloud-from-dictionary-values
wc = WordCloud(background_color="white",width=1000,height=1000, max_words=50,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(word_freq)
plt.title('Word Cloud of 50 Most Frequently Used Words')
plt.imshow(wc)

# %% [markdown]
# ### b. Author

# %%
plt.hist(metadata_df['author'].value_counts(), bins=10)
plt.title('Number of works per author')
plt.show()

# %%
metadata_df[['authoryearofbirth', 'authoryearofdeath']].describe()

# %%
continuous_vars = ['authoryearofbirth', 'authoryearofdeath']
for var in continuous_vars:
    sns.histplot(data=metadata_df, x=var, kde=True)
    plt.title(f'Distribution of {var}')
    plt.show()

# %%
sns.lineplot(x = metadata_df['authoryearofbirth'].value_counts().index, y=metadata_df['authoryearofbirth'].value_counts())
plt.title('No. Books in English by Year of Birth of Authors')
plt.xlabel('Year of Birth of Author')
plt.ylabel('Number of Works')

plt.show()

# %% [markdown]
# We can see that most English books in the dataset are written by people born around the 1800

# %%
ana_year_df = metadata_df[['author', 'authoryearofbirth']]
ana_year_df.insert(2, 'generation', ana_year_df['authoryearofbirth'].apply(lambda x: math.floor(x * 0.1)*10), True)

counts = ana_year_df.groupby('generation')['author'].count().sort_values(ascending=False).head(30)
sns.lineplot(x = counts.index, y=counts)
plt.title('How many authors are in a given generation')
plt.xlabel('Generation')
plt.ylabel('Number of authors')
plt.show()

# %% [markdown]
# By extension, the generation with the most amount of authors are in the late 1800s


