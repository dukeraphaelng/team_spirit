# Original source: https://github.com/pgcorpus/gutenberg-analysis/blob/master/src/data_io.py

"""Functions to handle data I/O."""

import numpy as np
import os

def get_book(pg_id, path_read, level = 'counts'):
    '''
    Retrieve the data from a single book.
    pg_id, str: id of book in format 'PG12345'
    OPTIONAL:
    
    level, which granularity
        - 'counts', dict(word,count) [default]
        - 'tokens', list of tokens (str)
        - 'text', single str
    '''

    if level == 'counts':
        ## counts -- returns a dictionary {str:int}
        fname_read = '%s_counts.txt'%(pg_id)
        filename = os.path.join(path_read,fname_read)
        with open(filename,'r') as f:
            x = f.readlines()

        words = [h.split()[0] for h in x]
        counts = [int(h.split()[1]) for h in x]
        dict_word_count = dict(zip(words,counts))
        return dict_word_count

    elif level == 'tokens':
        ## tokens --> returns a list of strings 
        fname_read = '%s_tokens.txt'%(pg_id)
        filename = os.path.join(path_read,fname_read)
        with open(filename,'r') as f:
            x = f.readlines()
        list_tokens = [h.strip() for h in x]
        return list_tokens[0:512]
    
    # Join tokens here
#     elif level == 'text':
#         ## text --> returns a string 
#         path_read = os.path.join(path_gutenberg,'data','text')
#         fname_read = '%s_text.txt'%(pg_id)
#         filename = os.path.join(path_read,fname_read)
#         with open(filename,'r') as f:
#             x = f.readlines()
#         text =  ' '.join([h.strip() for h in x])
#         return text

    else:
        print('ERROR: UNKNOWN LEVEL')
        return None

def get_dict_words_counts(filename):
    """
    Read a file and make a dictionary with words and counts.
    Parameters
    ----------
    filename : str
        Path to file.
    Returns
    -------
     : dict
        Dictionary with words in keys and counts in values.
    """
    with open(filename, 'r') as f:
        x = f.readlines()
    if x[0] == '\n':
        # an empty book
        words = []
        counts = []
    else:
        words = [h.split()[0] for h in x]
        counts = [int(h.split()[1]) for h in x]
    return dict(zip(words, counts))


def get_p12_same_support(
        dict_wc1,
        dict_wc2):
    """
    Get probabilities with common support.
    For two dictionaries of the form {word:count},
    make two arrays p1 and p2 holding probabilites
    in which the two distributions have the same support.
    Parameters
    -----------
    dict_wc1, dict_wc2 : dict
        Dictionaries of the form {word: count}.
    Returns
    -------
    arr_p1, arr_p2 : np.array (float)
        Normalized probabilites with common support.
    """
    N1 = sum(list(dict_wc1.values()))
    N2 = sum(list(dict_wc2.values()))
    # union of all words sorted alphabetically
    words1 = list(dict_wc1.keys())
    words2 = list(dict_wc2.keys())
    words_12 = sorted(list(set(words1).union(set(words2))))
    V = len(words_12)
    arr_p1 = np.zeros(V)
    arr_p2 = np.zeros(V)
    for i_w, w in enumerate(words_12):
        try:
            arr_p1[i_w] = dict_wc1[w]/N1
        except KeyError:
            pass
        try:
            arr_p2[i_w] = dict_wc2[w]/N2
        except KeyError:
            pass
    return arr_p1, arr_p2