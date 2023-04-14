# %%
import pandas as pd
import numpy as np

svm_df = pd.read_csv('svm_stats.csv')
svm_df['log_loss'] = [np.nan for i in range(len(svm_df))]
transformer_df = pd.read_csv('transformer_stats.csv')
stats = pd.concat([transformer_df, svm_df], axis=0)
stats = stats.rename(columns={stats.columns[0]: 'model'})
stats = stats.reset_index(drop=True)
stats.at[0, 'model'] = 'Max-Pooling BERT'
stats.at[1, 'model'] = 'Mean-Pooling BERT'
stats = stats.set_index('model')
stats.to_csv('stats.csv')
stats


