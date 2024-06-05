# %%
import pandas as pd

data = pd.read_csv('./RazoMejia2018_data.csv')
data['repressors'] *= 2
data.drop(columns=['mean_YFP_A', 'mean_YFP_bgcorr_A',
          'date', 'username'], inplace=True)
data.rename(columns={'fold_change_A': 'fold_change'}, inplace=True)

# Split into a training and test set
train = data[(data['rbs'] == 'RBS1027') & (data['operator'] == 'O2')]
test = data[~((data['rbs'] == 'RBS1027') & (data['operator'] == 'O2'))]
train.to_csv('./RazoMejia2018_train.csv', index=False)
test.to_csv('./RazoMejia2018_test.csv', index=False)

# %%
# Aggregate and compute mean fc for each induction condition
train_agg = train.groupby(['repressors', 'IPTG_uM'],
                          as_index=False).agg({'fold_change': 'mean'})
train_agg.to_csv('./RazoMejia2018_train_agg.csv', index=False)
