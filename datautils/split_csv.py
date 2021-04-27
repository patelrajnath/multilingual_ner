import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

numpy.random.rand(4)

data_df = pd.read_csv('../data/wallet/wallet_train_text.txt.csv', sep='\t', index_col=None, header=0)

print(data_df.count(axis=0))
df_unique = data_df.drop_duplicates()
df_test = df_unique
df_unique = df_unique[['labels', 'text']]
print(df_unique.count(axis=0))

train, dev = train_test_split(df_unique, test_size=0.1, random_state=42, shuffle=True)


# msk = numpy.random.rand(len(df_unique)) <= 0.9
#
# train_df = df_unique[msk]
# test = df_unique[~msk]
# print(train_df.shape)
# print(test.shape)
# msk = numpy.random.rand(len(train_df)) <= 0.95
# train = train_df[msk]
# dev = train_df[~msk]
# print(train.shape)
# print(dev.shape)

train.to_csv('../data/wallet/wallet_train.csv', index=False, sep='\t')
dev.to_csv('../data/wallet/wallet_dev.csv', index=False, sep='\t')

# train.to_csv('combined_3/train_combined_3_normalized.csv', index=False, sep='\t')
# test.to_csv('combined_3/test_combined_3_normalized.csv', index=False, sep='\t')
# dev.to_csv('combined_3/dev_combined_3_normalized.csv', index=False, sep='\t')
