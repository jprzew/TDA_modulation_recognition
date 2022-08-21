import pandas as pd

input_file = 'data/stats_train_plain.pkl'

# dir(pd)

df = pd.read_pickle(input_file)
print(df.columns)
