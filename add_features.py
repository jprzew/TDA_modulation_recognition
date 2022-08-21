import pandas as pd
from modurec import features

input_file = 'data/stats_train_plain_max.pkl'
output_file = 'data/stats_train_plain_max.pkl'

df = pd.read_pickle(input_file)

epsilon = 0.05

mean_lifetimes = [df.ff.create_feature('mean', n=0, dim=2),
                  df.ff.create_feature('mean', n=1, dim=2),
                  df.ff.create_feature('mean', n=0, dim=3),
                  df.ff.create_feature('mean', n=1, dim=3),
                  df.ff.create_feature('mean', n=0, dim=4),
                  df.ff.create_feature('mean', n=1, dim=4),
                  df.ff.create_feature('mean', n=0, dim=10),
                  df.ff.create_feature('mean', n=1, dim=10),
                  df.ff.create_feature('mean', n=0, dim=2, kind='abs'),
                  df.ff.create_feature('mean', n=1, dim=2, kind='abs'),
                  df.ff.create_feature('mean', n=0, dim=10, kind='abs'),
                  df.ff.create_feature('mean', n=1, dim=10, kind='abs'),
                  df.ff.create_feature('mean', n=0, dim=2, kind='phi'),
                  df.ff.create_feature('mean', n=1, dim=2, kind='phi'),
                  df.ff.create_feature('mean', n=0, dim=10, kind='phi'),
                  df.ff.create_feature('mean', n=1, dim=10, kind='phi'),
                  df.ff.create_feature('mean', n=0, dim=2, kind='abs', fil='star'),
                  df.ff.create_feature('mean', n=0, dim=2, kind='phi', fil='star'),
                  df.ff.create_feature('mean', n=0, dim=2, step=30),
                  df.ff.create_feature('mean', n=1, dim=2, step=30),
                  df.ff.create_feature('mean', n=0, dim=4, step=30),
                  df.ff.create_feature('mean', n=1, dim=4, step=30)]
                  # df.ff.create_feature('mean', n=0, dim=4, step='symbol_rate'),
                  # df.ff.create_feature('mean', n=1, dim=4, step='symbol_rate')]

counting_features = [df.ff.create_feature('no', n=1, dim=2),
                     df.ff.create_feature('no', n=1, dim=3),
                     df.ff.create_feature('no', n=1, dim=4),
                     df.ff.create_feature('no', n=1, dim=10),
                     df.ff.create_feature('no', n=0, dim=2, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=3, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=4, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=10, eps=epsilon),
                     df.ff.create_feature('no', n=0, dim=2, kind='abs', fil='star'),
                     df.ff.create_feature('no', n=0, dim=2, kind='phi', fil='star'),]
                     # df.ff.create_feature('no', n=0, dim=4, step='symbol_rate'),
                     # df.ff.create_feature('no', n=1, dim=4, step='symbol_rate')]

variance_features = [df.ff.create_feature('var', n=0, dim=2),
                     df.ff.create_feature('var', n=1, dim=2),
                     df.ff.create_feature('var', n=0, dim=3),
                     df.ff.create_feature('var', n=1, dim=3),
                     df.ff.create_feature('var', n=0, dim=4),
                     df.ff.create_feature('var', n=1, dim=4),
                     df.ff.create_feature('var', n=0, dim=10),
                     df.ff.create_feature('var', n=1, dim=10),
                     df.ff.create_feature('var', n=0, dim=2, kind='abs'),
                     df.ff.create_feature('var', n=1, dim=2, kind='abs'),
                     df.ff.create_feature('var', n=0, dim=10, kind='abs'),
                     df.ff.create_feature('var', n=1, dim=10, kind='abs'),
                     df.ff.create_feature('var', n=0, dim=2, kind='phi'),
                     df.ff.create_feature('var', n=1, dim=2, kind='phi'),
                     df.ff.create_feature('var', n=0, dim=10, kind='phi'),
                     df.ff.create_feature('var', n=1, dim=10, kind='phi'),
                     df.ff.create_feature('var', n=0, dim=2, kind='abs', fil='star'),
                     df.ff.create_feature('var', n=0, dim=2, kind='phi', fil='star'),
                     df.ff.create_feature('var', n=0, dim=2, step=30),
                     df.ff.create_feature('var', n=1, dim=2, step=30),
                     df.ff.create_feature('var', n=0, dim=4, step=30),
                     df.ff.create_feature('var', n=1, dim=4, step=30)]
                         # df.ff.create_feature('var', n=0, dim=4, step='symbol_rate'),
                     # df.ff.create_feature('var', n=1, dim=4, step='symbol_rate')]
        
entropy_features = [df.ff.create_feature('entropy', n=0, dim=2),
                    df.ff.create_feature('entropy', n=1, dim=2),
                    df.ff.create_feature('entropy', n=0, dim=3),
                    df.ff.create_feature('entropy', n=1, dim=3),
                    df.ff.create_feature('entropy', n=0, dim=4),
                    df.ff.create_feature('entropy', n=1, dim=4),
                    df.ff.create_feature('entropy', n=0, dim=10),
                    df.ff.create_feature('entropy', n=1, dim=10),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='abs'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='abs'),
                    df.ff.create_feature('entropy', n=0, dim=10, kind='abs'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='abs'),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='phi'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='phi'),
                    df.ff.create_feature('entropy', n=0, dim=10, kind='phi'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='phi'),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='abs', fil='star'),
                    df.ff.create_feature('entropy', n=0, dim=2, kind='phi', fil='star'),
                    df.ff.create_feature('entropy', n=0, dim=2, step=30),
                    df.ff.create_feature('entropy', n=1, dim=2, step=30),
                    df.ff.create_feature('entropy', n=0, dim=4, step=30),
                    df.ff.create_feature('entropy', n=1, dim=4, step=30, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=2, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=2, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=3, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=3, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=4, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=4, input='births'),
                    df.ff.create_feature('entropy', n=0, dim=10, input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=10, input='births'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='abs', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='abs', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='phi', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='phi', input='births'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='abs', input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='abs', input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=2, kind='phi', input='deaths'),
                    df.ff.create_feature('entropy', n=1, dim=10, kind='phi', input='deaths')]
        
# df.ff.create_feature('entropy', n=0, dim=20),
# df.ff.create_feature('entropy', n=1, dim=20),
     
p = 2
wasser_features=[df.ff.create_feature('wasser_ampl', n=0, p=p, dim=2),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=2),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=3),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=3),
                 df.ff.create_feature('wasser_ampl', n=0, p=p, dim=4),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=4),
                 df.ff.create_feature('wasser_ampl', n=0, p=p, dim=10),
                 df.ff.create_feature('wasser_ampl', n=1, p=p, dim=10),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=2, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=10, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=10, p=p, kind='abs'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=2, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=10, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=1, dim=10, p=p, kind='phi'),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, kind='abs', fil='star', p=p),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, kind='phi', fil='star', p=p),
                 df.ff.create_feature('wasser_ampl', n=0, dim=2, step=30, p=p),
                 df.ff.create_feature('wasser_ampl', n=1, dim=2, step=30, p=p),
                 df.ff.create_feature('wasser_ampl', n=0, dim=4, step=30, p=p),
                 df.ff.create_feature('wasser_ampl', n=1, dim=4, step=30, p=p)]

feat = mean_lifetimes + counting_features + variance_features + entropy_features + wasser_features

df.to_pickle(output_file)
