import pandas as pd
import numpy as np


def result_analysis(df):

    print('Avg Time: ', df.mean()['elapsed_time'])
    print('Learner avg: ', df.mean()['lrn_time'])
    print('Regulariser avg: ', df.mean()['reg_time'])
    print('Verifier avg: ', df.mean()['ver_time'])
    print('Trajectoriser avg: ', df.mean()['trj_time'])
    print('Max Time: ', df.max()['elapsed_time'])
    print('Min Time: ', df.min()['elapsed_time'])

    success_res = df.loc[df['found']==True]
    print('Avg success results: ', success_res.mean()['elapsed_time'])
    print('Max success results: ', success_res.max()['elapsed_time'])
    print('Number of Fail: ', len(df)-len(success_res))

    print('=== Formatted Results ===')
    # print first two digits for readability
    print(np.round(df.mean()['elapsed_time'], 2),
          '[', len(df)-len(success_res), ']',
          np.round(success_res.mean()['elapsed_time'], 2)
          )
    print(np.round(df.mean()['lrn_time'], 2),
          np.round(df.mean()['reg_time'], 2),
          np.round(df.mean()['ver_time'], 2),
          np.round(df.mean()['trj_time'], 2)
          )


res = pd.read_csv('robustness_barrier_hdn_10.csv', index_col=0)
result_analysis(res)
