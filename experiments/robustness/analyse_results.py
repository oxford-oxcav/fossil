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

    success_res = df.loc[df['found_bc']==True]
    print('Avg success results: ', success_res.mean()['elapsed_time'])
    print('Max success results: ', success_res.max()['elapsed_time'])
    print('Number of Fail: ', len(df)-len(success_res))

    print('=== Formatted Results ===')

    print(np.round(df.mean()['elapsed_time'], 2),
          '[',len(df)-len(success_res),']',
          np.round(success_res.mean()['elapsed_time'], 2)
          )
    print(np.round(df.mean()['lrn_time'], 2),
          np.round(df.mean()['reg_time'], 2),
          np.round(df.mean()['ver_time'], 2),
          np.round(df.mean()['trj_time'], 2)
          )


res     = pd.read_csv('darboux_hdn_100_1st_run.csv', index_col=0)
# res_bis = pd.read_csv('dreal_dom_10_hdn_100_4th_run.csv', index_col=0)
# total = pd.concat([res, res_bis], ignore_index=True)
# total.to_csv('dreal_dom_10_hdn_100_last.csv')

result_analysis(res)
