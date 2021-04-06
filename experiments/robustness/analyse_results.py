# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from argparse import ArgumentParser
import os.path

import pandas as pd
import numpy as np

def is_valid_file(parser, arg):
      if not os.path.isfile(arg):
            parser.error("The file {} does not exist".format(arg))
      else:
            return arg

def result_analysis(df):

    print('Avg Time: ', df.mean()['elapsed_time'])
    print('Learner avg: ', df.mean()['lrn_time'])
    print('Translator avg: ', df.mean()['reg_time'])
    print('Verifier avg: ', df.mean()['ver_time'])
    print('Consolidator avg: ', df.mean()['trj_time'])
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


if __name__ == '__main__':
      parser = ArgumentParser(description="Results analyser")
      parser.add_argument("-f", dest="filename", required=True,
                          help="csv input file", metavar="FILE", 
                          type=lambda x: is_valid_file(parser, x))
      args = parser.parse_args()
      res = pd.read_csv(args.filename, index_col=0)
      result_analysis(res)
