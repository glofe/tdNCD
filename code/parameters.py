# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:36:06 2020

@author: kdu
"""

param = {
        # path
        'dirCache': r'e:',
        
        # const
        'nameCenter': ['HH-W', 'PL-G', 'PL-S', 'QL-W', 'XW-H', 'XW-Z', 's07'],
        'nCenter': 7,
        'nameGroup': ['NC', 'MCI', 'AD'],
        'nGroup': 3,
        'nROI': 264,
        'nNet': 14,
        'nClassify': 3,
        
        # data
        'use_tdNCD': True,
        
        # FCNet train
        'calcHP': False,
        '2cuda': False,
        'HP_iter': [i * 10 for i in range(1,11)],
        'HP_lr': [1e-4, 1e-5],
        'saveParameters': True,

        # validation
        'validation': True,
        'useMCIAD': False,
        'load_param': True
        
        }
