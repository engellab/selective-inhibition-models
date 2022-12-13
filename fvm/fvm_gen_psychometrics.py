#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 10:19:31 2022

@author: roach
"""

import four_var_model as fvm
# import numpy as np
import scipy.io as io
import datetime

import sys

dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

see = 0.32
sei = 0.25
# sie = [-0.01,0,0.01]
# sei = float(sys.argv[1])
sie = float(sys.argv[1])
# sie = 0.01
sii = 0.0

s_lo = -20.0
s_hi =  20.0

runs = 1000

psychometrics = fvm.gen_psychometrics(runs,see,sei,sie,sii,s_lo,s_hi,fvm.abc_good_prams,fvm.time)


mdict = {'see':    see,
         'sei':    sei,
         'sie':    sie,
         'sii':    sii,
         's_lo':   s_lo,
         's_hi':   s_hi,
         'runs':   runs,
         'psycho': psychometrics['psycho'],
         'chrono': psychometrics['chrono'],
         'ntrial': psychometrics['ntrial'],
         'coh':    psychometrics['coh']  
        }

# io.savemat('psychos_sie'+str(sie)+'_'+dt+'.mat',mdict)
io.savemat('psychos_sei'+str(sei)+'_'+dt+'.mat',mdict)