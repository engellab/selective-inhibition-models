#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:53:28 2022

@author: roach
"""
import numpy as np
import four_var_model
import scipy.io as io

import sys

from multiprocessing import Pool

# see_arr = np.arange(0.,0.6,0.05)
# sei_arr = np.arange(0.,1.0,0.05)

def f(sei):
    see_arr = np.array([float(sys.argv[1])])
    sei_arr = np.array([sei])
    sie_arr = np.arange(-1.,1.,0.05)
    sii_arr = np.arange(-1.,1.,0.05)
    
    gdd_volume = four_var_model.fourD_fixedpoint_scan(see_arr,sei_arr,sie_arr,sii_arr,four_var_model.abc_good_prams)
    
    io.savemat('gddVol_file_See_'+str(see_arr)+str(sei_arr)+'.mat',{'gdd_volume':gdd_volume,
                                  'See_arr':see_arr,
                                  'Sei_arr':sei_arr,
                                  'Sie_arr':sie_arr,
                                  'Sii_arr':sii_arr})
    # return x*x


if __name__ == '__main__':
    p = Pool(8)
    
    with p:
        p.map(f, np.arange(0.,1.0,0.05))
    
    