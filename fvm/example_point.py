#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:16:34 2022

@author: roach
"""

import four_var_model as fvm
import numpy as np
import scipy.io as io


good_point = np.array([0.32,0.25,0.0,0.0])

tmpu = fvm.get_fixed_points(good_point[0],good_point[1],good_point[2],good_point[3],0,0,fvm.abc_good_prams)
tmps = fvm.get_fixed_points(good_point[0],good_point[1],good_point[2],good_point[3],40,0,fvm.abc_good_prams)

run = fvm.selective_inh_foureqn_prod(fvm.abc_good_prams,0,good_point[0],
                                           good_point[1],good_point[2],good_point[3],0.2,fvm.time)
trial_data = fvm.eval_single_trial(run,fvm.time)

io.savemat('ex_fixedpoints_good_one.mat',{'good_point':good_point,
                                          'tmpu': tmpu,
                                          'tmps': tmps,
                                          'run':run,
                                          'trial_data': trial_data,
                                          })
