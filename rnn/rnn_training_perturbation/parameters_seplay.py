#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:38:18 2020

@author: roachjp
"""

n_runs         = 1

# param_path     = "/Users/roachjp/Documents/GitHub/selectiveinhibition/parameters.py"
# write_path     = "/Users/roach/work/RNNs/"

nonlinearityE = "relu_slope"
slopeE = 1.0

nonlinearityI = "relu_slope"
slopeI = 1.0

he             = 100
hi             = 25
inputs         = 2
outputs        = 2

difficulty1     = 2
difficulty2     = 20
decision_sep   = 0.25
low_sep        = 0.25

dur = 0.35
trial_opt = "strict"


w_mu           = 0.0375
w_sig          = 0.5 
inh_init_scale = 4.0*slopeE/slopeI

ee_learning    = True
ei_learning    = True
ie_learning    = True
ii_learning    = True

ine_learning   = True
ini_learning   = False
o_learning     = True

normalize_weights = False

prohibit_self_synapses = True

ext_inp_to_inh = False

impose_inputs  = False
impose_outputs = False
f              = 0.15

trial_length   = 60 
batch_size     = 200;
binns          = np.arange(-20,22,2)
#binns          = np.array([-51.2,-25.6,-12.8,-6.4,0,6.4,12.8,25.6,51.2])
nbinns         = binns.size
psych_itr      = 100
batch_size_v   = nbinns*psych_itr;
catch_freq     = 0.5;

learning_rate  = 0.01
train2         = "perf"
loss_thresh    = 0.15
max_epoch      = 5000
mact_tar       = 0.0
lam_act        = 0.0
lam_w          = 1.0
lam_x          = 0.1

wL             = 1.0
rL             = 2.0
xL             = 2.0

mini_batch     = batch_size

alpha          = 0.2
noise          = 0.35

weight_min     = 1e-4

uo             = 0.2
sigin          = 0.05
scale          = 3.2
upfreq         = 1.0
downfreq       = 0.2
l              = 0.5
ma             = 0.1
mb             = 1.2
t_final        = 2.0
stim_on        = 0.25
stim_off       = 0.75