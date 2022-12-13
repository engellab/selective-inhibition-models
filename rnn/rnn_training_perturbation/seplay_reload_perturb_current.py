#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 07:03:30 2021

@author: roach
"""
import torch
import torch.nn as nn
#from generate_data import *
# import matplotlib.pyplot as plt
import numpy as np
import math
import time
import scipy
import scipy.io as IO
import os
from shutil import copyfile
import sys

from ei_rnn import *

# write_path = sys.argv[1]
comp_path  = sys.argv[1]
perturb_opt = sys.argv[2]

#shuffle_nsel = True
# shuffle_opt = "rank_low"
# shuffle_opt = "rank_high"
# shuffle_opt = "rand"

exec(open(perturb_opt).read())

# shuffle_opt = "rand"

# frac = 0.1

# shuffle_ee = False
# shuffle_ei = False
# shuffle_ie = False
# shuffle_ii = True

# shuffle_ine = False
# shuffle_ini = False
# shuffle_o   = False

timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)



# a = write_path+"2afc_layernet_homoII_newloss"
# wpath2 = a+timestr+'-'+sys.argv[1]+"/"
# # c = b+"data.mat"
# os.mkdir(wpath2)
# # IO.savemat(c,to_save_mat)

# # d = os.getcwd()
# # e = d+"/parameters.py"
# g = wpath2+"parameters.py"
# copyfile(param_path,g)
whoopie = os.listdir(comp_path)

for cc in range(whoopie.__len__()):
# for cc in range(1):
    if "2afc_layernet_" in whoopie[cc]:
        print(str(100*cc/whoopie.__len__())+"% complete")
        param_path = whoopie[cc]
        to_load = comp_path+"/"+whoopie[cc]+"/parameters.py"
        exec(open(to_load).read())
        
        to_load = comp_path+"/"+whoopie[cc]+"/data.mat"
        mat_contents = IO.loadmat(to_load)
        
        to_load = comp_path+"/"+whoopie[cc]+"/evo_sigmasel.mat"
        analyzed = IO.loadmat(to_load)

        to_load = comp_path+"/"+whoopie[cc]+"/m_act.mat"
        m_act = IO.loadmat(to_load)

        x = mat_contents["wee_hist"]
        x = mat_contents["wei_hist"]
        x = mat_contents["wie_hist"]
        x = mat_contents["wii_hist"]
        
        x = mat_contents["wine_hist"]
        x = mat_contents["wini_hist"]
        x = mat_contents["wo_hist"]
        
        
        # frac_e = int(np.ceil(frac*analyzed["auc_e"].shape[0]))
        # frac_i = int(np.ceil(frac*analyzed["auc_i"].shape[0]))
        
        # sel_rankE = np.argsort(np.abs(analyzed["auc_e"][:,0]-0.5))
        # sel_rankI = np.argsort(np.abs(analyzed["auc_i"][:,0]-0.5))
        # sel_rankE = sel_rankE[::-1]
        # sel_rankI = sel_rankI[::-1]
        
        mx_rankE = np.nanmax(m_act["sel_rank_e"])
        mx_rankI = np.nanmax(m_act["sel_rank_i"])
        
        
        
        # create rnn 
        x = mat_contents["wee_hist"]
        xx = x[:,:,-1]
        W_eeB = torch.from_numpy(xx).to(device).to(dtype)
        
        x = mat_contents["wei_hist"]
        xx = x[:,:,-1]
        W_eiB = torch.from_numpy(xx).to(device).to(dtype)          
        
        x = mat_contents["wie_hist"]
        xx = x[:,:,-1]
        W_ieB = torch.from_numpy(xx).to(device).to(dtype)
        
        x = mat_contents["wii_hist"]
        xx = x[:,:,-1]
        W_iiB = torch.from_numpy(xx).to(device).to(dtype)
        
        x = mat_contents["wine_hist"]
        xx = x[:,:,-1]
        W_ineB = torch.from_numpy(xx).to(device).to(dtype)
        
        x = mat_contents["wini_hist"]
        xx = x[:,:,-1]
        W_iniB = torch.from_numpy(xx).to(device).to(dtype)
        
        x = mat_contents["wo_hist"]
        xx = x[:,:,-1]
        W_outB = torch.from_numpy(xx).to(device).to(dtype)
        
        model=  RNN_seplay(nonlinearityE,slopeE,nonlinearityI,slopeI,alpha=alpha,sigma=noise,W_ee=W_eeB,W_ei=W_eiB,W_ie=W_ieB,
                           W_ii=W_iiB,W_ine=W_ineB,W_ini=W_iniB,W_out=W_outB,
                           ee_learning=ee_learning,ei_learning=ei_learning,
                           ie_learning=ie_learning,ii_learning=ii_learning,
                           ine_learning=ine_learning,ini_learning=ini_learning,
                           o_learning=o_learning)
        
        model.recurrent_layerEE.weight.data = W_eeB
        model.recurrent_layerEI.weight.data = W_eiB
        model.recurrent_layerIE.weight.data = W_ieB
        model.recurrent_layerII.weight.data = W_iiB
        model.input_layerE.weight.data = W_ineB
        model.input_layerI.weight.data = W_iniB
        model.output_layer.weight.data = W_outB
        
        model = model.to(device=device)

        trial_maker = gen_trials(trial_length,inputs,outputs,uo,sigin,scale,upfreq,downfreq,l,ma,mb,t_final,stim_on,stim_off)
        
        val_trials_inp, val_trials_out, val_trials_mask, coh_trials_v, bin_trials_v, trial_startsv, trial_endsv = \
            make_validation_batch2_fixeddur(trial_maker,batch_size_v,trial_length,inputs,outputs,device,nbinns,psych_itr,binns,dur)
        
        init_input = (0.2*torch.rand(batch_size_v, 1,model.input_size)).to(device=device)
        init_ex    = (2.0*torch.rand(batch_size_v, 1,model.Ne)-1).to(device=device)
        init_ih    = (2.0*torch.rand(batch_size_v, 1,model.Ni)-1).to(device=device)
        
        perturbE = torch.zeros([batch_size_v,val_trials_out.size()[1],model.Ne]); # Batch_size x time x N
        perturbI = torch.zeros([batch_size_v,val_trials_out.size()[1],model.Ni]); # Batch_size x time x N
        
        for i in range(val_trials_out.size()[0]):
            dur_p = trial_endsv[i] - trial_startsv[i]
            perturbE[i,trial_startsv[i]:trial_endsv[i],:] = E_perb*torch.ones([1,dur_p,model.Ne])
            perturbI[i,trial_startsv[i]:trial_endsv[i],:] = I_perb*torch.ones([1,dur_p,model.Ni])
        
        out_predB,input_actB,act_eB,act_iB,states_eB,states_iB= model(val_trials_inp,init_input,init_ex,init_ih,0.0*perturbE,0.0*perturbI)
        
        out_predP,input_actP,act_eP,act_iP,states_eP,states_iP= model(val_trials_inp,init_input,init_ex,init_ih,perturbE,perturbI)
        
        
        ntrialB, psychoB, chronoB, perf0B, perf1B, perfB,rts_arrayB,resp0B,resp1B, perf_easyB, perf_totalB,choice_vecB = \
            check_performance(binns,psych_itr,batch_size_v,out_predB,decision_sep,low_sep,trial_startsv,trial_endsv,bin_trials_v,trial_opt)
            
        ntrialP, psychoP, chronoP, perf0P, perf1P, perfP,rts_arrayP,resp0P,resp1P, perf_easyP, perf_totalP, choice_vecP = \
            check_performance(binns,psych_itr,batch_size_v,out_predP,decision_sep,low_sep,trial_startsv,trial_endsv,bin_trials_v,trial_opt)
            
        to_save = dict()
        
        to_save["trails_inp"] = val_trials_inp.cpu().detach().numpy()
        to_save["trails_out"] = val_trials_out.cpu().detach().numpy()
        to_save["trails_mask"] = val_trials_mask.cpu().detach().numpy()
        to_save["trails_coh"] = coh_trials_v
        to_save["trails_bin"] = bin_trials_v
        to_save["trails_starts"] = trial_startsv
        to_save["trails_ends"] = trial_endsv
        
        to_save["baseline_out"] = out_predB.cpu().detach().numpy()
        to_save["baseline_acte"] = act_iB.cpu().detach().numpy()
        to_save["baseline_acti"] = act_eB.cpu().detach().numpy()
        to_save["baseline_act_inp"] = input_actB.cpu().detach().numpy()
        
        to_save["baseline_ntrial"] = ntrialB
        to_save["baseline_psycho"] = psychoB, 
        to_save["baseline_chrono"] = chronoB, 
        to_save["baseline_perf0"] = perf0B, 
        to_save["baseline_perf1"] = perf1B, 
        to_save["baseline_perf"] = perfB,
        to_save["baseline_rts_array"] = rts_arrayB,
        to_save["baseline_resp0"] = resp0B,
        to_save["baseline_resp1"] = resp1B, 
        to_save["baseline_perf_easy"] = perf_easyB, 
        to_save["baseline_perf_total"] = perf_totalB,
        to_save["baseline_choice_vec"] = choice_vecB
        
        to_save["perturb_out"] = out_predP.cpu().detach().numpy()
        to_save["perturb_acte"] = act_eP.cpu().detach().numpy()
        to_save["perturb_acti"] = act_iP.cpu().detach().numpy()
        to_save["perturb_act_inp"] = input_actP.cpu().detach().numpy()
        
        to_save["perturb_ntrial"] = ntrialP
        to_save["perturb_psycho"] = psychoP, 
        to_save["perturb_chrono"] = chronoP, 
        to_save["perturb_perf0"] = perf0P, 
        to_save["perturb_perf1"] = perf1P, 
        to_save["perturb_perf"] = perfP,
        to_save["perturb_rts_array"] = rts_arrayP,
        to_save["perturb_resp0"] = resp0P,
        to_save["perturb_resp1"] = resp1P, 
        to_save["perturb_perf_easy"] = perf_easyP, 
        to_save["perturb_perf_total"] = perf_totalP,
        to_save["perturb_choice_vec"] = choice_vecP
        
        to_save["binns"] = binns
        
        tag = "perturb_E_"+str(E_perb)+"_"+"I_"+str(I_perb)
        
        a = comp_path+"/"+whoopie[cc]+"/"+tag+".mat"
        IO.savemat(a,to_save)