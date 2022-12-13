#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:35:55 2020

@author: roachjp
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
import warnings

warnings.filterwarnings('ignore')


from ei_rnn import *
# param_path = "/Users/roach/Documents/GitHub/rnn_selectiveinhibition/parameters_seplay.py"


param_path = sys.argv[2]
write_path = sys.argv[3]

exec(open(param_path).read())

timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)

a = write_path+"2afc_layernet_MSE"
wpath2 = a+timestr+'-'+sys.argv[1]+"/"
# c = b+"data.mat"
os.mkdir(wpath2)
# IO.savemat(c,to_save_mat)

# d = os.getcwd()
# e = d+"/parameters.py"
g = wpath2+"parameters.py"
copyfile(param_path,g)

# make synaptic weights initialize model

W_ee,W_ei,W_ie,W_ii,W_ine,W_ini,W_out = mk_init_w_layers(w_mu,w_sig,inh_init_scale,
                                                         he,hi,inputs,outputs,device,
                                                         dtype,dtype_dex,impose_inputs,impose_outputs,f)

model = RNN_seplay(nonlinearityE,slopeE,nonlinearityI,slopeI,alpha=alpha,sigma=noise,W_ee=W_ee,W_ei=W_ei,W_ie=W_ie,
                   W_ii=W_ii,W_ine=W_ine,W_ini=W_ini,W_out=W_out,
                   ee_learning=ee_learning,ei_learning=ei_learning,
                   ie_learning=ie_learning,ii_learning=ii_learning,
                   ine_learning=ine_learning,ini_learning=ini_learning,
                   o_learning=o_learning)
# model = RNN_seplay(alpha,sigma=noise,W_ee=W_ee,W_ei=W_ei,W_ie=W_ie,W_ii=W_ii,,W_ine=W_ine,W_ini=W_ini,W_out=W_out,ee_learning=ee_learning,ei_learning=ei_learning,ie_learning=ie_learning,ii_learning=ii_learning,ine_learning=ine_learning,ini_learning=ini_learning,o_learning=o_learning)

model = model.to(device=device)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)     
trial_maker = gen_trials(trial_length,inputs,outputs,uo,sigin,scale,upfreq,downfreq,l,ma,mb,t_final,stim_on,stim_off)


# now I need the training batch
batch_trials_inp,batch_trials_out,batch_trials_mask,is_catch,coh_trials,trial_ends = \
        make_training_batch(trial_maker,batch_size,trial_length,inputs,outputs,device,catch_freq,difficulty1,difficulty2)

# val_trials_inp, val_trials_out, val_trials_mask, coh_trials_v, bin_trials_v, trial_ends_v = \
#         make_validation_batch(trial_maker,batch_size_v,trial_length,inputs,outputs,device,nbinns,psych_itr,binns)

val_trials_inp, val_trials_out, val_trials_mask, coh_trials_v, bin_trials_v, trial_startsv, trial_endsv = \
    make_validation_batch2_fixeddur(trial_maker,batch_size_v,trial_length,inputs,outputs,device,nbinns,psych_itr,binns,dur)
        
n_mini = int(batch_size/mini_batch)
mini_trls = np.zeros([n_mini,mini_batch],dtype=int)
for i in range(n_mini):
    mini_trls[i,:] = np.arange(i*mini_batch,(i+1)*mini_batch,dtype=int)

wee_hist   = np.zeros([he,he,10000])
wei_hist  = np.zeros([hi,he,10000])
wie_hist  = np.zeros([he,hi,10000])
wii_hist  = np.zeros([hi,hi,10000])
wine_hist = np.zeros([he,inputs,10000])
wini_hist = np.zeros([hi,inputs,10000])
wo_hist   = np.zeros([outputs,he,10000])

choice_sel_histE = np.zeros([he,10000])
choice_sel_histI = np.zeros([hi,10000])

ntrial_hist = np.zeros([nbinns,10000]) 
psycho_hist = np.zeros([nbinns,10000])
chrono_hist = np.zeros([nbinns,10000])
perf0_hist  = np.zeros([10000]) 
perf1_hist  = np.zeros([10000])
perf_hist   = np.zeros([10000])
perf_easy_hist   = np.zeros([10000])
perf_total_hist   = np.zeros([10000])
rts_hist    = np.zeros([nbinns,psych_itr,10000])
resp0_hist  = np.zeros([nbinns,psych_itr,10000])
resp1_hist  = np.zeros([nbinns,psych_itr ,10000])

sidebias_hist = np.zeros([10000]) 


ep_count = 0
wee_hist[:,:,ep_count] = model.recurrent_layerEE.weight.data.cpu().detach().numpy()
wei_hist[:,:,ep_count] = model.recurrent_layerEI.weight.data.cpu().detach().numpy()
wie_hist[:,:,ep_count] = model.recurrent_layerIE.weight.data.cpu().detach().numpy()
wii_hist[:,:,ep_count] = model.recurrent_layerII.weight.data.cpu().detach().numpy()

wine_hist[:,:,ep_count] = model.input_layerE.weight.data.cpu().detach().numpy()
wini_hist[:,:,ep_count] = model.input_layerI.weight.data.cpu().detach().numpy()

wo_hist[:,:,ep_count] = model.output_layer.weight.data.cpu().detach().numpy()





init_input = (0.2*torch.rand(batch_size, 1,model.input_size)).to(device=device)
init_ex    = (2*torch.rand(batch_size, 1,model.Ne)-1).to(device=device)
init_ih    = (2*torch.rand(batch_size, 1,model.Ni)-1).to(device=device)

ep_count = 1
hist_loss   = []
hist_MSE    = []
nan_flag = True

perf_total = 0.0
ave_loss = 10*loss_thresh
ave_MSE  = 10*loss_thresh
epoch    = 0

if train2 == "perf":
    disc = 1.0-perf_total
elif train2 == "loss":
    disc = ave_loss
elif train2 == "MSE":
    disc = ave_MSE
        
while (disc > loss_thresh)&(epoch<=max_epoch):
        losses = []
        MSEs   = []
        epoch += 1
        
        batch_trials_inp,batch_trials_out,batch_trials_mask,is_catch,coh_trials,trial_ends = \
            make_training_batch(trial_maker,batch_size,trial_length,inputs,outputs,device,catch_freq,difficulty1,difficulty2)
        init_input = (0.2*torch.rand(batch_size, 1,model.input_size)).to(device=device)
        init_ex    = (2*mact_tar*torch.rand(batch_size, 1,model.Ne)-(1*mact_tar)).to(device=device)
        init_ih    = (2*mact_tar*torch.rand(batch_size, 1,model.Ni)-(1*mact_tar)).to(device=device)
        
        rand_trls = torch.arange(batch_size)
        for i in range(n_mini):
            optimiser.zero_grad()
            out_pred,input_act,act_e,act_i,states_e,states_i= model(batch_trials_inp[rand_trls[mini_trls[i,:]],:,:],init_input[rand_trls[mini_trls[i,:]],:,:],init_ex[rand_trls[mini_trls[i,:]],:,:],init_ih[rand_trls[mini_trls[i,:]],:,:])
            # loss = loss_fn(out_pred, batch_trials_out[rand_trls[mini_trls[i,:]],:,:], batch_trials_mask[rand_trls[mini_trls[i,:]],:,:], hl_act_t, lam_act, model, e_col, i_col, lam_w)
            loss,MSE = loss_fn_seplay(out_pred,batch_trials_out[rand_trls[mini_trls[i,:]],:,:],batch_trials_mask[rand_trls[mini_trls[i,:]],:,:],act_e,act_i,states_e,states_i,lam_act,lam_x,model,lam_w,mact_tar,wL,rL,xL)
            losses.append(loss.cpu().detach().numpy())
            MSEs.append(MSE.cpu().detach().numpy())
            if math.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
    #        e_col,i_col
            
            # clip recurrent weights
            tochange = model.recurrent_layerEE.weight.data < weight_min
            model.recurrent_layerEE.weight.data[tochange] = 0.0
            if prohibit_self_synapses:
                model.recurrent_layerEE.weight.data = model.recurrent_layerEE.weight.data*(1.0 - torch.eye(he,device=device))
            
            tochange = model.recurrent_layerEI.weight.data < weight_min
            model.recurrent_layerEI.weight.data[tochange] = 0.0
            
            tochange = model.recurrent_layerIE.weight.data < weight_min
            model.recurrent_layerIE.weight.data[tochange] = 0.0
            
            tochange = model.recurrent_layerII.weight.data < weight_min
            model.recurrent_layerII.weight.data[tochange] = 0.0
            if prohibit_self_synapses:
                model.recurrent_layerII.weight.data = model.recurrent_layerII.weight.data*(1.0 - torch.eye(hi,device=device))
            
            # normalize recurrent inputs 
            if normalize_weights:
                a = model.recurrent_layerEE.weight.data 
                b = torch.sum(a,dim=1)
                c = b.repeat(he,1)
                d = torch.transpose(c,0,1)
                e = a/d
                model.recurrent_layerEE.weight.data= e
                
                a = model.recurrent_layerEI.weight.data 
                b = torch.sum(a,dim=1)
                c = b.repeat(he,1)
                d = torch.transpose(c,0,1)
                e = a/d
                model.recurrent_layerEI.weight.data= e
                
                a = model.recurrent_layerIE.weight.data 
                b = torch.sum(a,dim=1)
                c = b.repeat(hi,1)
                d = torch.transpose(c,0,1)
                e = a/d
                model.recurrent_layerIE.weight.data= e
                
                a = model.recurrent_layerII.weight.data 
                b = torch.sum(a,dim=1)
                c = b.repeat(hi,1)
                d = torch.transpose(c,0,1)
                e = a/d
                model.recurrent_layerII.weight.data= e
            
            # clip input weights
            tochange = model.input_layerE.weight.data < weight_min
            model.input_layerE.weight.data[tochange] = 0.0
            
            if ext_inp_to_inh:
                tochange = model.input_layerI.weight.data < weight_min
                model.input_layerI.weight.data[tochange] = 0.0
            else:
                tochange = model.input_layerI.weight.data != 0.0
                model.input_layerI.weight.data[tochange] = 0.0
            
            # clip output weights
            tochange = model.output_layer.weight.data < weight_min
            model.output_layer.weight.data[tochange] = 0.0
        ave_loss = np.mean(losses)
        ave_MSE  = np.mean(MSEs)
        if math.isnan(ave_loss):
            nan_flag = False
            print(epoch)
            break
        if epoch % 50 == 0:
            print("Epoch ", epoch, "MSE: ", ave_loss, "DelW: ", model.recurrent_layerEE.weight.data[1,2].cpu().detach().numpy())
        if epoch % 1 == 0:
            
            init_input = (0.2*torch.rand(batch_size_v, 1,model.input_size)).to(device=device)
            init_ex    = (2.0*mact_tar*torch.rand(batch_size_v, 1,model.Ne)-(1*mact_tar)).to(device=device)
            init_ih    = (2.0*mact_tar*torch.rand(batch_size_v, 1,model.Ni)-(1*mact_tar)).to(device=device)
            out_pred2,input_act2,act_e2,act_i2,states_e,states_i= model(val_trials_inp,init_input,init_ex,init_ih)
            
            ntrial, psycho, chrono, perf0, perf1, perf,rts_array,resp0,resp1, perf_easy, perf_total,choice_vec = \
                check_performance(binns,psych_itr,batch_size_v,out_pred2,decision_sep,low_sep,trial_startsv,trial_endsv,bin_trials_v,trial_opt)
            
            side_biasM = (perf0-perf1)/(perf0+perf1)
            print("Perf.: ",np.round(perf,decimals=2)," Total Perf.",np.round(perf_total,decimals=2)," Bias: ",np.round(side_biasM,decimals=2)\
                  ,"Trial Comp: ",np.round(np.sum(ntrial)/(nbinns*psych_itr),decimals=2))
            
            aucs,auc_boot,choice_sel,zscore = measure_selectivity(50,choice_vec,act_e2,act_i2)    
                
            choice_sel_histE[:,ep_count] = choice_sel[0]    
            choice_sel_histI[:,ep_count] = choice_sel[1]
            
            sidebias_hist[ep_count]   = side_biasM # when >0 favors side 0 
            ntrial_hist[:,ep_count]   = ntrial
            psycho_hist[:,ep_count]   = psycho
            chrono_hist[:,ep_count]   = chrono
            perf0_hist[ep_count]      = perf0
            perf1_hist[ep_count]      = perf1
            perf_hist[ep_count]       = perf
            perf_easy_hist[ep_count]  = perf_easy
            perf_total_hist[ep_count] = perf_total
            rts_hist[:,:,ep_count]    = rts_array
            resp0_hist[:,:,ep_count]  = resp0
            resp1_hist[:,:,ep_count]  = resp1
    #        w_hist.append(model.recurrent_layer.weight.data.cpu().detach().numpy())
    #        print(model.recurrent_layer.weight.data[1,2].cpu().detach().numpy())
            wee_hist[:,:,ep_count] = model.recurrent_layerEE.weight.data.cpu().detach().numpy()
            wei_hist[:,:,ep_count] = model.recurrent_layerEI.weight.data.cpu().detach().numpy()
            wie_hist[:,:,ep_count] = model.recurrent_layerIE.weight.data.cpu().detach().numpy()
            wii_hist[:,:,ep_count] = model.recurrent_layerII.weight.data.cpu().detach().numpy()
            
            wine_hist[:,:,ep_count] = model.input_layerE.weight.data.cpu().detach().numpy()
            wini_hist[:,:,ep_count] = model.input_layerI.weight.data.cpu().detach().numpy()
            
            wo_hist[:,:,ep_count] = model.output_layer.weight.data.cpu().detach().numpy()
            ep_count += 1
    #        w_hist.append(model.recurrent_layer.weight.data[1,2].cpu().detach().numpy())
    #        w_hist[:,:,int(np.ceil(t/100))] = model.rnn.weight_hh_l0.data.cpu().detach().numpy()
    #        act_hist[:,:,:,int(np.ceil(t/100))] = hl_act_v.data.cpu().detach().numpy()
    #    if loss < 2.0:
    #        break
        hist_loss.append(ave_loss)
        hist_MSE.append(ave_MSE)
        if train2 == "perf":
            disc = 1.0-perf_total
        elif train2 == "loss":
            disc = ave_loss
        elif train2 == "MSE":
            disc = ave_MSE
        
init_input = (0.2*torch.rand(batch_size_v, 1,model.input_size)).to(device=device)
init_ex    = (2.0*torch.rand(batch_size_v, 1,model.Ne)-1).to(device=device)
init_ih    = (2.0*torch.rand(batch_size_v, 1,model.Ni)-1).to(device=device)
out_pred2,input_act2,act_e2,act_i2,states_e,states_i= model(val_trials_inp,init_input,init_ex,init_ih)

ntrial, psycho, chrono, perf0, perf1, perf,rts_array,resp0,resp1, perf_easy, perf_total,choice_vec = \
    check_performance(binns,psych_itr,batch_size_v,out_pred2,decision_sep,low_sep,trial_startsv,trial_endsv,bin_trials_v,trial_opt)


# resp   = np.zeros(len(binns))
# ntrial = np.zeros(len(binns))

# for i in range(0,batch_size_v):
# #    a  = np.abs(coh_trials_v[i]-binns);
# #    ai = np.argmin(a)
#     if torch.abs(out_pred2[i,trial_ends_v[i]+1,0] - out_pred2[i,trial_ends_v[i]+1,1]) > decison_sep: # a choice has been made
#         ntrial[bin_trials_v[i]] += 1
#         if out_pred2[i,trial_ends_v[i],0] > out_pred2[i,trial_ends_v[i],1]:
#             resp[bin_trials_v[i]] += 1
# psycho = resp/ntrial
is_gpu_run = next(model.parameters()).is_cuda #tests if model is being run on GPU (yes if True)
# print(ntrial)

if nan_flag:
    to_save_mat = {}
        
    to_save_mat["ne"]             = he
    to_save_mat["ni"]             = hi
    to_save_mat["inputs"]         = inputs
    to_save_mat["outputs"]        = outputs
    
    to_save_mat["difficulty1"]     = difficulty1
    to_save_mat["difficulty2"]     = difficulty2
    
    to_save_mat["alpha"]          = alpha
    to_save_mat["noise"]          = noise
    
    to_save_mat["w_mu"]           = w_mu
    to_save_mat["w_sig"]          = w_sig
    to_save_mat["inh_scale"]      = inh_init_scale
    to_save_mat["weight_min"]     = weight_min
    
    to_save_mat["lr"]             = learning_rate
    to_save_mat["lam_act"]        = lam_act
    to_save_mat["lam_w"]          = lam_w
    to_save_mat["loss_thresh"]    = loss_thresh
    to_save_mat["is_gpu_run"]     = is_gpu_run
    
    to_save_mat["input_baseline"] = uo 
    to_save_mat["input_noise"]    = sigin 
    to_save_mat["input_scale"]    = scale  
    to_save_mat["up_freq"]        = upfreq  
    to_save_mat["down_freq"]      = downfreq  
    to_save_mat["mean_stiml"]     = l        
    to_save_mat["min_stiml"]      = ma         
    to_save_mat["max_stiml"]      = mb      
    to_save_mat["t_final"]        = t_final 
    to_save_mat["stim_on_frac"]   = stim_on  
    to_save_mat["stim_off_frac"]  = stim_off  
    
    to_save_mat["trial_length"]   = trial_length
    to_save_mat["batch_size"]     = batch_size
    to_save_mat["batch_size_v"]   = batch_size_v
    to_save_mat["catch_freq"]     = catch_freq
    to_save_mat["mini_batch"]     = mini_batch
    
    to_save_mat["n_train"]        = epoch

    to_save_mat["inp_baseline"]   = trial_maker.uo
    to_save_mat["inp_noise"]      = trial_maker.sigin
    to_save_mat["inp_scale"]      = trial_maker.scale
    
    to_save_mat["loss_fcn"]        = hist_loss
    to_save_mat["loss_mse"]        = hist_MSE
    
    to_save_mat["wee_hist"]        = wee_hist[:,:,0:ep_count]
    to_save_mat["wei_hist"]        = wei_hist[:,:,0:ep_count]
    to_save_mat["wie_hist"]        = wie_hist[:,:,0:ep_count]
    to_save_mat["wii_hist"]        = wii_hist[:,:,0:ep_count]
    
    to_save_mat["wine_hist"]       = wine_hist[:,:,0:ep_count]
    to_save_mat["wini_hist"]       = wini_hist[:,:,0:ep_count]
    to_save_mat["wo_hist"]         = wo_hist[:,:,0:ep_count]
    
    
#    to_save_mat["act_hist"]       = act_hist

    to_save_mat["ree_weights"]    = model.recurrent_layerEE.weight.data.cpu().detach().numpy()
    to_save_mat["rei_weights"]    = model.recurrent_layerEI.weight.data.cpu().detach().numpy()
    to_save_mat["rie_weights"]    = model.recurrent_layerIE.weight.data.cpu().detach().numpy()
    to_save_mat["rii_weights"]    = model.recurrent_layerII.weight.data.cpu().detach().numpy()
    to_save_mat["ine_weights"]    = model.input_layerE.weight.data.cpu().detach().numpy()
    to_save_mat["ini_weights"]    = model.input_layerI.weight.data.cpu().detach().numpy()
    
    to_save_mat["output_weights"]   = model.output_layer.weight.data.cpu().detach().numpy()
    to_save_mat["val_pred"]         = out_pred2.data.cpu().detach().numpy()
    to_save_mat["trn_pred"]         = out_pred.data.cpu().detach().numpy()
    to_save_mat["trial_starts_val"] = trial_startsv
    to_save_mat["trial_end_val"]    = trial_endsv
    to_save_mat["trial_end_trn"]    = trial_ends
    
    to_save_mat["trn_act_inp"]    = input_act.data.cpu().detach().numpy()
    to_save_mat["val_act_inp"]    = input_act2.data.cpu().detach().numpy()
    
    to_save_mat["trn_act_e"]      = act_e.data.cpu().detach().numpy()
    to_save_mat["val_act_e"]      = act_e2.data.cpu().detach().numpy()
    
    to_save_mat["trn_act_i"]      = act_i.data.cpu().detach().numpy()
    to_save_mat["val_act_i"]      = act_i2.data.cpu().detach().numpy()
    
    to_save_mat["val_states_e"] = states_e.data.cpu().detach().numpy()
    to_save_mat["val_states_i"] = states_i.data.cpu().detach().numpy()
    
    to_save_mat["trn_inps"]       = batch_trials_inp.data.cpu().detach().numpy()
    to_save_mat["trn_outs"]       = batch_trials_out.data.cpu().detach().numpy()
    to_save_mat["trn_mask"]       = batch_trials_mask.data.cpu().detach().numpy()
    to_save_mat["val_inps"]       = val_trials_inp.data.cpu().detach().numpy()
    to_save_mat["val_outs"]       = val_trials_out.data.cpu().detach().numpy()
    to_save_mat["val_mask"]       = val_trials_mask.data.cpu().detach().numpy()
        
    to_save_mat["val_coh"]        = coh_trials_v
    to_save_mat["trn_coh"]        = coh_trials
        
    to_save_mat["psycho_b"]       = binns
    to_save_mat["psycho"]         = psycho
    to_save_mat["ntrial"]         = ntrial
    to_save_mat["chrono"]         = chrono
    to_save_mat["perf0"]          = perf0 
    to_save_mat["perf1"]          = perf1 
    to_save_mat["perf"]           = perf
    to_save_mat["rts"]            = rts_array
    to_save_mat["resp0"]          = resp0
    to_save_mat["resp1"]          = resp1 
    to_save_mat["perf_easy"]      = perf_easy 
    to_save_mat["perf_total"]     = perf_total
    to_save_mat["choice"]         = choice_vec
    
    
    to_save_mat["choice_selE"]         = choice_sel_histE 
    to_save_mat["choice_selI"]         = choice_sel_histI 
    
    to_save_mat["ntrial_hist"]   = ntrial_hist[:,0:ep_count] 
    to_save_mat["psycho_hist"]   = psycho_hist[:,0:ep_count] 
    to_save_mat["chrono_hist"]   = chrono_hist[:,0:ep_count] 
    to_save_mat["perf0_hist"]    =  perf0_hist[0:ep_count] 
    to_save_mat["perf1_hist"]    =  perf1_hist[0:ep_count] 
    to_save_mat["perf_hist"]     = perf_hist[0:ep_count]
    to_save_mat["perfE_hist"]    = perf_easy_hist[0:ep_count]
    to_save_mat["perfT_hist"]    = perf_total_hist[0:ep_count]
    to_save_mat["rts_hist"]      =  rts_hist[:,:,0:ep_count]
    to_save_mat["resp0_hist"]    =  resp0_hist[:,:,0:ep_count]
    to_save_mat["resp1_hist"]    =  resp1_hist[:,:,0:ep_count]
    to_save_mat["sidebias_hist"] =  sidebias_hist[0:ep_count]
    
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)
# #    a = "/home/roach/rnns/network_"
# #    a = "C:\\Users\jroa0\OneDrive\Documents\rnn_data\\2afc_net_"
# #    a = "C:/Users/jroa0/OneDrive/Documents/rnn_data/2afc_net_"
#     if os.path.exists("/Users/roach/work/RNNS/"):
#         a = "/Users/roach/work/RNNS/2afc_net2_"
#     elif os.path.exists("/Users/roachjp/work/RNNS/"):
#         a = "/Users/roachjp/work/RNNS/2afc_net2_"
#     elif os.path.exists("C:/Users/jroa0/OneDrive/Documents/rnn_data/"):
#         a = "C:/Users/jroa0/OneDrive/Documents/rnn_data/2afc_net2_"
        
#     elif os.path.exists("/home/roach/rnns/data/"):
#         a = "/home/roach/rnns/data/2afc_net2_"
        
#    a = "/home/roach/rnns/data/2afc_net_"

    # a = write_path+"2afc_layernet_"
    # b = a+timestr+"/"
    # c = b+"data.mat"
    # os.mkdir(b)
    
    c = wpath2+"data.mat"
    # os.mkdir(b)
    IO.savemat(c,to_save_mat)
    
    # IO.savemat(c,to_save_mat)
    
    # d = os.getcwd()
    # e = d+"/parameters.py"
    # g = b+"parameters.py"
    # copyfile(param_path,g)