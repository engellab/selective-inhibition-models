#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:54:54 2020

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
from timeit import default_timer as timer
from sklearn.metrics import roc_curve, auc


#establish CUDA device for GPU
if torch.cuda.is_available():
    device = 'cuda'
    cuda = torch.device('cuda') #or "cuda:0"
    # dtype = torch.cuda.FloatTensor  #changed from float
    dtype = torch.float
else:
    device = 'cpu'
    dtype = torch.float
dtype_dex = torch.long   

class RNN(nn.Module):
    
    def __init__(self,alpha,sigma,W_rec,W_in,W_out,r_learning,i_learning,o_learning):
           
        super(RNN,self).__init__()     
        self.W_rec = W_rec
        self.W_in = W_in
        self.W_out = W_out
        self.N = W_rec.shape[0]
        self.input_size = 2
        self.output_size = 2
        self.alpha = alpha
        self.sigma =sigma
        self.rate_constant=.01
        
        self.recurrent_layer = nn.Linear(self.N,self.N,bias=False)      
        self.recurrent_layer.weight.data = self.W_rec.to(device=device)
        self.recurrent_layer.weight.requires_grad=r_learning
        
        self.input_layer = nn.Linear(self.input_size,self.N,bias=False)
        self.input_layer.weight.data = self.W_in.to(device=device)
        self.input_layer.weight.requires_grad=i_learning
        
        self.output_layer = nn.Linear(self.N,self.output_size,bias=False)
        self.output_layer.weight.data = self.W_out.to(device=device)
        self.output_layer.weight.requires_grad=o_learning
  
    def forward(self,u,init_state):   
    
        T = u.shape[1]
        states = init_state
        batch_size = states.shape[0]
        noise = np.sqrt(2*self.alpha*self.sigma**2)*torch.empty(batch_size,T,self.N).normal_(mean=0,std=1).to(device=device)
       
        for i in range(T-1): 
            state_new = (1-self.alpha)*states[:,i,:] + self.alpha*(self.recurrent_layer(torch.relu(states[:,i,:])) + self.input_layer(u[:,i,:])) +  noise[:,i,:]
            states = torch.cat((states,state_new.unsqueeze_(1)),1) 
            
        out = self.output_layer(torch.relu(states)) 
        act = torch.relu(states)
        r=self.rate_constant/T/self.N/batch_size
        self.R = r*torch.sum(torch.pow(states,2))

        self.L1 = self.R
        
        return out,act,states

class RNN_seplay(nn.Module):
    
    def __init__(self,nonlinearityE,slopeE,nonlinearityI,slopeI,alpha,sigma,W_ee,W_ei,W_ie,W_ii,W_ine,W_ini,W_out,ee_learning, 
                 ei_learning,ie_learning,ii_learning,ine_learning,ini_learning,o_learning):
           
        super(RNN_seplay,self).__init__()     
        
        self.nonlinearityE = nonlinearityE
        self.slopeE = slopeE
        
        self.nonlinearityI = nonlinearityI
        self.slopeI = slopeI
        
        self.W_ee = W_ee
        self.W_ei = W_ei
        self.W_ie = W_ie
        self.W_ii = W_ii
        
        self.W_ine = W_ine
        self.W_ini = W_ini
        
        self.W_out = W_out
        
        
        self.Ne = W_ee.shape[0]
        self.Ni = W_ii.shape[0]

        
        self.input_size = W_ine.shape[1]
        self.output_size = W_out.shape[0]
        self.alpha = alpha
        self.sigma =sigma
        # self.rate_constant=.01
        
        self.recurrent_layerEE = nn.Linear(self.Ne,self.Ne,bias=False)      
        self.recurrent_layerEE.weight.data = self.W_ee.to(device=device)
        self.recurrent_layerEE.weight.requires_grad=ee_learning
        
        self.recurrent_layerEI = nn.Linear(self.Ni,self.Ne,bias=False)      
        self.recurrent_layerEI.weight.data = self.W_ei.to(device=device)
        self.recurrent_layerEI.weight.requires_grad=ei_learning
        
        self.recurrent_layerIE = nn.Linear(self.Ne,self.Ni,bias=False)      
        self.recurrent_layerIE.weight.data = self.W_ie.to(device=device)
        self.recurrent_layerIE.weight.requires_grad=ie_learning
        
        self.recurrent_layerII = nn.Linear(self.Ni,self.Ni,bias=False)      
        self.recurrent_layerII.weight.data = self.W_ii.to(device=device)
        self.recurrent_layerII.weight.requires_grad=ii_learning
        
        self.input_layerE = nn.Linear(self.input_size,self.Ne,bias=False)
        self.input_layerE.weight.data = self.W_ine.to(device=device)
        self.input_layerE.weight.requires_grad=ine_learning
        
        self.input_layerI = nn.Linear(self.input_size,self.Ni,bias=False)
        self.input_layerI.weight.data = self.W_ini.to(device=device)
        self.input_layerI.weight.requires_grad=ini_learning
        
        self.output_layer = nn.Linear(self.Ne,self.output_size,bias=False)
        self.output_layer.weight.data = self.W_out.to(device=device)
        self.output_layer.weight.requires_grad=o_learning
    
    def actE(self,x):
        if self.nonlinearityE == "relu":
            act = torch.relu(x)
        elif self.nonlinearityE == "tanh":
            act = torch.tanh(x)
        elif self.nonlinearityE == "sigmoid":
            act = torch.sigmoidE(x)
        elif self.nonlinearityE == "silu":
            m = torch.nn.SiLU()
            act = torch.relu(m(x))
        elif self.nonlinearityE == "relu_slope":
            act = self.slopeE*torch.relu(x)
        elif self.nonlinearityE == "relu_exp":
            act = torch.relu(x)**self.slopeE
        return act 
    def actI(self,x):
        if self.nonlinearityI == "relu":
            act = torch.relu(x)
        elif self.nonlinearityI == "tanh":
            act = torch.tanh(x)
        elif self.nonlinearityI == "sigmoid":
            act = torch.sigmoidE(x)
        elif self.nonlinearityI == "silu":
            m = torch.nn.SiLU()
            act = torch.relu(m(x))
        elif self.nonlinearityI == "relu_slope":
            act = self.slopeI*torch.relu(x)
        elif self.nonlinearityI == "relu_exp":
            act = torch.relu(x)**self.slopeI
        return act 
    
    def forward(self,u,init_state_in,init_state_e,init_state_i,perturbE,perturbI):   
    
        T = u.shape[1]
        states_in = init_state_in
        states_e  = init_state_e
        states_i  = init_state_i
        
        batch_size = states_e.shape[0]
        noise_e = np.sqrt(2*self.alpha*self.sigma**2)*torch.empty(batch_size,T,self.Ne).normal_(mean=0,std=1).to(device=device)
        noise_i = np.sqrt(2*self.alpha*self.sigma**2)*torch.empty(batch_size,T,self.Ni).normal_(mean=0,std=1).to(device=device)
        
        for i in range(T-1): 
            
            state_in_new = (1-self.alpha)*states_in[:,i,:] +\
                self.alpha*(u[:,i,:])
            
            state_e_new  = (1-self.alpha)*states_e[:,i,:] + \
                self.alpha*(self.recurrent_layerEE(self.actE(states_e[:,i,:])) - self.recurrent_layerIE(self.actI(states_i[:,i,:])) + self.input_layerE(states_in[:,i,:]) + perturbE[:,i,:]) +  noise_e[:,i,:]
                
            state_i_new  = (1-self.alpha)*states_i[:,i,:] + \
                self.alpha*(self.recurrent_layerEI(self.actE(states_e[:,i,:])) - self.recurrent_layerII(self.actI(states_i[:,i,:])) + self.input_layerI(states_in[:,i,:]) + perturbI[:,i,:]) +  noise_i[:,i,:]
            
            states_in = torch.cat((states_in,state_in_new.unsqueeze_(1)),1) 
            states_e  = torch.cat((states_e,state_e_new.unsqueeze_(1)),1) 
            states_i  = torch.cat((states_i,state_i_new.unsqueeze_(1)),1) 
            
        out = self.output_layer(self.actE(states_e)) 
        act_e = self.actE(states_e)
        act_i = self.actI(states_i)
        # r=self.rate_constant/T/self.N/batch_size
        # self.R = r*torch.sum(torch.pow(states,2))

        # self.L1 = self.R
        
        return out,states_in,act_e,act_i,states_e,states_i

class RNN_seplay_blankII(nn.Module):
    
    def __init__(self,alpha,sigma,he,hi,W_ee,W_ei,W_ie,wii0,W_ine,W_ini,W_out,ee_learning, 
                  ei_learning,ie_learning,ii_learning,ine_learning,ini_learning,o_learning,use_bias):
           
        super(RNN_seplay_blankII,self).__init__()     
        self.W_ee = W_ee
        self.W_ei = W_ei
        self.W_ie = W_ie
        # self.W_ii = W_ii
        
        self.W_ine = W_ine
        self.W_ini = W_ini
        
        self.W_out = W_out
        
        if device == 'cuda':
            self.wii0  = nn.Parameter(torch.tensor(wii0).cuda())
        else:
            self.wii0  = nn.Parameter(torch.tensor(wii0))
        # self.wii0  = self.wii0.to(device=device)
        print('At model.__init__ wii0 device:', self.wii0.device)
        self.register_buffer('w0ii', torch.ones([hi,hi])-torch.eye(hi))
        
        
        self.Ne = he
        self.Ni = hi

        
        self.input_size = W_ine.shape[1]
        self.output_size = W_out.shape[0]
        self.alpha = alpha
        self.sigma =sigma
        # self.rate_constant=.01
        
        self.recurrent_layerEE = nn.Linear(self.Ne,self.Ne,bias=use_bias)      
        self.recurrent_layerEE.weight.data = self.W_ee.to(device=device)
        self.recurrent_layerEE.weight.requires_grad=ee_learning
        
        self.recurrent_layerEI = nn.Linear(self.Ni,self.Ne,bias=use_bias)      
        self.recurrent_layerEI.weight.data = self.W_ei.to(device=device)
        self.recurrent_layerEI.weight.requires_grad=ei_learning
        
        self.recurrent_layerIE = nn.Linear(self.Ne,self.Ni,bias=use_bias)      
        self.recurrent_layerIE.weight.data = self.W_ie.to(device=device)
        self.recurrent_layerIE.weight.requires_grad=ie_learning
        
        self.recurrent_layerII = nn.Linear(self.Ni,self.Ni,bias=False)      
        self.recurrent_layerII.weight.data = self.w0ii.to(device=device)
        self.recurrent_layerII.weight.requires_grad=False
        self.wii0.requires_grad=ii_learning
        
        self.input_layerE = nn.Linear(self.input_size,self.Ne,bias=use_bias)
        self.input_layerE.weight.data = self.W_ine.to(device=device)
        self.input_layerE.weight.requires_grad=ine_learning
        
        self.input_layerI = nn.Linear(self.input_size,self.Ni,bias=use_bias)
        self.input_layerI.weight.data = self.W_ini.to(device=device)
        self.input_layerI.weight.requires_grad=ini_learning
        
        self.output_layer = nn.Linear(self.Ne,self.output_size,bias=use_bias)
        self.output_layer.weight.data = self.W_out.to(device=device)
        self.output_layer.weight.requires_grad=o_learning
  
    def forward(self,u,init_state_in,init_state_e,init_state_i,bias_corr):   
    
        T = u.shape[1]
        states_in = init_state_in.to(device=device)
        states_e  = init_state_e.to(device=device)
        states_i  = init_state_i.to(device=device)
    
        
        batch_size = states_e.shape[0]
        noise_e = np.sqrt(2*self.alpha*self.sigma**2)*torch.empty(batch_size,T,self.Ne).normal_(mean=0,std=1).to(device=device)
        noise_i = np.sqrt(2*self.alpha*self.sigma**2)*torch.empty(batch_size,T,self.Ni).normal_(mean=0,std=1).to(device=device)
        
        for i in range(T-1): 
            
            state_in_new = (1-self.alpha)*states_in[:,i,:] +\
                self.alpha*(u[:,i,:]+bias_corr[:,0,:])
            # print(state_in_new.dtype)
            state_e_new  = (1-self.alpha)*states_e[:,i,:] + \
                self.alpha*(self.recurrent_layerEE(torch.relu(states_e[:,i,:])) - self.recurrent_layerIE(torch.relu(states_i[:,i,:])) + self.input_layerE(states_in[:,i,:])) +  noise_e[:,i,:]
                
            state_i_new  = (1-self.alpha)*states_i[:,i,:] + \
                self.alpha*(self.recurrent_layerEI(torch.relu(states_e[:,i,:])) - self.wii0*self.recurrent_layerII(torch.relu(states_i[:,i,:])) + self.input_layerI(states_in[:,i,:])) +  noise_i[:,i,:]
            
            states_in = torch.cat((states_in,state_in_new.unsqueeze_(1)),1) 
            states_e  = torch.cat((states_e,state_e_new.unsqueeze_(1)),1) 
            states_i  = torch.cat((states_i,state_i_new.unsqueeze_(1)),1) 
            
        out = self.output_layer(torch.relu(states_e)) 
        act_e = torch.relu(states_e)
        act_i = torch.relu(states_i)
        # r=self.rate_constant/T/self.N/batch_size
        # self.R = r*torch.sum(torch.pow(states,2))

        # self.L1 = self.R
        
        return out,states_in,act_e,act_i,states_e,states_i

class RNN_seplay_selinh(nn.Module):
    
     def __init__(self,alpha,sigma,he,hi,Ns,f,ext_to_inh,wee0,wei0,wie0,wii0,win0,wou0,see0,
                  sei0,sie0,sii0,wee0_learning,wei0_learning,wie0_learning,wii0_learning,
                  win0_learning,wou0_learning,sigma_ee_learning,sigma_ei_learning,
                  sigma_ie_learning,sigma_ii_learning):
         
         super(RNN_seplay_selinh,self).__init__() 
         self.alpha = alpha
         self.sigma = sigma
         self.he = he
         self.hi = hi
         self.Ns = Ns
         self.f  = f
         
         gsze = int(self.f*self.he + 0.5)
         gszi = int(self.f*self.hi + 0.5)
         
         self.register_buffer('wpee', torch.zeros([self.he,self.he]))
         # self.wpee = torch.zeros([self.he,self.he])
         self.wpee[0:gsze,0:gsze]           = 1
         self.wpee[gsze:2*gsze,gsze:2*gsze] = 1
         
         self.register_buffer('wpei', torch.zeros([self.hi,self.he]))
         # self.wpei = torch.zeros([self.hi,self.he])
         self.wpei[0:gszi,0:gsze]           = 1
         self.wpei[gszi:2*gszi,gsze:2*gsze] = 1
         
         self.register_buffer('wpie', torch.zeros([self.he,self.hi]))
         # self.wpie = torch.zeros([self.he,self.hi])
         self.wpie[0:gsze,0:gszi]           = 1
         self.wpie[gsze:2*gsze,gszi:2*gszi] = 1
         
         self.register_buffer('wpii', torch.zeros([self.hi,self.hi]))
         # self.wpii = torch.zeros([self.hi,self.hi])
         self.wpii[0:gszi,0:gszi]           = 1
         self.wpii[gszi:2*gszi,gszi:2*gszi] = 1
         
         self.register_buffer('wmee', torch.zeros([self.he,self.he]))
         # self.wmee = torch.zeros([self.he,self.he])
         self.wmee[0:gsze,gsze:2*gsze]      = 1
         self.wmee[gsze:2*gsze,0:gsze]      = 1
         
         self.register_buffer('wmei', torch.zeros([self.hi,self.he]))
         # self.wmei = torch.zeros([self.hi,self.he])
         self.wmei[0:gszi,gsze:2*gsze]      = 1
         self.wmei[gszi:2*gszi,0:gsze]      = 1
         
         self.register_buffer('wmie', torch.zeros([self.he,self.hi]))
         # self.wmie = torch.zeros([self.he,self.hi])
         self.wmie[0:gsze,gszi:2*gszi]      = 1
         self.wmie[gsze:2*gsze,0:gszi]      = 1
         
         self.register_buffer('wmii', torch.zeros([self.hi,self.hi]))
         # self.wmii = torch.zeros([self.hi,self.hi])
         self.wmii[0:gszi,gszi:2*gszi]      = 1
         self.wmii[gszi:2*gszi,0:gszi]      = 1
         
         self.register_buffer('w0ee', torch.ones([self.he,self.he]))
         # self.w0ee = torch.ones([self.he,self.he])
         self.w0ee[self.wpee == 1] = 0
         self.w0ee[self.wmee == 1] = 0
         
         self.register_buffer('w0ei', torch.ones([self.hi,self.he]))
         # self.w0ei = torch.ones([self.hi,self.he])
         self.w0ei[self.wpei == 1] = 0
         self.w0ei[self.wmei == 1] = 0
         
         self.register_buffer('w0ie', torch.ones([self.he,self.hi]))
         # self.w0ie = torch.ones([self.he,self.hi])
         self.w0ie[self.wpie == 1] = 0
         self.w0ie[self.wmie == 1] = 0
         
         self.register_buffer('w0ii', torch.ones([self.hi,self.hi]))
         # self.w0ii = torch.ones([self.hi,self.hi])
         self.w0ii[self.wpii == 1] = 0
         self.w0ii[self.wmii == 1] = 0 
         
         self.register_buffer('w_ine', torch.zeros([self.he,self.Ns]))
         self.w_ine = torch.zeros([self.he,self.Ns])
         self.w_ine[0:gsze,0]          = 1
         self.w_ine[gsze:2*gsze,1]     = 1
         
         self.register_buffer('w_ini', torch.zeros([self.hi,self.Ns]))
         # self.w_ini = torch.zeros([self.hi,self.Ns])
         if ext_to_inh:
             self.w_ini[0:gszi,0]      = 1
             self.w_ini[gszi:2*gszi,1] = 1
         
         self.register_buffer('w_out', torch.zeros([self.Ns,self.he]))
         # self.w_out = torch.zeros([self.Ns,self.he])
         self.w_out[0,0:gsze]          = 1
         self.w_out[1,gsze:2*gsze]     = 1
         
         
         self.w_ine.requires_grad = False
         self.w_ini.requires_grad = False
         self.w_out.requires_grad = False
         
         self.wpee.requires_grad = False
         self.wpei.requires_grad = False
         self.wpie.requires_grad = False
         self.wpii.requires_grad = False
         
         self.wmee.requires_grad = False
         self.wmei.requires_grad = False
         self.wmie.requires_grad = False
         self.wmii.requires_grad = False
         
         self.w0ee.requires_grad = False
         self.w0ei.requires_grad = False
         self.w0ie.requires_grad = False
         self.w0ii.requires_grad = False
         
     # def prime(self,we0,wi0,win0,wou0,see0,sei0,sie0,sii0,we0_learning,wi0_learning,
     #           win0_learning,wou0_learning,sigma_ee_learning,sigma_ei_learning,
     #           sigma_ie_learning,sigma_ii_learning):
         self.wee0  = nn.Parameter(torch.tensor(wee0))
         self.wei0  = nn.Parameter(torch.tensor(wei0))
         self.wie0  = nn.Parameter(torch.tensor(wie0))
         self.wii0  = nn.Parameter(torch.tensor(wii0))
         self.win0 = nn.Parameter(torch.tensor(win0))
         self.wou0 = nn.Parameter(torch.tensor(wou0))
         
         self.sigma_ee = nn.Parameter(torch.tensor(see0))
         self.sigma_ei = nn.Parameter(torch.tensor(sei0))
         self.sigma_ie = nn.Parameter(torch.tensor(sie0))
         self.sigma_ii = nn.Parameter(torch.tensor(sii0))
         
         self.wee0.requires_grad  = wee0_learning
         self.wei0.requires_grad  = wei0_learning
         self.wie0.requires_grad  = wie0_learning
         self.wii0.requires_grad  = wii0_learning
         self.win0.requires_grad = win0_learning
         self.wou0.requires_grad = wou0_learning
         
         self.sigma_ee.requires_grad = sigma_ee_learning
         self.sigma_ei.requires_grad = sigma_ei_learning
         self.sigma_ie.requires_grad = sigma_ie_learning
         self.sigma_ii.requires_grad = sigma_ii_learning
         
         
     def forward(self,u,init_state_in,init_state_e,init_state_i):
        T = u.shape[1]
        states_in = init_state_in
        states_e  = init_state_e
        states_i  = init_state_i
        states_o  = torch.matmul(torch.relu(states_e),torch.transpose(self.wou0*self.w_out,0,1))
        
        batch_size = states_e.shape[0]
        noise_e = np.sqrt(2*self.alpha*self.sigma**2)*torch.empty(batch_size,T,self.he).normal_(mean=0,std=1).to(device=device)
        noise_i = np.sqrt(2*self.alpha*self.sigma**2)*torch.empty(batch_size,T,self.hi).normal_(mean=0,std=1).to(device=device)
        
        
        self.W_ee = (self.wee0 *self.Ns)/(self.Ns + self.sigma_ee*(2-self.Ns))*((1+self.sigma_ee)*self.wpee + (1-self.sigma_ee)*self.wmee) + self.wee0*self.w0ee
        self.W_ie = (self.wie0 *self.Ns)/(self.Ns + self.sigma_ie*(2-self.Ns))*((1+self.sigma_ie)*self.wpie + (1-self.sigma_ie)*self.wmie) + self.wie0*self.w0ie
        self.W_ei = (self.wie0 *self.Ns)/(self.Ns + self.sigma_ei*(2-self.Ns))*((1+self.sigma_ei)*self.wpei + (1-self.sigma_ei)*self.wmei) + self.wei0*self.w0ei
        self.W_ii = (self.wii0 *self.Ns)/(self.Ns + self.sigma_ii*(2-self.Ns))*((1+self.sigma_ii)*self.wpii + (1-self.sigma_ii)*self.wmii) + self.wii0*self.w0ii
        
        
        w_eeT = torch.transpose(self.W_ee,0,1)
        w_eiT = torch.transpose(self.W_ei,0,1)
        w_ieT = torch.transpose(self.W_ie,0,1)
        w_iiT = torch.transpose(self.W_ii,0,1)
        
        w_ineT = torch.transpose(self.win0*self.w_ine,0,1)
        w_iniT = torch.transpose(self.win0*self.w_ini,0,1)  
        
        for i in range(T-1): 
            state_in_new = (1-self.alpha)*states_in[:,i,:] +\
                self.alpha*(u[:,i,:])
                
            state_e_new  = (1-self.alpha)*states_e[:,i,:] + \
                self.alpha*( torch.matmul(torch.relu(states_e[:,i,:]),w_eeT)  \
                            - torch.matmul(torch.relu(states_i[:,i,:]),w_ieT) \
                            + torch.matmul(states_in[:,i,:],w_ineT)) +  noise_e[:,i,:]
                            
            state_i_new  = (1-self.alpha)*states_i[:,i,:] + \
                self.alpha*( torch.matmul(torch.relu(states_e[:,i,:]),w_eiT) \
                            - torch.matmul(torch.relu(states_i[:,i,:]),w_iiT) \
                            + torch.matmul(states_in[:,i,:],w_iniT)) +  noise_i[:,i,:]
            # state_o_new = torch.matmul(torch.relu(state_e_new),torch.transpose(self.wou0*self.w_out,0,1))
            
            states_in = torch.cat((states_in,state_in_new.unsqueeze_(1)),1) 
            states_e  = torch.cat((states_e,state_e_new.unsqueeze_(1)),1) 
            states_i  = torch.cat((states_i,state_i_new.unsqueeze_(1)),1) 
            # states_o  = torch.cat((states_o,state_o_new.unsqueeze_(1)),1)
        states_o = torch.matmul(torch.relu(states_e),torch.transpose(self.wou0*self.w_out,0,1))
        act_e = torch.relu(states_e)
        act_i = torch.relu(states_i)
        
        return states_o,states_in,act_e,act_i

class gen_trials2():
    def __init__(self,p):
        self.uo    = p["u0"]
        self.sigin = p["sigin"]
        self.scale = p["scale"]
        
        self.upfreq   = p["upfreq"]
        self.downfreq = p["downfreq"]
        
        self.dt_train = p["dt_train"]
        self.dt_test  = p["dt_test"]
        
        
        # self.t_fix  = p["t_fix"] # the fixation period should vary
        self.t_resp = p["t_resp"] # This is the same for all trials
        # self.t_trial = p["t_trial"]
        self.t_catch = p["t_catch"]
        
        self.train_length_trial = int(self.t_trial/self.dt_train)
        self.train_length_catch = int(self.t_catch/self.dt_train)
        
        self.test_length_trial = int(self.t_trial/self.dt_test)
        self.test_length_catch = int(self.t_catch/self.dt_test)
        
        self.tauin = p["tauin"]
        self.taurc = p["taurc"]
        
        self.n_in  = p[""]
        self.n_out = p[""]
        

        if self.t_fix + self.t_resp > 0.5*self.trial:
            print("not enough time for stimulus")
        
    def mk_coh_trial(self,coh,dur,is_train=True):
        
        # I need to know when the stim starts and ends
        t_stim  = self.t_trial - self.t_fix - self.t_resp
        t_start = (t_stim - dur)*np.random.rand() + self.t_fix
        t_end   = t_start + dur
        
        if is_train:
            alphain = self.dt_train/self.tauin
            step_start = int(t_start/self.dt_train)
            step_end   = int(t_end/self.dt_train)
            if self.t_trial <= self.t_catch:
                trial_steps = int(self.t_catch/self.dt_train)
                pad_flag  = True
                pad_steps = int(self.t_catch - self.t_trial)
            else:
                trial_steps = int(self.t_trial/self.dt_train)
                pad_flag = False
        else:
            alphain = self.dt_test/self.tauin
            step_start = int(t_start/self.dt_test)
            step_end   = int(t_end/self.dt_test)
            if self.t_trial <= self.t_catch:
                trial_steps = int(self.t_catch/self.dt_test)
                pad_flag = True
                pad_steps = int(self.t_catch - self.t_trial)
            else:
                trial_steps = int(self.t_trial/self.dt_test)
                pad_flag = False
                
        inputs  = torch.zeros([trial_steps,self.n_in],dtype=torch.float32)
        uo      = self.uo*torch.ones([trial_steps,self.n_in],dtype=torch.float32)
        noise   = (1.0/alphain)*np.sqrt(2.0*alphain*self.sigin**2.0)*torch.ones([trial_steps,self.n_in],dtype=torch.float32).normal_(mean=0,std=1)
        outputs = self.downfreq*torch.ones([trial_steps,self.n_out],dtype=torch.float32)
        mask    = torch.ones([trial_steps,self.n_out],dtype=torch.float32)
        
        mask[step_start:step_end,:] = 0.0
        inputs[step_start:step_end,0] = (1 + self.scale*coh/100)/2
        inputs[step_start:step_end,1] = (1 - self.scale*coh/100)/2
        if coh < 0.0:
            outputs[step_end:,0] = self.upfreq
        elif coh > 0.0:
            outputs[step_end:,1] = self.upfreq
        utot = inputs+uo+noise
        if pad_flag:
            #pad trial to account for difference btwn catch and trial 
            utot[0:pad_steps,:]  = 0.0
            outputs[0:pad_steps,:] = 0.0
            mask[0:pad_steps,:]    = 0.0

        return utot,outputs,noise,step_start,step_end
    
    def mk_catch_trial(self,is_train=True):
        
        if is_train:
            alphain = self.dt_train/self.tauin
            if self.t_trial >= self.t_catch:
                trial_steps = int(self.t_trial/self.dt_train)
                pad_flag  = True
                pad_steps = int(self.t_trial - self.t_catch)
            else:
                trial_steps = int(self.t_catch/self.dt_train)
                pad_flag = False
        else:
            alphain = self.dt_test/self.tauin
            if self.t_trial >= self.t_catch:
                trial_steps = int(self.t_trial/self.dt_test)
                pad_flag = True
                pad_steps = int(self.t_trial - self.t_cactch)
            else:
                trial_steps = int(self.t_catch/self.dt_test)
                pad_flag = False
        
        inputs  = torch.zeros([trial_steps,self.n_in],dtype=torch.float32)
        uo      = self.uo*torch.ones([trial_steps,self.n_in],dtype=torch.float32)
        noise   = (1.0/alphain)*np.sqrt(2.0*alphain*self.sigin**2.0)*torch.ones([trial_steps,self.n_in],dtype=torch.float32).normal_(mean=0,std=1)
        outputs = self.downfreq*torch.ones([trial_steps,self.n_out],dtype=torch.float32)
        mask    = torch.ones([trial_steps,self.n_out],dtype=torch.float32)
        utot = inputs+uo+noise
        if pad_flag:
            #pad trial to account for difference btwn catch and trial 
            utot[0:pad_steps,:]    = 0.0
            outputs[0:pad_steps,:] = 0.0
            mask[0:pad_steps,:]    = 0.0
            
        return utot,outputs,noise
        
    
class gen_trials():
    def __init__(self, trial_len,input_dim,output_dim,uo,sigin,scale,upfreq,downfreq,l,ma,mb,t_final,stim_on,stim_off):
        self.uo       = uo
        self.sigin    = sigin
        self.scale    = scale
        
        self.upfreq   = upfreq
        self.downfreq = downfreq
        
        self.trial_l  = trial_len
        
        self.l        = l
        self.ma       = ma
        self.mb       = mb
        self.t_final  = t_final
        self.stim_on  = stim_on
        self.stim_off = stim_off
        self.st_on    = t_final*stim_on
        self.tstep    = self.t_final/self.trial_l
        
        self.i_streams  = input_dim
        self.o_streams  = output_dim
        self.stim_start = int(np.floor(stim_on*trial_len))
        self.stim_end   = int(np.ceil(stim_off*trial_len))
        
    def mk_coh_trial(self,coh):
        
        alpha_in = 0.2
        
        p  = np.random.rand()
        x  = (-1/self.l)*np.log(np.exp(-self.l*self.ma) - p*(np.exp(-self.l*self.ma) - np.exp(-self.l*self.mb)))
        self.stim_end = int(np.ceil( (self.st_on+x)/self.tstep ))
        
        a = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        b = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        c = torch.ones([self.trial_l], dtype=torch.float32)
        d = torch.ones([self.trial_l], dtype=torch.float32)
        e = torch.zeros([self.trial_l], dtype=torch.float32)
        # a.normal_(self.uo,self.sigin)
        # b.normal_(self.uo,self.sigin)
        c = c*self.downfreq
        d = d*self.downfreq
        
        a[self.stim_start:self.stim_end-1] = a[self.stim_start:self.stim_end-1] + (1 + self.scale*coh/100)/2
        b[self.stim_start:self.stim_end-1] = b[self.stim_start:self.stim_end-1] + (1 + self.scale*-coh/100)/2
        
        if (coh > 0):
            c[self.stim_end:] = self.upfreq
        elif (coh < 0):
            d[self.stim_end:] = self.upfreq
          
        e[0:self.stim_start-1] = 1
        e[self.stim_end:] = 1
        e = e*e.size()[0]/e.sum()
        inps = torch.stack([a,b])
        otps = torch.stack([c,d])
        return inps,otps,e,self.stim_end
    
    def mk_catch_trial(self):
        alpha_in = 0.2
        a = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        b = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        c = torch.ones([self.trial_l], dtype=torch.float32)
        d = torch.ones([self.trial_l], dtype=torch.float32)
        e = torch.ones([self.trial_l], dtype=torch.float32)
        # a.normal_(self.uo,self.sigin)
        # b.normal_(self.uo,self.sigin)
        c = c*self.downfreq
        d = d*self.downfreq
        e = e*e.size()[0]/e.sum()   
        inps = torch.stack([a,b])
        otps = torch.stack([c,d])
        return inps,otps,e
    
    def mk_onehigh_trial(self):
        a = torch.ones([self.trial_l], dtype=torch.float32)
        b = torch.ones([self.trial_l], dtype=torch.float32)
        c = torch.ones([self.trial_l], dtype=torch.float32)
        d = torch.ones([self.trial_l], dtype=torch.float32)
        e = torch.ones([self.trial_l], dtype=torch.float32)
        a.normal_(self.uo,self.sigin)
        b.normal_(self.uo,self.sigin)
        c = c*self.upfreq
        d = d*self.downfreq
        e = e*e.size()[0]/e.sum()   
        inps = torch.stack([a,b])
        otps = torch.stack([c,d])
        return inps,otps,e
    
    def mk_twohigh_trial(self):
        a = torch.ones([self.trial_l], dtype=torch.float32)
        b = torch.ones([self.trial_l], dtype=torch.float32)
        c = torch.ones([self.trial_l], dtype=torch.float32)
        d = torch.ones([self.trial_l], dtype=torch.float32)
        e = torch.ones([self.trial_l], dtype=torch.float32)
        a.normal_(self.uo,self.sigin)
        b.normal_(self.uo,self.sigin)
        c = c*self.downfreq
        d = d*self.upfreq
        e = e*e.size()[0]/e.sum()   
        inps = torch.stack([a,b])
        otps = torch.stack([c,d])
        return inps,otps,e
    
    def mk_fixeddur_trial(self,dur,coh):
        alpha_in = 0.2
        start = self.stim_on + 0.15*np.random.rand()
        end   = start + dur
        
        istart = int(np.floor(start*self.trial_l))
        iend   = int(np.ceil(end*self.trial_l))
        # print('istart: ',istart,' iend: ',iend)
        a = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        b = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        c = torch.ones([self.trial_l], dtype=torch.float32)
        d = torch.ones([self.trial_l], dtype=torch.float32)
        e = torch.zeros([self.trial_l], dtype=torch.float32)
        # a.normal_(self.uo,self.sigin)
        # b.normal_(self.uo,self.sigin)
        c = c*self.downfreq
        d = d*self.downfreq
    
        a[istart:iend-1] = a[istart:iend-1] + (1 + self.scale*coh/100)/2
        b[istart:iend-1] = b[istart:iend-1] + (1 + self.scale*-coh/100)/2
        
        if (coh > 0):
            c[iend:] = self.upfreq
        elif (coh < 0):
            d[iend:] = self.upfreq
          
        e[0:istart-1] = 1
        e[iend:] = 1
        e = e*e.size()[0]/e.sum()
        inps = torch.stack([a,b])
        otps = torch.stack([c,d])
        return inps,otps,e,istart,iend  
    
    def mk_RT_trial(self,dur,coh):
        alpha_in = 0.2
        start = self.stim_on + 0.15*np.random.rand()
        end   = start + dur
        
        istart = int(np.floor(start*self.trial_l))
        iend   = int(np.ceil(end*self.trial_l))
        # print('istart: ',istart,' iend: ',iend)
        a = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        b = self.uo*torch.ones([self.trial_l], dtype=torch.float32) + (1.0/alpha_in)*np.sqrt(2*alpha_in*self.sigin**2.0)*torch.empty([self.trial_l], dtype=torch.float32).normal_(mean=0,std=1)
        c = torch.ones([self.trial_l], dtype=torch.float32)
        d = torch.ones([self.trial_l], dtype=torch.float32)
        e = torch.zeros([self.trial_l], dtype=torch.float32)
        # a.normal_(self.uo,self.sigin)
        # b.normal_(self.uo,self.sigin)
        c = c*self.downfreq
        d = d*self.downfreq
    
        a[istart:] = a[istart:] + (1 + self.scale*coh/100)/2
        b[istart:] = b[istart:] + (1 + self.scale*-coh/100)/2
        
        if (coh > 0):
            c[iend:] = self.upfreq
        elif (coh < 0):
            d[iend:] = self.upfreq
          
        e[0:istart-1] = 1
        e[iend:] = 1
        e = e*e.size()[0]/e.sum()
        inps = torch.stack([a,b])
        otps = torch.stack([c,d])
        return inps,otps,e,istart,iend
    
def loss_fn(pred,target,mask,act,lam_act,model,ex_columns,ih_columns,lam_w):
    a = pred*mask
    b = target*mask
    
    w = model.recurrent_layer.weight.data  
    
#    loss = torch.mean(((b-a)/torch.mean(b))**2) + lam_act*torch.mean(act**2) + lam_w*torch.mean(torch.abs(w)**2.0)
    loss = torch.mean((b-a)**2) + lam_act*torch.mean(act**2) + lam_w*torch.mean(torch.abs(w)**2.0)
    return loss

def loss_fn_seplay(pred,target,mask,act_e,act_i,states_e,states_i,lam_act,lam_x,model,lam_w,mact_tar,wL,rL,xL):
    a = pred*mask
    b = target*mask
    
    w_ee = model.recurrent_layerEE.weight.data 
    w_ei = model.recurrent_layerEI.weight.data 
    w_ie = model.recurrent_layerIE.weight.data 
    w_ii = model.recurrent_layerII.weight.data 
    
    MSE           = torch.mean((b-a)**2.0)
    rate_normE    = lam_act*torch.mean((act_e-mact_tar)**rL)
    rate_normI    = lam_act*torch.mean((act_i-mact_tar)**rL)
    
    state_normE = lam_x*torch.mean((states_e)**xL)
    state_normI = lam_x*torch.mean((states_i)**xL)
    
    weight_normEE = lam_w*torch.mean(torch.abs(w_ee)**wL)
    weight_normEI = lam_w*torch.mean(torch.abs(w_ei)**wL)
    weight_normIE = lam_w*torch.mean(torch.abs(w_ie)**wL)
    weight_normII = lam_w*torch.mean(torch.abs(w_ii)**wL)
    
#    loss = torch.mean(((b-a)/torch.mean(b))**2) + lam_act*torch.mean(act**2) + lam_w*torch.mean(torch.abs(w)**2.0)
    # loss = torch.mean((b-a)**2) + lam_act*torch.mean(act_e**2) + lam_act*torch.mean(act_i**2) + lam_w*torch.mean(torch.abs(w)**2.0)
    loss = MSE + rate_normE + rate_normI + weight_normEE + weight_normEI + \
        weight_normIE + weight_normII + state_normE + state_normI
    return loss,MSE

def loss_fcn_selinh(pred,target,mask,act_e,act_i,lam_act):
    a   = pred*mask
    b   = target*mask
    MSE = torch.mean((b-a)**2)
    rate_normE    = lam_act*torch.mean(act_e**2.0)
    rate_normI    = lam_act*torch.mean(act_i**2.0)
    loss = MSE + rate_normE + rate_normI
    return loss,MSE

def loss_fcn2(pred,target,act_e,act_i,lam_act,trial_starts,trial_ends,catch_lam,choice_lam,rts_lam,cdiff_lam,nchoice_lam,decision_sep,low_sep,trial_length):
    
    # pred = pred.cpu()
    # target = target.cpu()
    # act_e = act_e.cpu()
    # act_i = act_i.cpu()
    
    
    
    # rate_normE    = lam_act*torch.mean(act_e**2.0)
    # rate_normI    = lam_act*torch.mean(act_i**2.0)
    
    rate_normE    = torch.mean(act_e**2.0)
    rate_normI    = torch.mean(act_i**2.0)
    
    rate_tot = (rate_normE + rate_normI)/2.0
    
    dd = pred[:,:,0] - pred[:,:,1] # difference of outputs
    ee = target[:,:,0] - target[:,:,1]
    # before stim
    # catch_low   = torch.tensor(0.0,device='cpu')
    # choice_diff = torch.tensor(0.0,device='cpu')
    # choice      = torch.tensor(0.0,device='cpu')
    # rts         = torch.tensor(0.0,device='cpu')
    # nchoice     = torch.tensor(0.0,device='cpu')
    
    catch_low   = torch.tensor(0.0,device=device)
    choice_diff = torch.tensor(0.0,device=device)
    choice      = torch.tensor(0.0,device=device)
    rts         = torch.tensor(0.0,device=device)
    nchoice     = torch.tensor(0.0,device=device,requires_grad=True)
    
    catch_count    = 0
    choice_count   = 0
    
    for i in range(dd.size()[0]):
        s = trial_starts[i]
        e = trial_ends[i]
        if ((s == 0) &(e==0)): # catch trial!
            # print('catch!! s = ',s,' e = ',e)
            # print(dd.size())
            # test_val1 = torch.sum(torch.abs(dd[i,:]) > low_sep)
            # test_val2 = test_val1.to(dtype)
            # test_val3 = test_val2/trial_length
            
            
            catch_low   += torch.div(torch.sum(torch.abs(dd[i,:]) > low_sep).to(dtype),trial_length)
            catch_count += 1
        else: # not a catch trial
            # print('no catch!!')
            up = (torch.abs(dd[i,0:e]) > decision_sep).nonzero() # was there separation ?
            # print(up)
            if (up.size()[0] != 0): # choice trial
                ww = up[1:] - up[:-1]
                ff = (ww > 1).nonzero()
                if (ff.size()[0] != 0):
                    rt = up[ff[-1][0]+1][0]
                    # print("ff ",rt)
                else:
                    rt = up[0][0]
                    # print("first ", rt)
                if ( rt > s): # valid trial
                    # rts += rt
                    rt = rt.to(dtype)
                    rts += torch.div(rt,(e+1-s)) # make max rt == 1
                    # choice_diff += torch.abs(dd[i,e:-1]).mean() # difference between outs after stim
                    choice_diff += torch.true_divide(torch.sum(torch.abs(dd[i,e:-1]) < decision_sep),(trial_length-e)) # difference between outs after stim
                    if (dd[i,e+1].sign() != ee[i,e+1].sign()):
                        choice += torch.tensor(1.0)
                    choice_count += 1
            else: # invalid trial, no separation
                nchoice += torch.tensor(1.0) # penalize for no separation
    print('NC grad ',nchoice.grad_fn)
    print('NC requires grad?' ,nchoice.requires_grad)
    print('Rate ',rate_tot.grad_fn)
    # average by trials for RT and activity measures
    if (catch_count >= 1):
        catch_low   = torch.true_divide(catch_low,catch_count)
    if (choice_count >= 1):
        rts         = torch.true_divide(rts,choice_count)
    if (choice_count >= 1):
        choice_diff = torch.true_divide(choice_diff,choice_count)
    
    # loss = rate_normE + rate_normI + catch_lam*catch_low + choice_lam*choice + rts_lam*rts \
    #     + cdiff_lam*(1.0/(choice_diff + 0.001)) + nchoice_lam*nchoice
    # loss = rate_normE + rate_normI + catch_lam*catch_low + choice_lam*choice + rts_lam*rts \
    #     + cdiff_lam*choice_diff + nchoice_lam*nchoice
    loss = lam_act*rate_tot + catch_lam*catch_low + choice_lam*choice + rts_lam*rts \
        + cdiff_lam*choice_diff + nchoice_lam*nchoice
    
    # return loss,(rate_normE + rate_normI)/lam_act,catch_low,choice,rts,choice_diff,nchoice
    return loss,rate_tot,catch_low,choice,rts,choice_diff,nchoice

def loss_onlydiff(pred,mask,is_catch,lam_dec,lam_bias):
    mz = mask*pred
    
    ntrials = mz.size(dim=0)
    noncatch = np.sum(is_catch==0)
    
    mdz = torch.zeros([ntrials])
    adz = torch.zeros([ntrials])
    for i in range(ntrials): 
        if is_catch[i]:
            mdz[i] = torch.mean((mz[i,:,0]-mz[i,:,1])**2)
        else:
            mdz1 = torch.mean((mz[i,0:10,0]-mz[i,0:10,1])**2)
            mdz2 = 1.0/(torch.mean((mz[i,-10:,0]- mz[i,-10:,1])**2)+1e-16)
            mdz[i] = mdz1+mdz2
            
            # need a term to reduce bias
            adz[i] = mz[i,-1,0]- mz[i,-1,1]
    dec = torch.mean(mdz)
    bias = torch.abs(torch.sum(adz))**2
    
    loss = lam_dec*torch.mean(mdz) + lam_bias*bias
    
    return loss, dec, bias

def make_loss3_masks(batch_trials_out,is_catch,trial_starts,trial_ends,coh_trials):
    
    dd = batch_trials_out[:,:,0] - batch_trials_out[:,:,1]
    
    during_stim_mask = torch.zeros(dd.size()).to(device)
    trl_end_mask0    = torch.zeros(dd.size()).to(device)
    trl_end_mask1    = torch.zeros(dd.size()).to(device)
    should_low_mask  = torch.zeros(dd.size()).to(device)
    should_high_mask = torch.zeros(dd.size()).to(device)
    
    for q in range(dd.size()[0]):
        during_stim_mask[q,(trial_starts[q]+1):trial_ends[q]] = 1
        
        if is_catch[q] == 0:
            should_low_mask[q,0:trial_starts[q]] = 1
            should_high_mask[q,trial_ends[q]:]  = 1
        else:
            should_low_mask[q,:] = 1
                  
        if coh_trials[q] < 0.0:
            # trl_end_mask0[q,(trial_ends[q]+1):] = 1
            trl_end_mask0[q,-1] = 1
        elif coh_trials[q] > 0.0:
            # trl_end_mask1[q,(trial_ends[q]+1):] = 1
            trl_end_mask1[q,-1] = 1
        elif coh_trials[q] == 0:
            dev = np.random.rand()
            if dev > 0.5:
                # trl_end_mask0[q,(trial_ends[q]+1):] = 1
                trl_end_mask0[q,-1] = 1
            else:
                # trl_end_mask1[q,(trial_ends[q]+1):] = 1
                trl_end_mask1[q,-1] = 1
    
    loss_masks = {}
    loss_masks["during_stim_mask"] = during_stim_mask
    loss_masks["trl_end_mask0"]    = trl_end_mask0
    loss_masks["trl_end_mask1"]    = trl_end_mask1
    loss_masks["should_low_mask"]  = should_low_mask
    loss_masks["should_high_mask"] = should_high_mask
            
    return loss_masks

def loss3(out_pred,act_e,act_i,loss_masks,lams,mini_dex,decision_sep,low_sep):
    dd = out_pred[:,:,0] - out_pred[:,:,1]
    ee = out_pred[:,:,1] - out_pred[:,:,0] 
    
    # abs(dd) below decision threshold
    below_threshA = torch.nn.Threshold(decision_sep,1.0)
    below_threshB = torch.nn.Threshold(-decision_sep,0.0)
    
    below_decA = below_threshA(torch.abs(dd))
    below_decB = below_threshB(-below_decA)
    below_dec  = torch.abs(below_decB)
    
    # abs(dd) above low thresh
    above_threshA = torch.nn.Threshold(low_sep,0.0)
    above_threshB = torch.nn.Threshold(-low_sep,1.0)
    
    above_lowA = above_threshA(torch.abs(dd))
    above_low  = above_threshB(-above_lowA)
    
    # dd > decision_thresh -- 1 when choice 0 is high
    one_threshA = torch.nn.Threshold(decision_sep,0.0)
    one_threshB = torch.nn.Threshold(-decision_sep,1.0)
    
    choose_1A = one_threshA(dd)
    choose_1  = one_threshB(-choose_1A)
    
    # ee > decision_thresh -- 1 when choice 1 is high 
    zero_threshA = torch.nn.Threshold(decision_sep,0.0)
    zero_threshB = torch.nn.Threshold(-decision_sep,1.0)
    
    choose_0A = zero_threshA(ee)
    choose_0  = zero_threshB(-choose_0A)
      
    # Catch and pre
    
    catchNpreA = loss_masks["should_low_mask"][mini_dex,:] * above_low
    per_trial_c = torch.sum(catchNpreA,dim=1)/torch.sum(loss_masks["should_low_mask"][mini_dex,:],dim=1)     
    catchNpre = torch.mean(per_trial_c)
    
    # RT
    
    rtA          = loss_masks["during_stim_mask"][mini_dex,:] * below_dec
    per_trial_rt = torch.sum(rtA,dim=1)/torch.sum(loss_masks["during_stim_mask"][mini_dex,:],dim=1)
    rt           = torch.mean(per_trial_rt)
    
    # Choice
    
    choiceA0          = loss_masks["trl_end_mask0"][mini_dex,:] * choose_0
    choiceA1          = loss_masks["trl_end_mask1"][mini_dex,:] * choose_1      
    per_trial_choice  = torch.sum(choiceA0,dim=1) + torch.sum(choiceA1,dim=1)
    choice            = torch.mean(per_trial_choice)
    
    # Persistant
    
    persistA = loss_masks["should_high_mask"][mini_dex,:] * below_dec
    per_trial_per = torch.sum(persistA,dim=1)/torch.sum(loss_masks["should_high_mask"][mini_dex,:],dim=1)
    persist = torch.mean(per_trial_per)
    
    # activity 
    
    sae = torch.sum(act_e**2)
    sai = torch.sum(act_i**2)
    
    m2_act = (sae + sai)/(act_e.size()[2] + act_i.size()[2])
    
    
    loss = lams["act"]*m2_act + lams["catch"]*catchNpre + lams["rt"]*rt \
        + lams["choice"]*choice + lams["persist"]*persist 
    return loss, m2_act, catchNpre, rt, choice, persist 

def loss_test(out_pred,act_e,act_i,loss_masks,lams,mini_dex,decision_sep,low_sep):
    
    
    c0_diff = out_pred[:,:,0] - out_pred[:,:,1]
    c1_diff = out_pred[:,:,1] - out_pred[:,:,0]
    
    
    # is separation high when it should be
    aa = torch.abs(c0_diff) - decision_sep               # (-) means below decision_sep
    aa = -aa                                             # (-) means above decision_sep 
    bb = aa.clamp(min=0.0,max=1.0)                       # reducing so above decision_sep is zero
    cc = loss_masks["should_high_mask"][mini_dex,:] * bb # mask for when abs(choice should be high)
    rr = torch.sum(cc,dim=1)/(torch.sum(loss_masks["should_high_mask"][mini_dex,:],dim=1)+1)                             # pertrial average
    ee = torch.mean(rr) # so now i have the averagproportion of a trial that is below when it should be above
    
    # is separation low when it should be
    var1 = torch.abs(c0_diff) - low_sep # (-) means below low_sep 
    var2 = var1.clamp(min=0.0,max=1.0)  # reducing so below low_sep is zero
    var3 = loss_masks["should_low_mask"][mini_dex,:] * var2
    var4 = torch.sum(var3,dim=1)/torch.sum(loss_masks["should_low_mask"][mini_dex,:],dim=1)  
    var5 = torch.mean(var4) # the average proportion of a trial where sep is above low_sp and shouldn't
    
    # is separation as fast as possible
    # start with bb which is zero when abs(diff(outs)) is above decision_sep
    ron = loss_masks["during_stim_mask"][mini_dex,:] * bb
    rob = torch.sum(ron,dim=1)/(torch.sum(loss_masks["during_stim_mask"][mini_dex,:],dim=1)+1)  
    ren = torch.mean(rob)
    
    # is the choice correct
    rose  = c0_diff - decision_sep # (-) mean below sep needed to choose 0
    rose  = -rose                  # (-) mean above sep needed to choose 0
    reba  = rose.clamp(min=0.0,max=1.0) # this is zero when the network is choosing 0
    
    wendy  = c1_diff - decision_sep # (-) mean below sep needed to choose 1
    wendy  = -wendy
    warren = wendy.clamp(min=0.0,max=1.0)
    
    choiceA0          = loss_masks["trl_end_mask0"][mini_dex,:] * warren
    choiceA1          = loss_masks["trl_end_mask1"][mini_dex,:] * reba    
    per_trial_choice  = (torch.sum(choiceA0,dim=1) + torch.sum(choiceA1,dim=1))
    choice            = torch.mean(per_trial_choice)
    
    sae = torch.sum(act_e**2)
    sai = torch.sum(act_i**2)
    
    m2_act = (sae + sai)/(act_e.size()[2] + act_i.size()[2])
    
    mout_act = torch.mean(torch.sum(out_pred,dim=2)) 
    out_bias = torch.abs(torch.mean(loss_masks["should_high_mask"][mini_dex,:]*c0_diff))
    
    # smoothness
    if (lams["smoothness"] > 0):
        # summer = (out_pred[:,1:,:] - out_pred[:,:-1,:])**2
        # fall   = (out_pred**2).mean(dim=1)
        # winter = summer.sum(dim=1)/fall
        # spring = winter.mean()
        summer = out_pred[:,:,1]-out_pred[:,:,0]
        fall = (summer[:,1:] - summer[:,:-1])**2
        winter = fall.sum(dim=1)
        spring = winter.mean()
    else:
        spring = torch.tensor(0.0,device=device)
    
    loss = lams["act"]*m2_act + lams["persist"]*ee + \
        lams["catch"]*var5 + lams["rt"]*ren + lams["choice"]*choice + \
        lams["out_rate"]*mout_act + lams["antibias"]*out_bias + \
        lams["smoothness"]*spring
    
    return loss, m2_act, ee, var5, ren, choice, mout_act, out_bias, spring
 

def loss_omega(mse,dldx,act_e,Wee,alpha):
   ntrial  = dldx.size()[0]
   ntime   = dldx.size()[1]
   nneuron = dldx.size()[2]
   
   # omegas = torch.zeros([ntrial,ntime])
   # ok, here i take the first 5 and last five, this prevents getting zeros for 
   # parts of the trial that are masked from MSE
   aa_late  = act_e[:,-5:,:]
   aa_early = act_e[:,0:5,:]
   # offset activity by 1 step
   bb = torch.zeros([ntrial,1,nneuron])
   cc_late  = torch.cat([bb,aa_late],1)
   cc_early = torch.cat([bb,aa_early],1)
   
   rprime_late  = torch.zeros(cc_late.size())
   rprime_early = torch.zeros(cc_early.size())
   a_late = cc_late >  0
   a_early = cc_early >  0
   # b = act_e <= 0
   rprime_late[a_late] = 1 
   rprime_early[a_early] = 1 
   
   num1_l = (1.0 - alpha)*dldx[:,-6:,:] + alpha*torch.matmul(dldx[:,-6:,:],Wee)*rprime_late
   num2_l = torch.norm(num1_l,dim=2)**2
   den_l  = torch.norm(dldx[:,-6:,:],dim=2)**2
   
   num1_e = (1.0 - alpha)*dldx[:,0:6,:] + alpha*torch.matmul(dldx[:,0:6,:],Wee)*rprime_early
   num2_e = torch.norm(num1_e,dim=2)**2
   den_e  = torch.norm(dldx[:,0:6,:],dim=2)**2
   
   
   omegas = torch.cat([torch.sum((num2_e/(den_e+1e-12) - 1.0)**2,dim=1),torch.sum((num2_l/(den_l+1e-12) - 1.0)**2,dim=1)],0)
   omega = torch.mean(omegas)
   
   
   # for j in range(ntrial):
   #     for i in range(1,ntime):
   #         der       = dldx[j,i,:].unsqueeze(0)
   #         num1      = (1.0 - alpha)*der  + alpha*torch.matmul(der,Wee)*rprime[j,i,:].unsqueeze(0)
   #         # num1      = (1.0 - alpha)*der  + alpha*torch.matmul(der,Wee)
   #         num2      = torch.norm(num1)**2
   #         den       = torch.norm(der)**2
   #         omegas[j,i] = ((num2/(den+1e-8)) - 1.0)**2
   # omega = torch.mean(omegas) 
   return omega 

def loss_song(mse,omega,act_e,act_i,Wee,Wei,Wie,wii0,lm,lo,lr,lw):
    cost = lm*mse + lo*omega + lr*torch.mean(act_e**2) + lr*torch.mean(act_i**2) + lw*torch.mean(torch.abs(Wee)) \
        + lw*torch.mean(torch.abs(Wei))  + lw*torch.mean(torch.abs(Wie))  + lw*wii0  
    return cost

def omega_trials(out_pred,mask,states_e,target,alpha,Wee):
    ntrial = out_pred.size()[0]
    ntime  = out_pred.size()[1]
    
    rprime = torch.zeros(states_e.size(),device=out_pred.device)
    a = states_e > 0.0
    rprime[a] = 1.0
    omegas = torch.zeros([ntrial,ntime],device=out_pred.device)
    
    for j in range(ntrial):
        mse = torch.mean((mask[j,:,:]*out_pred[j,:,:] - mask[j,:,:]*target[j,:,:])**2)
        mse.backward(retain_graph=True)
        dldx = torch.autograd.grad(mse,states_e,retain_graph=True,allow_unused=True)[0]
        for i in range(1,ntime):
            if mask[j,i,0] != 0.0:            
                der = dldx[j,i,:].unsqueeze(0)
                num1 = (1.0-alpha)*der + alpha*torch.matmul(der,Wee)*rprime[j,i-1,:]
                num  = torch.norm(num1)**2
                den  = torch.norm(der)**2
                # print(num,den)
                if den.data.cpu().detach().numpy() !=0:
                    omegas[j,i] = ((num/den)-1.0)**2
                else:
                    omegas[j,i] = (-1.0)**2
                
    omega_trial = torch.sum(omegas,dim=1)
    omega = torch.mean(omega_trial)
  
    # print("hey")
    
    return omega

def mk_init_w(w_mu,w_sig,inh_init_scale,h1,he,hi,inputs,outputs,device,dtype,dtype_dex,impose_inputs,f):
    xx = np.random.gamma(w_mu,w_sig,[h1,he])
    yy = -1.0*np.random.gamma(inh_init_scale*w_mu,w_sig,[h1,hi])
    zz = np.concatenate((xx, yy), axis=1)
    aa = np.arange(h1)
    zz[aa,aa] = 0.0
    
#    W_rec = torch.from_numpy(zz).to(dtype).to(device)
    W_rec = torch.from_numpy(zz).to(device).to(dtype)
    
    e_col = torch.arange(0,he,dtype=dtype_dex).to(device)
    i_col = torch.arange(he,he+hi,dtype=dtype_dex).to(device)
    
    ex_w = torch.zeros(h1,h1).type(dtype_dex).to(device)
    ex_w[:,e_col] = 1
    ih_w = torch.zeros(h1,h1).type(dtype_dex).to(device)
    ih_w[:,i_col] = 1
    
    if impose_inputs:
        W_in = torch.ones([h1,inputs],dtype=dtype).to(device)
        W_out = torch.ones([outputs,h1],dtype=dtype).to(device)
        
        gp_sz = int(np.ceil(f*he))
        
        group1 = torch.arange(0,gp_sz,dtype=dtype_dex).to(device)
        group2 = torch.arange(gp_sz,2*gp_sz,dtype=dtype_dex).to(device)
        
        W_in[group1,0] = 2.0
        W_in[group1,1] = 0.0
        W_in[group2,0] = 0.0
        W_in[group2,1] = 2.0
        
        W_in[i_col,:]  = 0.0
        W_out[:,i_col] = 0.0
        
    else:
        W_in = torch.rand([h1,inputs],dtype=dtype).to(device)
        W_out = torch.rand([outputs,h1],dtype=dtype).to(device)
        W_in[i_col,:]  = 0.0
        W_out[:,i_col] = 0.0
    
    return W_rec,W_in,W_out,e_col,i_col,ex_w,ih_w

def mk_init_w_layers(w_mu,w_sig,inh_init_scale,he,hi,inputs,outputs,device,dtype,dtype_dex,impose_inputs,impose_outputs,f):
    
    #make w_ee
    xx = np.random.gamma(w_mu,w_sig,[he,he]) # this is an array of random gamma deviates
    aa = np.arange(he)
    xx[aa,aa] = 0.0
    W_ee = torch.from_numpy(xx).to(device=device,dtype=dtype)
    
    #make w_ei
    xx = np.random.gamma(w_mu,w_sig,[hi,he]) # this is an array of random gamma deviates
    W_ei = torch.from_numpy(xx).to(device=device,dtype=dtype)
    
    #make w_ie
    xx = np.random.gamma(inh_init_scale*w_mu,w_sig,[he,hi]) # this is an array of random gamma deviates
    W_ie = torch.from_numpy(xx).to(device=device,dtype=dtype)
    
    #make w_ii
    xx = np.random.gamma(inh_init_scale*w_mu,w_sig,[hi,hi]) # this is an array of random gamma deviates
    aa = np.arange(hi)
    xx[aa,aa] = 0.0
    W_ii = torch.from_numpy(xx).to(device=device,dtype=dtype)
    
    if impose_inputs & impose_outputs:
        W_ine = torch.ones([he,inputs],dtype=dtype).to(device)
        W_ini = torch.zeros([hi,inputs],dtype=dtype).to(device)
        W_out = torch.ones([outputs,he],dtype=dtype).to(device)
        
        gp_sz = int(np.ceil(f*he))
        
        group1 = torch.arange(0,gp_sz,dtype=dtype_dex).to(device)
        group2 = torch.arange(gp_sz,2*gp_sz,dtype=dtype_dex).to(device)
        
        W_ine[group1,0] = 2.0
        W_ine[group1,1] = 0.0
        W_ine[group2,0] = 0.0
        W_ine[group2,1] = 2.0
        
        W_out[0,group1] = 2.0
        W_out[0,group2] = 0.0
        W_out[1,group1] = 0.0
        W_out[1,group2] = 2.0
        
        a = W_ine
        b = torch.sum(a,dim=1)
        b = torch.unsqueeze(b,1)
        W_ine = 1.0*a/b
        
        a = W_out
        b = torch.sum(a,dim=1)
        b = torch.unsqueeze(b,1)
        W_out = 1.0*a/b
        
        
        
    elif impose_inputs:
        W_ine = torch.ones([he,inputs],dtype=dtype).to(device)
        W_ini = torch.zeros([hi,inputs],dtype=dtype).to(device)
        W_out = torch.rand([outputs,he],dtype=dtype).to(device)
        
        gp_sz = int(np.ceil(f*he))
        
        group1 = torch.arange(0,gp_sz,dtype=dtype_dex).to(device)
        group2 = torch.arange(gp_sz,2*gp_sz,dtype=dtype_dex).to(device)
        
        W_ine[group1,0] = 2.0
        W_ine[group1,1] = 0.0
        W_ine[group2,0] = 0.0
        W_ine[group2,1] = 2.0
        
        a = W_ine
        b = torch.sum(a,dim=1)
        b = torch.unsqueeze(b,1)
        W_ine = 1.0*a/b
        
        a = W_out
        b = torch.sum(a,dim=1)
        b = torch.unsqueeze(b,1)
        W_out = 1.0*a/b
        
    elif impose_outputs:
        W_ine = torch.rand([he,inputs],dtype=dtype).to(device)
        W_ini = torch.zeros([hi,inputs],dtype=dtype).to(device)
        W_out = torch.ones([outputs,he],dtype=dtype).to(device)
        
        gp_sz = int(np.ceil(f*he))
        
        group1 = torch.arange(0,gp_sz,dtype=dtype_dex).to(device)
        group2 = torch.arange(gp_sz,2*gp_sz,dtype=dtype_dex).to(device)
        
        W_out[0,group1] = 2.0
        W_out[0,group2] = 0.0
        W_out[1,group1] = 0.0
        W_out[1,group2] = 2.0
        
        a = W_ine
        b = torch.sum(a,dim=1)
        b = torch.unsqueeze(b,1)
        W_ine = 1.0*a/b
        
        a = W_out
        b = torch.sum(a,dim=1)
        b = torch.unsqueeze(b,1)
        W_out = 1.0*a/b
        
    else:
        # W_ine = torch.rand([he,inputs],dtype=dtype).to(device)
        # W_ine = W_ine/torch.sum(W_ine,dim=0)
        # W_ini = torch.rand([hi,inputs],dtype=dtype).to(device)
        # W_ini = W_ini/torch.sum(W_ini,dim=0)
        # W_out = torch.rand([outputs,he],dtype=dtype).to(device)
        # W_out = W_out/torch.sum(W_out,dim=1)
        
        # W_ine = 1.0*torch.rand([he,inputs],dtype=dtype).to(device)
        # W_ini = 1.0*torch.rand([hi,inputs],dtype=dtype).to(device)
        # W_out = 1.0*torch.rand([outputs,he],dtype=dtype).to(device)
        
        a = torch.rand([he,inputs],dtype=dtype).to(device)
        b = torch.sum(a,dim=0)
        b = torch.unsqueeze(b,0)
        W_ine = 1.0*a/b
        a = torch.rand([hi,inputs],dtype=dtype).to(device)
        b = torch.sum(a,dim=0)
        b = torch.unsqueeze(b,0)
        W_ini = 0.0*a/b
        a = torch.rand([outputs,he],dtype=dtype).to(device)
        b = torch.sum(a,dim=1)
        b = torch.unsqueeze(b,1)
        W_out = 1.0*a/b
            
    return W_ee,W_ei,W_ie,W_ii,W_ine,W_ini,W_out
def mk_selinh_w_layers(we0,wi0,w0in,w0out,Ns,sigma_ee,sigma_ei,sigma_ie,sigma_ii,he,hi,device,dtype,dtype_dex,f):
    
    
    # make recurrent weights
    what_ee = Ns*we0/(Ns + sigma_ee*(2-Ns))
    what_ei = Ns*we0/(Ns + sigma_ei*(2-Ns))
    what_ie = Ns*wi0/(Ns + sigma_ie*(2-Ns))
    what_ii = Ns*wi0/(Ns + sigma_ii*(2-Ns))
    
    
    wpee = what_ee + sigma_ee*what_ee
    wmee = what_ee - sigma_ee*what_ee
    
    print(wpee)
    
    wpei = what_ei + sigma_ei*what_ei
    wmei = what_ei - sigma_ei*what_ei
    
    wpie = what_ie + sigma_ie*what_ie
    wmie = what_ie - sigma_ie*what_ie
    
    wpii = what_ii + sigma_ii*what_ii
    wmii = what_ii - sigma_ii*what_ii
    
    gp_sze = int(np.ceil(f*he))
    gp_szi = int(np.ceil(f*hi))
    
    
    # gp1e = torch.arange(0,gp_sze,dtype=dtype_dex).to(device)
    # gp1i = torch.arange(0,gp_szi,dtype=dtype_dex).to(device)
    # gp2e = torch.arange(gp_sze,2*gp_sze,dtype=dtype_dex).to(device)
    # gp2i = torch.arange(gp_szi,2*gp_szi,dtype=dtype_dex).to(device)
    
    group1e = torch.arange(0,gp_sze,dtype=dtype_dex).to(device)
    group1i = torch.arange(0,gp_szi,dtype=dtype_dex).to(device)

    group2e = torch.arange(gp_sze,2*gp_sze,dtype=dtype_dex).to(device)
    group2i = torch.arange(gp_szi,2*gp_szi,dtype=dtype_dex).to(device)
    
    print(group1e), print(group2e)
    
    # group1e[gp1e] = 1
    # group1i[gp1i] = 1
    # group2e[gp2e] = 1
    # group2i[gp2i] = 1
    
    W_ee = we0*torch.ones([he,he],dtype=dtype).to(device)
    W_ei = we0*torch.ones([hi,he],dtype=dtype).to(device)
    W_ie = wi0*torch.ones([he,hi],dtype=dtype).to(device)
    W_ii = wi0*torch.ones([hi,hi],dtype=dtype).to(device)
    
    W_ee[0:gp_sze,               0:gp_sze] = wpee
    W_ee[0:gp_sze,        gp_sze:2*gp_sze] = wmee
    W_ee[gp_sze:2*gp_sze,        0:gp_sze] = wmee
    W_ee[gp_sze:2*gp_sze, gp_sze:2*gp_sze] = wpee
    
    W_ei[0:gp_szi,               0:gp_sze] = wpei
    W_ei[0:gp_szi,        gp_sze:2*gp_sze] = wmei
    W_ei[gp_szi:2*gp_szi,        0:gp_sze] = wmei
    W_ei[gp_szi:2*gp_szi, gp_sze:2*gp_sze] = wpei
    
    W_ie[0:gp_sze,               0:gp_szi] = wpie
    W_ie[0:gp_sze,        gp_szi:2*gp_szi] = wmie
    W_ie[gp_sze:2*gp_sze,        0:gp_szi] = wmie
    W_ie[gp_sze:2*gp_sze, gp_szi:2*gp_szi] = wpie
    
    W_ii[0:gp_szi,               0:gp_szi] = wpii
    W_ii[0:gp_szi,        gp_szi:2*gp_szi] = wmii
    W_ii[gp_szi:2*gp_szi,        0:gp_szi] = wmii
    W_ii[gp_szi:2*gp_szi, gp_szi:2*gp_szi] = wpii
    
    #input and output layers
    W_ine = 0.000*torch.zeros([he,Ns],dtype=dtype).to(device)
    W_ini = 0.000*torch.zeros([hi,Ns],dtype=dtype).to(device)
    W_out = 0.001*torch.ones([Ns,he],dtype=dtype).to(device) 
    
    W_ine[group1e,0] = 1
    W_ine[group2e,1] = 1
    
    # W_ini[group1i,0] = 1
    # W_ini[group2i,1] = 1
    
    W_out[0,group1e] = 1
    W_out[1,group2e] = 1
    
    return W_ee,W_ei,W_ie,W_ii,W_ine,W_ini,W_out
    
def make_training_batch(trial_maker,batch_size,trial_length,inputs,outputs,device,catch_freq,difficulty1,difficulty2):
    batch_trials_inp  = torch.zeros([batch_size,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    batch_trials_out  = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    batch_trials_mask = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    batch_trials_inp = batch_trials_inp.to(device)
    batch_trials_out = batch_trials_out.to(device)
    batch_trials_mask = batch_trials_mask.to(device)
    
    
    is_catch   = np.zeros([batch_size])
    coh_trials = np.zeros([batch_size])
    trial_ends = np.zeros([batch_size],dtype= np.long)
    
    for i in range(0,batch_size):
        a = np.random.rand()
        if a < (catch_freq):
            inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
            is_catch[i] = 1
            coh_trials[i] = np.nan
        else:
            chce = np.random.rand()
            if chce >=0.5:
                b = difficulty2 - (difficulty2-difficulty1)*np.random.rand()
            else:
                b = -1.0*(difficulty2 - (difficulty2-difficulty1)*np.random.rand())
            
            # b = -100 + 200*np.random.rand()
    #        if b >= 0:
    #            b = 100
    #        else:
    #            b = -100
            inp_streams,out_streams,mask,e = trial_maker.mk_coh_trial(b)
            trial_ends[i] = e
    #        print(e)
    #        print(trial_ends[i])
            coh_trials[i] = b
        msk = torch.stack([mask,mask])
        batch_trials_inp[i,:,:] = torch.transpose(inp_streams,0,1)
        batch_trials_out[i,:,:] = torch.transpose(out_streams,0,1)
        batch_trials_mask[i,:,:] = torch.transpose(msk,0,1)
    return batch_trials_inp, batch_trials_out, batch_trials_mask, is_catch, coh_trials, trial_ends 

def make_unstim_training_batch(trial_maker,batch_size,trial_length,inputs,outputs,device,low_freq,one_high_freq):
    batch_trials_inp  = torch.zeros([batch_size,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    batch_trials_out  = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    batch_trials_mask = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    batch_trials_inp = batch_trials_inp.to(device)
    batch_trials_out = batch_trials_out.to(device)
    batch_trials_mask = batch_trials_mask.to(device)
    
    trial_type = np.zeros([batch_size])
    
    for i in range(0,batch_size):
        a = np.random.rand()
        if a < low_freq:
            inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
            trial_type[i] = 0
        elif (a >= low_freq) & (a<(low_freq+one_high_freq)):
            inp_streams,out_streams,mask = trial_maker.mk_onehigh_trial()
            trial_type[i] = 1
        else:
            inp_streams,out_streams,mask = trial_maker.mk_twohigh_trial()
            trial_type[i] = 2

        msk = torch.stack([mask,mask])
        batch_trials_inp[i,:,:] = torch.transpose(inp_streams,0,1)
        batch_trials_out[i,:,:] = torch.transpose(out_streams,0,1)
        batch_trials_mask[i,:,:] = torch.transpose(msk,0,1)
    return batch_trials_inp, batch_trials_out, batch_trials_mask, trial_type


def make_training_batch2(trial_maker,batch_size,trial_length,inputs,outputs,device,catch_freq,dur,difficulty1,difficulty2,side_bias):
    batch_trials_inp  = torch.zeros([batch_size,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    batch_trials_out  = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    batch_trials_mask = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    batch_trials_inp = batch_trials_inp.to(device)
    batch_trials_out = batch_trials_out.to(device)
    batch_trials_mask = batch_trials_mask.to(device)
    
    is_catch     = np.zeros([batch_size])
    coh_trials   = np.zeros([batch_size])
    trial_starts = np.zeros([batch_size],dtype= np.long)
    trial_ends   = np.zeros([batch_size],dtype= np.long)
    
    # make trial outcome vector
    
    catch_trials = int(catch_freq*batch_size)
    normal_trials = batch_size - catch_trials
    zero_trials = int((0.5-side_bias)*normal_trials)
    one_trials  = int(normal_trials-zero_trials)
    
    a = np.ones(batch_size)
    a[0:(catch_trials-1)] = -1
    a[catch_trials:(catch_trials+zero_trials)] = 0
    
    for i in range(0,batch_size):
        # a = np.random.rand()
        if a[i] == -1: # catch_trials
            inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
            is_catch[i] = 1
            coh_trials[i] = np.nan
        elif a[i] == 0:
            b = difficulty2 - (difficulty2-difficulty1)*np.random.rand()
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_fixeddur_trial(dur,b)
            # print('istart: ',istart,' iend: ',iend)
            trial_starts[i] = istart
            trial_ends[i] = iend
            coh_trials[i] = b
        else:
            b = -1.0*(difficulty2 - (difficulty2-difficulty1)*np.random.rand())
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_fixeddur_trial(dur,b)
            # print('istart: ',istart,' iend: ',iend)
            trial_starts[i] = istart
            trial_ends[i] = iend
            coh_trials[i] = b
        
        msk = torch.stack([mask,mask])
        batch_trials_inp[i,:,:] = torch.transpose(inp_streams,0,1)
        batch_trials_out[i,:,:] = torch.transpose(out_streams,0,1)
        batch_trials_mask[i,:,:] = torch.transpose(msk,0,1)
    return batch_trials_inp, batch_trials_out, batch_trials_mask, is_catch, coh_trials, trial_starts, trial_ends

def make_training_batch3(trial_maker,batch_size,trial_length,inputs,outputs,device,catch_freq,dur,binns,side_bias):
    batch_trials_inp  = torch.zeros([batch_size,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    batch_trials_out  = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    batch_trials_mask = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    batch_trials_inp = batch_trials_inp.to(device)
    batch_trials_out = batch_trials_out.to(device)
    batch_trials_mask = batch_trials_mask.to(device)
    
    is_catch     = np.zeros([batch_size])
    coh_trials   = np.zeros([batch_size])
    trial_starts = np.zeros([batch_size],dtype= np.long)
    trial_ends   = np.zeros([batch_size],dtype= np.long)
    
    # make trial outcome vector
    
    catch_trials = int(catch_freq*batch_size)
    normal_trials = batch_size - catch_trials
    zero_trials = int((0.5-side_bias)*normal_trials)
    one_trials  = int(normal_trials-zero_trials)
    
    a = np.ones(batch_size)
    a[0:(catch_trials-1)] = -1
    a[catch_trials:(catch_trials+zero_trials)] = 0
    
    difu = np.unique(np.abs(binns))
    difu = difu[difu!=0]
    for i in range(0,batch_size):
        # a = np.random.rand()
        if a[i] == -1: # catch_trials
            inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
            is_catch[i] = 1
            coh_trials[i] = np.nan
        elif a[i] == 0:
            b = np.random.choice(difu)
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_fixeddur_trial(dur,b)
            # print('istart: ',istart,' iend: ',iend)
            trial_starts[i] = istart
            trial_ends[i] = iend
            coh_trials[i] = b
        else:
            b = -1.0*np.random.choice(difu)
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_fixeddur_trial(dur,b)
            # print('istart: ',istart,' iend: ',iend)
            trial_starts[i] = istart
            trial_ends[i] = iend
            coh_trials[i] = b
        
        msk = torch.stack([mask,mask])
        batch_trials_inp[i,:,:] = torch.transpose(inp_streams,0,1)
        batch_trials_out[i,:,:] = torch.transpose(out_streams,0,1)
        batch_trials_mask[i,:,:] = torch.transpose(msk,0,1)
    return batch_trials_inp, batch_trials_out, batch_trials_mask, is_catch, coh_trials, trial_starts, trial_ends

def make_training_batch_RT(trial_maker,batch_size,trial_length,inputs,outputs,device,catch_freq,dur,binns,side_bias):
    batch_trials_inp  = torch.zeros([batch_size,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    batch_trials_out  = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    batch_trials_mask = torch.zeros([batch_size,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    batch_trials_inp = batch_trials_inp.to(device)
    batch_trials_out = batch_trials_out.to(device)
    batch_trials_mask = batch_trials_mask.to(device)
    
    is_catch     = np.zeros([batch_size])
    coh_trials   = np.zeros([batch_size])
    trial_starts = np.zeros([batch_size],dtype= np.long)
    trial_ends   = np.zeros([batch_size],dtype= np.long)
    
    # make trial outcome vector
    
    catch_trials = int(catch_freq*batch_size)
    normal_trials = batch_size - catch_trials
    zero_trials = int((0.5-side_bias)*normal_trials)
    one_trials  = int(normal_trials-zero_trials)
    
    a = np.ones(batch_size)
    a[0:(catch_trials-1)] = -1
    a[catch_trials:(catch_trials+zero_trials)] = 0
    
    difu = np.unique(np.abs(binns))
    difu = difu[difu!=0]
    for i in range(0,batch_size):
        # a = np.random.rand()
        if a[i] == -1: # catch_trials
            inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
            is_catch[i] = 1
            coh_trials[i] = np.nan
        elif a[i] == 0:
            b = np.random.choice(difu)
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_RT_trial(dur,b)
            # print('istart: ',istart,' iend: ',iend)
            trial_starts[i] = istart
            trial_ends[i] = iend
            coh_trials[i] = b
        else:
            b = -1.0*np.random.choice(difu)
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_RT_trial(dur,b)
            # print('istart: ',istart,' iend: ',iend)
            trial_starts[i] = istart
            trial_ends[i] = iend
            coh_trials[i] = b
        
        msk = torch.stack([mask,mask])
        batch_trials_inp[i,:,:] = torch.transpose(inp_streams,0,1)
        batch_trials_out[i,:,:] = torch.transpose(out_streams,0,1)
        batch_trials_mask[i,:,:] = torch.transpose(msk,0,1)
    return batch_trials_inp, batch_trials_out, batch_trials_mask, is_catch, coh_trials, trial_starts, trial_ends

def make_validation_batch(trial_maker,batch_size_v,trial_length,inputs,outputs,device,nbinns,psych_itr,binns):
    val_trials_inp  = torch.zeros([batch_size_v,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    val_trials_out  = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    val_trials_mask = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    val_trials_inp = val_trials_inp.to(device)
    val_trials_out = val_trials_out.to(device)
    val_trials_mask = val_trials_mask.to(device)
    
    coh_trials_v = np.zeros([batch_size_v])
    bin_trials_v = np.zeros([batch_size_v],dtype=int)
    trial_ends_v = np.zeros([batch_size_v],dtype= np.long)
    
    #binns  = np.arange(-40,45,5)
    
    batch_counter = 0
    
    for i in range(0,nbinns):
        for j in range(0,psych_itr):
        #    a = np.random.rand()
        #    if a < (catch_freq):
        #        inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
        #        is_catch_v[i] = 1
        #        coh_trials_v[i] = np.nan
        #    else:
        #    b = -100 + 200*np.random.rand()
        #    b = np.random.randint(np.size(binns))
        #        if b >= 0:
        #            b = 100
        #        else:
        #            b = -100
            inp_streams,out_streams,mask,e = trial_maker.mk_coh_trial(binns[i])
            trial_ends_v[batch_counter] = e
        #    print(e)
        #    print(trial_ends_v[i])
            coh_trials_v[batch_counter] = binns[i]
            bin_trials_v[batch_counter] = i
            msk = torch.stack([mask,mask])
            val_trials_inp[batch_counter,:,:] = torch.transpose(inp_streams,0,1)
            val_trials_out[batch_counter,:,:] = torch.transpose(out_streams,0,1)
            val_trials_mask[batch_counter,:,:] = torch.transpose(msk,0,1)
            batch_counter += 1
    return val_trials_inp, val_trials_out, val_trials_mask, coh_trials_v, bin_trials_v, trial_ends_v

def make_validation_batch2(trial_maker,batch_size_v,trial_length,inputs,outputs,device,nbinns,psych_itr,binns):
    val_trials_inp  = torch.zeros([batch_size_v,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    val_trials_out  = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    val_trials_mask = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    val_trials_inp = val_trials_inp.to(device)
    val_trials_out = val_trials_out.to(device)
    val_trials_mask = val_trials_mask.to(device)
    
    coh_trials_v = np.zeros([batch_size_v])
    bin_trials_v = np.zeros([batch_size_v],dtype=int)
    trial_ends_v = np.zeros([batch_size_v],dtype= np.long)
    
    #binns  = np.arange(-40,45,5)
    
    batch_counter = 0
    
    for i in range(0,nbinns):
        for j in range(0,psych_itr):
        #    a = np.random.rand()
        #    if a < (catch_freq):
        #        inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
        #        is_catch_v[i] = 1
        #        coh_trials_v[i] = np.nan
        #    else:
        #    b = -100 + 200*np.random.rand()
        #    b = np.random.randint(np.size(binns))
        #        if b >= 0:
        #            b = 100
        #        else:
        #            b = -100
            inp_streams,out_streams,mask,e = trial_maker.mk_coh_trial(binns[i])
            trial_ends_v[batch_counter] = e
        #    print(e)
        #    print(trial_ends_v[i])
            coh_trials_v[batch_counter] = binns[i]
            bin_trials_v[batch_counter] = i
            msk = torch.stack([mask,mask])
            val_trials_inp[batch_counter,:,:] = torch.transpose(inp_streams,0,1)
            val_trials_out[batch_counter,:,:] = torch.transpose(out_streams,0,1)
            val_trials_mask[batch_counter,:,:] = torch.transpose(msk,0,1)
            batch_counter += 1
    return val_trials_inp, val_trials_out, val_trials_mask, coh_trials_v, bin_trials_v, trial_ends_v

def make_validation_batch2_fixeddur(trial_maker,batch_size_v,trial_length,inputs,outputs,device,nbinns,psych_itr,binns,dur):
    val_trials_inp  = torch.zeros([batch_size_v,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    val_trials_out  = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    val_trials_mask = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    val_trials_inp = val_trials_inp.to(device)
    val_trials_out = val_trials_out.to(device)
    val_trials_mask = val_trials_mask.to(device)
    
    coh_trials_v = np.zeros([batch_size_v])
    bin_trials_v = np.zeros([batch_size_v],dtype=int)
    trial_starts = np.zeros([batch_size_v],dtype= np.long)
    trial_ends   = np.zeros([batch_size_v],dtype= np.long)
    
    #binns  = np.arange(-40,45,5)
    
    batch_counter = 0
    
    for i in range(0,nbinns):
        for j in range(0,psych_itr):
        #    a = np.random.rand()
        #    if a < (catch_freq):
        #        inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
        #        is_catch_v[i] = 1
        #        coh_trials_v[i] = np.nan
        #    else:
        #    b = -100 + 200*np.random.rand()
        #    b = np.random.randint(np.size(binns))
        #        if b >= 0:
        #            b = 100
        #        else:
        #            b = -100
            #inp_streams,out_streams,mask,e = trial_maker.mk_coh_trial(binns[i])
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_fixeddur_trial(dur,binns[i])
            # trial_ends_v[batch_counter] = e
            
            trial_starts[batch_counter] = istart
            trial_ends[batch_counter]   = iend
            
        #    print(e)
        #    print(trial_ends_v[i])
            coh_trials_v[batch_counter] = binns[i]
            bin_trials_v[batch_counter] = i
            msk = torch.stack([mask,mask])
            val_trials_inp[batch_counter,:,:] = torch.transpose(inp_streams,0,1)
            val_trials_out[batch_counter,:,:] = torch.transpose(out_streams,0,1)
            val_trials_mask[batch_counter,:,:] = torch.transpose(msk,0,1)
            batch_counter += 1
    return val_trials_inp, val_trials_out, val_trials_mask, coh_trials_v, bin_trials_v, trial_starts, trial_ends 

def make_validation_batch2_RT(trial_maker,batch_size_v,trial_length,inputs,outputs,device,nbinns,psych_itr,binns,dur):
    val_trials_inp  = torch.zeros([batch_size_v,trial_length,inputs],dtype=torch.float32) # this should be batch x trial_length x inputs. CHANGED TO TRIAL_L X BATCH X OUTS; no difference relu, worked tanh 
    val_trials_out  = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    val_trials_mask = torch.zeros([batch_size_v,trial_length,outputs],dtype=torch.float32)
    
    #send tensors to GPU
    val_trials_inp = val_trials_inp.to(device)
    val_trials_out = val_trials_out.to(device)
    val_trials_mask = val_trials_mask.to(device)
    
    coh_trials_v = np.zeros([batch_size_v])
    bin_trials_v = np.zeros([batch_size_v],dtype=int)
    trial_starts = np.zeros([batch_size_v],dtype= np.long)
    trial_ends   = np.zeros([batch_size_v],dtype= np.long)
    
    #binns  = np.arange(-40,45,5)
    
    batch_counter = 0
    
    for i in range(0,nbinns):
        for j in range(0,psych_itr):
        #    a = np.random.rand()
        #    if a < (catch_freq):
        #        inp_streams,out_streams,mask = trial_maker.mk_catch_trial()
        #        is_catch_v[i] = 1
        #        coh_trials_v[i] = np.nan
        #    else:
        #    b = -100 + 200*np.random.rand()
        #    b = np.random.randint(np.size(binns))
        #        if b >= 0:
        #            b = 100
        #        else:
        #            b = -100
            #inp_streams,out_streams,mask,e = trial_maker.mk_coh_trial(binns[i])
            inp_streams,out_streams,mask,istart,iend  = trial_maker.mk_RT_trial(dur,binns[i])
            # trial_ends_v[batch_counter] = e
            
            trial_starts[batch_counter] = istart
            trial_ends[batch_counter]   = iend
            
        #    print(e)
        #    print(trial_ends_v[i])
            coh_trials_v[batch_counter] = binns[i]
            bin_trials_v[batch_counter] = i
            msk = torch.stack([mask,mask])
            val_trials_inp[batch_counter,:,:] = torch.transpose(inp_streams,0,1)
            val_trials_out[batch_counter,:,:] = torch.transpose(out_streams,0,1)
            val_trials_mask[batch_counter,:,:] = torch.transpose(msk,0,1)
            batch_counter += 1
    return val_trials_inp, val_trials_out, val_trials_mask, coh_trials_v, bin_trials_v, trial_starts, trial_ends 

def check_performance(binns,psych_itr,batch_size,out_pred,decision_sep,low_sep,trial_starts,trial_ends,bin_trials_v,trial_opt):
    
    resp0   = np.empty([len(binns),psych_itr])
    resp1   = np.empty([len(binns),psych_itr])
    ntrial = np.zeros([len(binns)])
    rts    = np.empty([len(binns),psych_itr])
    
    resp0[:] = np.nan
    resp1[:] = np.nan
    rts[:] = np.nan
    
    b_itr  = np.zeros([len(binns)])
    choice_vect = np.empty(out_pred.size()[0])
    choice_vect[:] = np.nan
    
    for i in range(0,batch_size):
        if trial_opt == "strict":
            persistant_choice = torch.sum(torch.abs(out_pred[i,(trial_ends[i]+1):,0] - out_pred[i,(trial_ends[i]+1):,1]) > decision_sep)
            low_b4 = torch.sum(torch.abs(out_pred[i,0:trial_starts[i],0] - out_pred[i,0:trial_starts[i],1]) < low_sep)
            if (persistant_choice >= 0.5*(out_pred[i,(trial_ends[i]+1):,0].size()[0]))\
                & (low_b4 >= 0.75*(out_pred[i,0:trial_starts[i],0].size()[0])): # a valid choice has been made
                a = torch.nonzero(torch.abs(out_pred[i,:,0] - out_pred[i,:,1]) > decision_sep).cpu().detach().numpy()
                b = np.diff(a)
                c = np.where(b>1)
                if c[0].size == 0:
                    rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0] - trial_starts[i])
                else:
                    rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[c[0][-1]] - trial_starts[i])
                
                ntrial[bin_trials_v[i]] += 1
                if out_pred[i,trial_ends[i],0] > out_pred[i,trial_ends[i],1]:
                    resp0[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                    choice_vect[i] = 0
                else:
                    resp1[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                    choice_vect[i] = 1
            b_itr[bin_trials_v[i]] +=1
        elif trial_opt == "loose":
            if out_pred[i,-1,0] > out_pred[i,-1,1]:
                ntrial[bin_trials_v[i]] += 1
                resp0[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                choice_vect[i] = 0
                # a = torch.nonzero(out_pred[i,:,0] - out_pred[i,:,1] > 0.0).cpu().detach().numpy()
                # b = np.diff(a)
                # c = np.where(b>1)
                # if c[0].size == 0:
                #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0] - trial_starts[i])
                # else:
                #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[c[0][-1]] - trial_starts[i])
                x = (out_pred[i,:,0]-out_pred[i,:,1]).cpu().detach().numpy()
                y = np.sign(x)
                z = np.diff(y)
                a = np.where(z)
                if a[0].__len__() !=0:
                        rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0][-1] - trial_starts[i])
                else:
                    rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (0 - trial_starts[i])
            elif out_pred[i,-1,1] > out_pred[i,-1,0]:
                ntrial[bin_trials_v[i]] += 1
                resp1[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                choice_vect[i] = 1
                # a = torch.nonzero(out_pred[i,:,1] - out_pred[i,:,0] > 0.0).cpu().detach().numpy()
                # b = np.diff(a)
                # c = np.where(b>1)
                # if c[0].size == 0:
                #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0] - trial_starts[i])
                # else:
                #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[c[0][-1]] - trial_starts[i])
                x = (out_pred[i,:,0]-out_pred[i,:,1]).cpu().detach().numpy()
                y = np.sign(x)
                z = np.diff(y)
                a = np.where(z)
                if a[0].__len__() !=0:
                        rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0][-1] - trial_starts[i])
                else:
                    rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (0 - trial_starts[i])
            b_itr[bin_trials_v[i]] +=1
            
        elif trial_opt == "intermediate":
            
            dz = (out_pred[i,:,0]-out_pred[i,:,1]).cpu().detach().numpy()
            sdz = np.sign(dz)
            dsdz = np.abs(np.diff(sdz))
            decision = np.sum(dsdz[trial_ends[i]-2:])
            
            if ((decision == 0) & (np.sum(np.abs(dz))>0.0)):   
                if out_pred[i,-1,0] > out_pred[i,-1,1]:
                    ntrial[bin_trials_v[i]] += 1
                    resp0[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                    choice_vect[i] = 0
                    # a = torch.nonzero(out_pred[i,:,0] - out_pred[i,:,1] > 0.0).cpu().detach().numpy()
                    # b = np.diff(a)
                    # c = np.where(b>1)
                    # if c[0].size == 0:
                    #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0] - trial_starts[i])
                    # else:
                    #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[c[0][-1]] - trial_starts[i])
                    x = (out_pred[i,:,0]-out_pred[i,:,1]).cpu().detach().numpy()
                    y = np.sign(x)
                    z = np.diff(y)
                    a = np.where(z)
                    if a[0].__len__() !=0:
                        rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0][-1] - trial_starts[i])
                    else:
                        rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (0 - trial_starts[i])
                elif out_pred[i,-1,1] > out_pred[i,-1,0]:
                    ntrial[bin_trials_v[i]] += 1
                    resp1[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                    choice_vect[i] = 1
                    # a = torch.nonzero(out_pred[i,:,1] - out_pred[i,:,0] > 0.0).cpu().detach().numpy()
                    # b = np.diff(a)
                    # c = np.where(b>1)
                    # if c[0].size == 0:
                    #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0] - trial_starts[i])
                    # else:
                    #     rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[c[0][-1]] - trial_starts[i])
                    x = (out_pred[i,:,0]-out_pred[i,:,1]).cpu().detach().numpy()
                    y = np.sign(x)
                    z = np.diff(y)
                    a = np.where(z)
                    if a[0].__len__() !=0:
                        rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (a[0][-1] - trial_starts[i])
                    else:
                        rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (0 - trial_starts[i])
                b_itr[bin_trials_v[i]] +=1
        elif trial_opt == "reaction_time":
            dz = (out_pred[i,:,0]-out_pred[i,:,1]).cpu().detach().numpy()
            adz = np.abs(dz)
            xx = np.where(adz>decision_sep)
            if xx[0].size == 0:
                # no choice
                choice_vect[i] = np.nan
            elif xx[0][0] < trial_starts[i]:
                # choice before stimulus
                choice_vect[i] = -1
            elif dz[xx[0][0]] > 0.0:
                # choose0
                ntrial[bin_trials_v[i]] += 1
                resp0[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                choice_vect[i] = 0
                rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (xx[0][0] - trial_starts[i])
            elif dz[xx[0][0]] < 0.0:
                # choose1
                ntrial[bin_trials_v[i]] += 1
                resp1[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = 1
                choice_vect[i] = 1
                rts[bin_trials_v[i],int(b_itr[bin_trials_v[i]])] = (xx[0][0] - trial_starts[i])
            b_itr[bin_trials_v[i]] +=1
            # sdz = np.sign(dz)
            # dsdz = np.abs(np.diff(sdz))
            # decision = np.sum(dsdz[trial_ends[i]-2:])

    psycho = np.nansum(resp0,axis=1)/ntrial
    chrono = np.nanmean(rts,axis=1)        
    perf0 = np.nansum(resp1[binns<0,:])/np.sum(ntrial[binns<0])
    perf1 = np.nansum(resp0[binns>0,:])/np.sum(ntrial[binns>0])
    
    perf = (np.nansum(resp1[binns<0,:]) + np.nansum(resp0[binns>0,:]))/np.sum(ntrial)
    
    perf_easy  = (np.nansum(resp1[0,:]) + np.nansum(resp0[-1,:]))/(2*resp1.shape[1])
    perf_total = (np.nansum(resp1[binns<0,:]) + np.nansum(resp0[binns>0,:]))/(resp1.shape[0]*resp1.shape[1])
    
    return ntrial, psycho, chrono, perf0, perf1, perf,rts,resp0,resp1, perf_easy, perf_total,choice_vect 

def measure_selectivity(nboot,choice_vec,act_e,act_i):
    n_boot = 50
    he = act_e.size()[-1]
    hi = act_i.size()[-1]
    aucs = dict()
    auc_boot = dict()
    choice_sel = dict()
    zscore = dict()
    
    aucs[0]       = np.empty([he])
    auc_boot[0]   = np.empty([he,n_boot])
    choice_sel[0] = np.empty([he])
    zscore[0]     = np.empty([he])
    zscore[0][:]  = np.nan
    
    aucs[1]       = np.empty([hi])
    auc_boot[1]   = np.empty([hi,n_boot])
    choice_sel[1] = np.empty([hi])
    zscore[1]     = np.empty([hi])
    zscore[1][:]  = np.nan
    
    a = np.where(~np.isnan(choice_vec))
    b = np.sum(choice_vec==0)
    c = np.sum(choice_vec==1)
    if ((a[0].__len__() != 0) & (b > 2) & (c > 2)):
        for i in range(he):
            x = act_e.cpu()[:,-1,i].detach().numpy()
            tpr,fpr,xx = roc_curve(choice_vec[a[0]],x[a[0]])
            aucs[0][i] = auc(fpr,tpr)
            for j in range(n_boot):
                shuffle = np.random.permutation(a[0].__len__())
                tpr,fpr,xx = roc_curve(choice_vec[a[0][shuffle]],x[a[0]])
                auc_boot[0][i,j] = auc(fpr,tpr)
            
            if np.std(auc_boot[0][i,:]):    
                zscore[0][i] = (aucs[0][i] - np.mean(auc_boot[0][i,:]))/np.std(auc_boot[0][i,:])
            
            if (zscore[0][i] > 1.96): 
                choice_sel[0][i] = 1
            elif (zscore[0][i] < -1.96): 
                choice_sel[0][i] = -1
            else:
                choice_sel[0][i] = 0
                
        for i in range(hi):
            x = act_i[:,-1,i].cpu().detach().numpy()
            tpr,fpr,xx = roc_curve(choice_vec[a[0]],x[a[0]])
            aucs[1][i] = auc(fpr,tpr)
            for j in range(n_boot):
                shuffle = np.random.permutation(a[0].__len__())
                tpr,fpr,xx = roc_curve(choice_vec[a[0][shuffle]],x[a[0]])
                auc_boot[1][i,j] = auc(fpr,tpr)
            
            if np.std(auc_boot[1][i,:]):    
                zscore[1][i] = (aucs[1][i] - np.mean(auc_boot[1][i,:]))/np.std(auc_boot[1][i,:])
            
            if (zscore[1][i] > 1.96): 
                choice_sel[1][i] = 1
            elif (zscore[1][i] < -1.96): 
                choice_sel[1][i] = -1
            else:
                choice_sel[1][i] = 0
        
    else:
        aucs[0][:]       = np.nan
        auc_boot[0] [:]  = np.nan
        choice_sel[0][:] = np.nan
        zscore[0][:]     = np.nan
        
        aucs[1][:]       = np.nan
        auc_boot[1][:]   = np.nan
        choice_sel[1][:] = np.nan
        zscore[1][:]     = np.nan
        
    return aucs,auc_boot,choice_sel,zscore