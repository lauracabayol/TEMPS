import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from astropy.io import fits
import os
from astropy.table import Table
from scipy.spatial import KDTree
from scipy.special import erf
from scipy.stats import norm
import sys

sys.path.append('/.')
from utils import maximum_mean_discrepancy, compute_kernel

class Temps_module():
    """ Define class"""
    
    def __init__(self, modelF, modelZ, batch_size=100,rejection_param=1, da=True, verbose=False):
        self.modelZ=modelZ
        self.modelF=modelF
        self.da=da
        self.verbose=verbose
        self.ngaussians=modelZ.ngaussians

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size=batch_size
        self.rejection_parameter=rejection_param
        
            
            
    def _get_dataloaders(self, input_data, target_data, input_data_DA, target_data_DA, val_fraction=0.1):
        input_data = torch.Tensor(input_data)
        target_data = torch.Tensor(target_data)
        if input_data_DA is not None:
            input_data_DA = torch.Tensor(input_data_DA)
            target_data_DA = torch.Tensor(target_data_DA)
        else:
            input_data_DA = input_data.clone()
            target_data_DA = target_data.clone()
            
        dataset = TensorDataset(input_data, input_data_DA, target_data, target_data_DA)
        trainig_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-val_fraction)), int(len(dataset)*val_fraction)+1])
        loader_train = DataLoader(trainig_dataset, batch_size=self.batch_size, shuffle = True)
        loader_val = DataLoader(val_dataset, batch_size=64, shuffle = True)

        return loader_train, loader_val

                


    def _loss_function(self,mean, std, logmix, true):
                        
        log_prob =   logmix - 0.5*(mean - true[:,None]).pow(2) / std.pow(2) - torch.log(std) 
        log_prob = torch.logsumexp(log_prob, 1)
        loss = -log_prob.mean()
            
        return loss  
    
    def _loss_function_DA(self,f1, f2):
        kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)
        loss = kl_loss(f1, f2)
        loss = torch.log(loss)
        #print('f1',f1)
        #print('f2',f2)
        
        return loss   

    def _to_numpy(self,x):
        return x.detach().cpu().numpy()
    
       
        
    def train(self,input_data, 
              input_data_DA, 
              target_data, 
              target_data_DA, 
              nepochs=10, 
              step_size = 100,
              val_fraction=0.1, 
              lr=1e-3,
             weight_decay=0):
        self.modelZ = self.modelZ.train()
        self.modelF = self.modelF.train()

        loader_train, loader_val = self._get_dataloaders(input_data, target_data, input_data_DA, target_data_DA, val_fraction=0.1)
        optimizerZ = optim.Adam(self.modelZ.parameters(), lr=lr, weight_decay=weight_decay)
        optimizerF = optim.Adam(self.modelF.parameters(), lr=lr, weight_decay=weight_decay)

        schedulerZ = torch.optim.lr_scheduler.StepLR(optimizerZ, step_size=step_size, gamma =0.1)
        schedulerF = torch.optim.lr_scheduler.StepLR(optimizerF, step_size=step_size, gamma =0.1)    
        
        self.modelZ = self.modelZ.to(self.device)
        self.modelF = self.modelF.to(self.device)

        self.loss_train, self.loss_validation = [],[]
        
        for epoch in range(nepochs):
            for input_data, input_data_da, target_data, target_data_DA  in loader_train:
                _loss_train, _loss_validation = [],[]

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                if self.da:
                    input_data_da = input_data_da.to(self.device)
                    target_data_DA = target_data_DA.to(self.device)

                optimizerF.zero_grad()
                optimizerZ.zero_grad()

                features = self.modelF(input_data)
                if self.da:
                    features_DA = self.modelF(input_data_da)

                mu, logsig, logmix_coeff = self.modelZ(features)
                logsig = torch.clamp(logsig,-6,2)
                sig = torch.exp(logsig)

                lossZ = self._loss_function(mu, sig, logmix_coeff, target_data)
                
                #mu, logsig, logmix_coeff = self.modelZ(features_DA)
                #logsig = torch.clamp(logsig,-6,2)
                #sig = torch.exp(logsig)   
                
                #lossZ_DA = self._loss_function(mu, sig, logmix_coeff, target_data_DA)

                if self.da:
                    lossDA = maximum_mean_discrepancy(features, features_DA, kernel_type='rbf')
                    lossDA = lossDA.sum()
                    loss = lossZ +1e3*lossDA
                else:
                    loss = lossZ
                                
                _loss_train.append(lossZ.item())
                                
                loss.backward()
                optimizerF.step()
                optimizerZ.step()
            
            schedulerF.step()   
            schedulerZ.step()   
                                
            self.loss_train.append(np.mean(_loss_train))

            for input_data, _, target_data, _ in loader_val:

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)


                features = self.modelF(input_data)
                mu, logsig, logmix_coeff = self.modelZ(features)   
                
                logsig = torch.clamp(logsig,-6,2)
                sig = torch.exp(logsig)

                loss_val = self._loss_function(mu, sig, logmix_coeff, target_data)
                _loss_validation.append(loss_val.item())

            self.loss_validation.append(np.mean(_loss_validation))
            
            
            if self.verbose:

                print(f'training_loss:{loss}',f'testing_loss:{loss_val}')
           

    def get_features(self, input_data):
        self.modelF = self.modelF.eval()
        self.modelF = self.modelF.to(self.device)
        
        input_data = input_data.to(self.device)
        
        features = self.modelF(input_data)
        
        return features.detach().cpu().numpy()
        

    def get_pz(self,input_data, return_pz=True, return_flag=True, retrun_odds=False):
        self.modelZ = self.modelZ.eval()
        self.modelZ = self.modelZ.to(self.device)
        self.modelF = self.modelF.eval()
        self.modelF = self.modelF.to(self.device)

        input_data = input_data.to(self.device)
                

        features = self.modelF(input_data)
        mu, logsig, logmix_coeff = self.modelZ(features)
        logsig = torch.clamp(logsig,-6,2)
        sig = torch.exp(logsig)

        mix_coeff = torch.exp(logmix_coeff)

        z = (mix_coeff * mu).sum(1)
        zerr = torch.sqrt( (mix_coeff * sig**2).sum(1) + (mix_coeff * (mu - mu.mean(1)[:,None])**2).sum(1))

        mu,  mix_coeff, sig = mu.detach().cpu().numpy(),  mix_coeff.detach().cpu().numpy(), sig.detach().cpu().numpy()
        
        
        if return_pz==True:
            zgrid = np.linspace(0, 5, 1000)
            pdf_mixture = np.zeros(shape=(len(input_data), len(zgrid)))
            for ii in range(len(input_data)):
                for i in range(self.ngaussians):
                    pdf_mixture[ii] += mix_coeff[ii,i] * norm.pdf(zgrid, mu[ii,i], sig[ii,i])
            if return_flag==True:
                #narrow peak
                pdf_mixture = pdf_mixture / pdf_mixture.sum(1)[:,None]
                diff_matrix = np.abs(self._to_numpy(z)[:,None] - zgrid[None,:])
                #odds
                idx_peak = np.argmax(pdf_mixture,1)
                zpeak = zgrid[idx_peak]
                diff_matrix_upper = np.abs((zpeak+0.05)[:,None] - zgrid[None,:])
                diff_matrix_lower = np.abs((zpeak-0.05)[:,None] - zgrid[None,:])
                
                idx = np.argmin(diff_matrix,1)
                idx_upper = np.argmin(diff_matrix_upper,1)
                idx_lower = np.argmin(diff_matrix_lower,1)
                
                p_z_x = np.zeros(shape=(len(z)))
                odds = np.zeros(shape=(len(z)))

                for ii in range(len(z)):
                    p_z_x[ii] = pdf_mixture[ii,idx[ii]]
                    odds[ii] = pdf_mixture[ii,:idx_upper[ii]].sum() - pdf_mixture[ii,:idx_lower[ii]].sum()
                    

                                        
                return self._to_numpy(z),self._to_numpy(zerr), pdf_mixture, p_z_x, odds
            else:

                return self._to_numpy(z),self._to_numpy(zerr), pdf_mixture
    
        else:
            return self._to_numpy(z),self._to_numpy(zerr)
        
    def pit(self, input_data, target_data):
        
        pit_list = []

        self.modelF = self.modelF.eval()
        self.modelF = self.modelF.to(self.device)
        self.modelZ = self.modelZ.eval()
        self.modelZ = self.modelZ.to(self.device)

        input_data = input_data.to(self.device)
                

        features = self.modelF(input_data)
        mu, logsig, logmix_coeff = self.modelZ(features)
        
        logsig = torch.clamp(logsig,-6,2)
        sig = torch.exp(logsig)

        mix_coeff = torch.exp(logmix_coeff)
        
        mu,  mix_coeff, sig = mu.detach().cpu().numpy(),  mix_coeff.detach().cpu().numpy(), sig.detach().cpu().numpy() 
        
        for ii in range(len(input_data)):
            pit = (mix_coeff[ii] * norm.cdf(target_data[ii]*np.ones(mu[ii].shape),mu[ii], sig[ii])).sum()
            pit_list.append(pit)
        
        
        return pit_list
    
    def crps(self, input_data, target_data):

        def measure_crps(cdf, t):
            zgrid = np.linspace(0,4,1000)
            Deltaz = zgrid[None,:] - t[:,None]
            DeltaZ_heaviside = np.where(Deltaz < 0,0,1)
            integral = (cdf-DeltaZ_heaviside)**2
            crps_value = integral.sum(1) / 1000

            return crps_value


        crps_list = []

        self.modelF = self.modelF.eval()
        self.modelF = self.modelF.to(self.device)
        self.modelZ = self.modelZ.eval()
        self.modelZ = self.modelZ.to(self.device)

        input_data = input_data.to(self.device)


        features = self.modelF(input_data)
        mu, logsig, logmix_coeff = self.modelZ(features)
        logsig = torch.clamp(logsig,-6,2)
        sig = torch.exp(logsig)

        mix_coeff = torch.exp(logmix_coeff)


        mu,  mix_coeff, sig = mu.detach().cpu().numpy(),  mix_coeff.detach().cpu().numpy(), sig.detach().cpu().numpy() 

        z = (mix_coeff * mu).sum(1)

        x = np.linspace(0, 4, 1000)
        pdf_mixture = np.zeros(shape=(len(target_data), len(x)))
        for ii in range(len(input_data)):
            for i in range(6):
                pdf_mixture[ii] += mix_coeff[ii,i] * norm.pdf(x, mu[ii,i], sig[ii,i])

        pdf_mixture = pdf_mixture / pdf_mixture.sum(1)[:,None]


        cdf_mixture = np.cumsum(pdf_mixture,1)

        crps_value = measure_crps(cdf_mixture, target_data)



        return crps_value


        
