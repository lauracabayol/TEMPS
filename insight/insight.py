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

class Insight_module():
    """ Define class"""
    
    def __init__(self, model, batch_size):
        self.model=model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size=batch_size
        
    def _get_dataloaders(self, input_data, target_data, val_fraction=0.1):
        input_data = torch.Tensor(input_data)
        target_data = torch.Tensor(target_data)

        dataset = TensorDataset(input_data, target_data)

        trainig_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-val_fraction)), int(len(dataset)*val_fraction)+1])
        loader_train = DataLoader(trainig_dataset, batch_size=self.batch_size, shuffle = True)
        loader_val = DataLoader(val_dataset, batch_size=64, shuffle = True)

        return loader_train, loader_val

                


    def _loss_function(self,mean, std, logmix, true):
                
        logerf = torch.log(erf(true.cpu()[:,None]/(np.sqrt(2)*std.detach().cpu())+1))
        
        log_prob =   logmix - 0.5*(mean - true[:,None]).pow(2) / std.pow(2) - torch.log(std) #- logerf.to(self.device)
        log_prob = torch.logsumexp(log_prob, 1)
        loss = -log_prob.mean()

        return loss     

    def _to_numpy(self,x):
        return x.detach().cpu().numpy()
        
    def train(self,input_data, target_data,  nepochs=10, step_size = 100, val_fraction=0.1, lr=1e-3 ):
        self.model = self.model.train()
        loader_train, loader_val = self._get_dataloaders(input_data, target_data, val_fraction=0.1)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma =0.1)
    
        
        self.model = self.model.to(self.device)
        
        self.loss_train, self.loss_validation = [],[]
        
        

        for epoch in range(nepochs):
            for input_data, target_data in loader_train:
                _loss_train, _loss_validation = [],[]

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)


                optimizer.zero_grad()

                mu, logsig, logmix_coeff = self.model(input_data)
                logsig = torch.clamp(logsig,-6,2)
                sig = torch.exp(logsig)


                #print(mu,sig,target_data,torch.exp(logmix_coeff))

                loss = self._loss_function(mu, sig, logmix_coeff, target_data)
                _loss_train.append(loss.item())
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()   
                                
            self.loss_train.append(np.mean(_loss_train))

            for input_data, target_data in loader_val:


                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)


                mu, logsig, logmix_coeff = self.model(input_data)
                logsig = torch.clamp(logsig,-6,2)
                sig = torch.exp(logsig)

                loss_val = self._loss_function(mu, sig, logmix_coeff, target_data)
                _loss_validation.append(loss_val.item())

            self.loss_validation.append(np.mean(_loss_validation))

            #print(f'training_loss:{loss}',f'testing_loss:{loss_val}')
           
            
    def get_photoz(self,input_data, target_data):
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        input_data = input_data.to(self.device)
        target_data = target_data.to(self.device)

        for ii in range(len(input_data)):

            mu, logsig, logmix_coeff = self.model(input_data)
            logsig = torch.clamp(logsig,-6,2)
            sig = torch.exp(logsig)

            mix_coeff = torch.exp(logmix_coeff)

            z = (mix_coeff * mu).sum(1)
            zerr = torch.sqrt( (mix_coeff * sig**2).sum(1) + (mix_coeff * (mu - target_data[:,None])**2).sum(1))

              
        return self._to_numpy(z),self._to_numpy(zerr)


        #return model

    def plot_photoz(self, df, nbins,xvariable,metric, type_bin='bin'):
        bin_edges = stats.mstats.mquantiles(df[xvariable].values, np.linspace(0.1,1,nbins))
        ydata,xlab = [],[]


        for k in range(len(bin_edges)-1):
            edge_min = bin_edges[k]
            edge_max = bin_edges[k+1]

            mean_mag =  (edge_max + edge_min) / 2

            if type_bin=='bin':
                df_plot = df_test[(df_test.imag > edge_min) & (df_test.imag < edge_max)]
            elif type_bin=='cum':
                df_plot = df_test[(df_test.imag < edge_max)]
            else:
                raise ValueError("Only type_bin=='bin' for binned and 'cum' for cumulative are supported")


            xlab.append(mean_mag)
            if metric=='sig68':
                ydata.append(sigma68(df_plot.zwerr))
            elif metric=='bias':
                ydata.append(np.mean(df_plot.zwerr))
            elif metric=='nmad':
                ydata.append(nmad(df_plot.zwerr))
            elif metric=='outliers':
                ydata.append(len(df_plot[np.abs(df_plot.zwerr)>0.15])/len(df_plot))

        plt.plot(xlab,ydata, ls = '-', marker = '.', color = 'navy',lw = 1, label = '')
        plt.ylabel(f'{metric}$[\Delta z]$', fontsize = 18)
        plt.xlabel(f'{xvariable}', fontsize = 16)

        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)

        plt.grid(False)

        plt.show()
   
        
        
    