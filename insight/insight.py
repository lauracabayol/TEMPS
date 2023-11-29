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

class Insight_module():
    """ Define class"""
    
    def __init__(self, model, batch_size=100,rejection_param=1):
        self.model=model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size=batch_size
        self.rejection_parameter=rejection_param
        
            
            
    def _get_dataloaders(self, input_data, target_data, additional_data=None, val_fraction=0.1):
        input_data = torch.Tensor(input_data)
        target_data = torch.Tensor(target_data)
        
        if additional_data is None:
            dataset = TensorDataset(input_data, target_data)
        else:
            additional_data = torch.Tensor(additional_data)
            dataset = TensorDataset(input_data, target_data,additional_data)
                    

        trainig_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*(1-val_fraction)), int(len(dataset)*val_fraction)+1])
        loader_train = DataLoader(trainig_dataset, batch_size=self.batch_size, shuffle = True)
        loader_val = DataLoader(val_dataset, batch_size=64, shuffle = True)

        return loader_train, loader_val

                


    def _loss_function(self,mean, std, logmix, true):
                        
        log_prob =   logmix - 0.5*(mean - true[:,None]).pow(2) / std.pow(2) - torch.log(std) 
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

            print(f'training_loss:{loss}',f'testing_loss:{loss_val}')
           
        

        
    def get_pz(self,input_data, target_data, return_pz=False):
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        input_data = input_data.to(self.device)
        target_data = target_data.to(self.device)
                

        
        mu, logsig, logmix_coeff = self.model(input_data)
        logsig = torch.clamp(logsig,-6,2)
        sig = torch.exp(logsig)

        mix_coeff = torch.exp(logmix_coeff)

        z = (mix_coeff * mu).sum(1)
        zerr = torch.sqrt( (mix_coeff * sig**2).sum(1) + (mix_coeff * (mu - target_data[:,None])**2).sum(1))

        mu,  mix_coeff, sig = mu.detach().cpu().numpy(),  mix_coeff.detach().cpu().numpy(), sig.detach().cpu().numpy()
        
        
        if return_pz==True:
            x = np.linspace(0, 4, 1000)
            pdf_mixture = np.zeros(shape=(len(target_data), len(x)))
            for ii in range(len(input_data)):
                for i in range(6):
                    pdf_mixture[ii] += mix_coeff[ii,i] * norm.pdf(x, mu[ii,i], sig[ii,i])

            return self._to_numpy(z),self._to_numpy(zerr), pdf_mixture
    
        else:
            return self._to_numpy(z),self._to_numpy(zerr)
        
    def pit(self, input_data, target_data):
        
        pit_list = []
        
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        input_data = input_data.to(self.device)
                

        mu, logsig, logmix_coeff = self.model(input_data)
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

        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        input_data = input_data.to(self.device)


        mu, logsig, logmix_coeff = self.model(input_data)
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
   
        
    def plot_pz(self, m, pz, specz):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the PDF with a label
        ax.plot(np.linspace(0, 4, 1000), pz[m], label='PDF', color='navy')

        # Add a vertical line for 'specz_test'
        ax.axvline(specz[m], color='black', linestyle='--', label=r'$z_{\rm s}$')

        # Add labels and a legend
        ax.set_xlabel(r'$z$', fontsize = 18)
        ax.set_ylabel('Probability Density', fontsize=16)
        ax.legend(fontsize = 18)

        # Display the plot
        plt.show()
    