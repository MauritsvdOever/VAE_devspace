# -*- coding: utf-8 -*-
"""
Own implementations of GAUSS VAE

Created on Thu Apr 14 11:29:10 2022

@author: gebruiker
"""
# imports
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class VAE(nn.Module):
    """
    Inherits from nn.Module to construct VAE based on given data and 
    desired dimensions. 
    
    """
    
    def __init__(self, X, dim_Z, layers=3, standardize = True, batch_wise=True, done=False, plot=False, dist='normal'):
        """
        Constructs attributes, such as the autoencoder structure itself

        Inputs for instantiating:
        -------------------------
        X           : multidimensional np array or pd dataframe
        
        dim_Z       : desired amount of dimensions in the latent space 
        
        layers      : int, amount of layers for the encoder and decoder, default = 3, must be >= 2
        
        standardize : bool, if true than X gets mean var standardized

        """
        # imports
        super(VAE, self).__init__()
        from collections import OrderedDict
        import numpy as np
        import torch
        from torch import nn
        import matplotlib.pyplot as plt
        
        # make X a tensor, and standardize based on standardize
        if standardize:
            self.X     = self.standardize_X(self.force_tensor(X)) # first force X to be a float tensor, and then standardize it
        else:
            self.X     = self.force_tensor(X)
        
        
        self.multivariate = True

        self.dim_X = X.shape[1]
        self.dim_Z = dim_Z
        self.dim_Y = int((self.dim_X + self.dim_Z) / 2)
        self.n     = X.shape[0]
        self.K     = X.shape[1]
        self.done_bool = done
        self.plot = plot
        self.dist = dist
        
        if dist == 't':
            self.nu = 5.0
        
        self.beta = 1.0 # setting beta to zero is equivalent to a normal autoencoder
        self.batch_wise = batch_wise
            
        # LeakyReLU for now
        self.encoder = self.construct_encoder(layers)
        self.decoder = self.construct_decoder(layers)
        
        
    def construct_encoder(self, layers):
        """
        Generates the encoder neural net dynamically based on layers parameter

        Parameters
        ----------
        layers : int, amount of layers, same as the decoder

        Returns
        -------
        instantiation of the nn.Sequential class, with the appropriate amount
        of layers

        """
        network = OrderedDict()
        network['0'] = nn.Linear(self.dim_X, self.dim_Y)
        network['1'] = nn.LeakyReLU() 
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.LeakyReLU()
            count += 2
        
        network[str(count)] = nn.Linear(self.dim_Y, self.dim_Z)
        
        return nn.Sequential(network)
    
        
    def construct_decoder(self, layers):
        """
        Generates the decoder neural net dynamically based on layers parameter

        Parameters
        ----------
        layers : int, amount of layers, same as the enoder

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        network = OrderedDict()
        network['0'] = nn.Linear(self.dim_Z, self.dim_Y)
        network['1'] = nn.LeakyReLU()
        
        count = 2
        for i in range(layers-2):
            network[str(count)]   = nn.Linear(self.dim_Y, self.dim_Y)
            network[str(count+1)] = nn.LeakyReLU()
            count += 2
        
        network[str(count)] = nn.Linear(self.dim_Y, self.dim_X)
        
        return nn.Sequential(network)
    
    def standardize_X(self, X):
        """
        Class method that stores the mean and variances of the given data for 
        later unstandardisation, and standardizes the data

        Parameters
        ----------
        X : multidimensional float tensor

        Returns
        -------
        Standardized version of multidimensional float tensor

        """
        # write code that stores mean and var, so u can unstandardize X_prime
        self.means_vars_X = (X.mean(axis=0), X.std(axis=0))
        
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def unstandardize_Xprime(self, X_prime):
        """
        Using previously stores means and variances, unstandardize the predicted
        data

        Parameters
        ----------
        X_prime : multidimensial float tensor

        Returns
        -------
        Rescaled multidimensional float tensor

        """
        return (X_prime * self.means_vars_X[1] + self.means_vars_X[0])
    
    def force_tensor(self, X):
        """
        forces the given object into a float tensor

        Parameters
        ----------
        X : np.array or pd.DataFrame of data

        Returns
        -------
        float tensor of given data

        """
        # write code that forces X to be a tensor
        if type(X) != torch.Tensor:
            return torch.Tensor(np.array(X)).float()
        else:
            return X.float() # force it to float anyway
    
    def forward(self, data):
        """
        Function that standardizes the given data, and feeds it through the 
        architecture

        Parameters
        ----------
        data : Multidimensional array of data, has to match the model 
        instantiation in terms of feature count

        Returns
        -------
        Data that has been fed through the model

        """
        if self.X.shape[1] != data.shape[1]:
            print('data does not match instantiation data in feature count')
            return None
        
        data = self.standardize_X(self.force_tensor(data))
        
        return self.unstandardize_Xprime(self.decoder(self.encoder(data))).detach().numpy()
        
    def MM(self, z):
        # MULTIVARIATE
        if self.multivariate:
            if self.dist == 'normal':
                std_target = 1.0
                kurt_target = 3.0
            elif self.dist == 't':
                std_target = 1.0 # / np.sqrt((self.nu-2)/self.nu)
                kurt_target = 6.0 /(self.nu-4)
            
            cov_z = torch.cov(z.T)
            
            # first moment, expected value of all variables
            mean_score = torch.linalg.norm(z.mean(dim=0), ord=2)
            
            # second moment
            std_score = torch.linalg.norm(cov_z - torch.eye(z.shape[1])*std_target, ord=2)
            
            # third and fourth moment
            diffs = z - z.mean(dim=0)
            zscores = diffs / diffs.std(dim=0)
            
            skews = torch.mean(torch.pow(zscores, 3.0), dim=0)
            kurts = torch.mean(torch.pow(zscores, 4.0), dim=0) - kurt_target
            
            skew_score = torch.linalg.norm(skews, ord=2) # works but subject to sample var
            kurt_score = torch.mean(kurts - kurt_target)
            
        else:
            #UNIVARIATE SEPARATE 
            if self.dist == 'normal':
                std_target  = torch.Tensor([1]*self.dim_Z)
                kurt_target = torch.Tensor([3]*self.dim_Z)
                
            elif self.dist == 't':
                std_target = torch.Tensor([(1 / np.sqrt((self.nu-2)/self.nu))]*self.dim_Z) # 
                kurt_target = torch.Tensor([6/(self.nu-4)]*self.dim_Z) 
            
            means = z.mean(dim=0)
            diffs = z - means
            std = z.std(dim=0)
            zscores = diffs / std
            skews = (torch.pow(zscores, 3.0)).mean(dim=0)
            kurts = torch.pow(zscores, 4.0).mean(dim=0)
            
            mean_score = (means**2).mean()
            std_score = ((std - std_target)**2).mean()
            skew_score = (skews**2).mean()
            kurt_score = ((kurts - kurt_target)**2).mean()
        
        #print(std_score)
        # return (1/22)*mean_score + (10/22)*std_score + (1/22)*skew_score + (10/22)*kurt_score        
        return std_score #mean_score + std_score + skew_score + kurt_score
    
    
    def RE_MM_metric(self, epoch):
        """
        Function that calculates the loss of the autoencoder by
        RE and MM. 

        Returns
        -------
        tuple of RE and MM

        """
        batch = 500
        epoch_scale_threshold = 0.99
        
        if self.X.shape[0] < 1000:
            self.batch_wise = False
        
        if epoch > self.epochs * epoch_scale_threshold:
            batch += int((self.X.shape[0]-batch) / 
                         (self.epochs - self.epochs*epoch_scale_threshold) * 
                         (epoch-self.epochs*epoch_scale_threshold))
        
        if self.batch_wise == True:
            X = self.X[torch.randperm(self.X.shape[0])[0:batch],:]
            self.n = X.shape[0]
        else:
            X = self.X
        
        z       = self.encoder(X)
        
        noise   = torch.normal(mean=0, std=0.01, size=z.shape)
        
        x_prime = self.decoder(z + noise) # kingma & welling reparameterization
        
        # get negative average log-likelihood here
        MM = self.MM(z)
        
        self.REs = (X - x_prime)**2
        
        RE = self.REs.mean() # mean squared error of reconstruction
        
        
        return (RE, MM)

    
    def loss_function(self, RE_MM):
        """
        function that reconciles RE and MM in loss equation

        Parameters
        ----------
        RE_MM : tuple of RE and MM

        Returns
        -------
        calculated loss as a product of RE and MM

        """
        return RE_MM[0] + self.beta * RE_MM[1]
        # return RE_MM[0]/ 2 * RE_MM[0]**2 + RE_MM[1]

    
    def fit(self, epochs):
        """
        Function that fits the model based on instantiated data
        """
        from tqdm import tqdm
        
        self.train() # turn into training mode
        REs  = []
        MMs  = []
        
        optimizer = torch.optim.Adagrad(self.parameters(),
                             lr = 0.01,
                             weight_decay = 0.001) # specify some hyperparams for the optimizer
        
        
        self.epochs = epochs
        
        REs = np.zeros(epochs)
        MMs = np.zeros(epochs)
        
        for epoch in tqdm(range(epochs)):
        # for epoch in range(epochs):
            RE_MM = self.RE_MM_metric(epoch) # store RE and KL in tuple
            loss = self.loss_function(RE_MM) # calculate loss function based on tuple
            
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            REs[epoch] = RE_MM[0].detach().numpy()
            MMs[epoch] = RE_MM[1].detach().numpy() # RE and KLs are stored for analysis
        if self.plot:
            plt.plot(range(epochs), REs)
            plt.title('Reconstruction errors')
            plt.show()
            plt.plot(range(epochs), MMs)
            plt.title('neg avg MMs')
            plt.show()
        self.eval() # turn back into performance mode
        if self.done_bool:
            self.done()
        
        return 
    
    def done(self):
        import win32api
        win32api.MessageBox(0, 'The model is done calibrating :)', 'Done!', 0x00001040)
        return
    
    def fit_garchs(self):
        """
        fit and store garchs, and vols. To be run after fitting the model

        Returns
        -------
        None.

        """
        
        from arch import arch_model
        
        
        scaling = 1
        z = self.encoder(self.X)
        
        self.garchs = []
        self.sigmas   = np.zeros((self.X.shape[0], self.dim_Z))
        
        for col in range(self.dim_Z):
            series = self.X[:,col].detach().numpy()
            
            garch = arch_model(series*scaling, mean='zero', vol='GARCH', dist=self.dist, rescale=True)
            garch = garch.fit(disp=False)
            # print(garch.params)
            # print(f'estimated nu = {garch.params.nu}')
            self.sigmas[:,col] = garch.conditional_volatility/scaling
            self.garchs += [garch]
        
        return 
    
    def insample_VaRs(self, quantile, plot = False, output = False):
        from scipy import stats
        
        sim_count = 1000
        
        self.fit_garchs()
        VaRs = np.zeros(shape=(len(self.sigmas)))
        avg_return = self.unstandardize_Xprime(self.X).detach().numpy().mean(axis=1)
        
        for row in range(len(VaRs)):
            sims = np.random.normal(loc=0, scale=1, size=(sim_count, self.dim_Z))
            sims = torch.Tensor(sims * self.sigmas[row,:])
            sims = self.unstandardize_Xprime(self.decoder(sims)).detach().numpy()
            port_return = sims.mean(axis=1)
            VaRs[row] = np.quantile(port_return, 0.05)
        
        if plot:
            plt.plot(avg_return)
            plt.plot(VaRs)
            plt.title("Value at Risks for X, q = " + str(quantile))
            plt.show()
        
        if output:
            exceedances = sum(VaRs > avg_return)
            binom_pval = stats.binomtest(exceedances, len(VaRs), p=quantile).pvalue
            print("")
            print("exceedance ratio = ", exceedances/len(VaRs))
            print("p-value          = ", binom_pval)
            print("")
            
        return VaRs
            
    def outofsample_VaRs(self, data, quantile, plot = False, output = False):
        """
        

        Parameters
        ----------
        data : numpy array, pd dataframe, or tensor array of data

        Returns
        -------
        None.

        """
        from arch import arch_model
        from scipy import stats
        import numpy as np
        import matplotlib.pyplot as plt
        
        sim_count = 1000
        
        self.fit_garchs()
        z = self.encoder(self.X)
        tensor_data = self.encoder(self.standardize_X(self.force_tensor(data)))
        avg_return = data.mean(axis=1)
        
        VaRs = np.zeros(shape=(len(tensor_data)))
        
        sigmas = np.zeros(shape=(len(tensor_data), self.dim_Z))
        sigmas_init = self.sigmas[-1,:]
        
        
        for col in range(self.dim_Z):
            print
            sigmas[0,col] = self.garchs[col].params[0] + self.garchs[col].params[1] * z[-1,col]**2 + self.garchs[col].params[2] * self.sigmas[-1,col]**2
            
            for row in range(1,len(sigmas)):
                sigmas[row,col] = self.garchs[col].params[0] + self.garchs[col].params[1] * tensor_data[row-1,col]**2 + self.garchs[col].params[2] * sigmas[row-1,col]
        
        for row in range(len(VaRs)):
            sims = np.random.normal(loc=0, scale=1, size=(sim_count, self.dim_Z))
            sims = torch.Tensor(sims * sigmas[row,:])
            sims = self.unstandardize_Xprime(self.decoder(sims)).detach().numpy()
            port_return = sims.mean(axis=1)
            VaRs[row] = np.quantile(port_return, 0.05)
          
        if plot:
            plt.plot(avg_return)
            plt.plot(VaRs)
            plt.title("Value at Risks for X, q = " + str(quantile))
            plt.show()
        
        if output:
            exceedances = sum(VaRs > avg_return)
            binom_pval = stats.binomtest(exceedances, len(VaRs), p=quantile).pvalue
            print("")
            print("exceedance ratio = ", exceedances/len(VaRs))
            print("p-value          = ", binom_pval)
            print("")
        
        return VaRs
        #return tensor_data.detach().numpy()