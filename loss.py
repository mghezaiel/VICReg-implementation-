import numpy as np 
import torch 

class VICLoss(): 
    def __init__(self,**kwargs): 
        self.__dict__.update(kwargs) 
    
    def variance_loss(self,x): 
        def func(x,gamma,epsilon):
            return max([0,gamma - torch.sqrt(x.var()+epsilon)])
        return sum([func(x[:,dim],self.gamma,self.epsilon) for dim in range(x.shape[1])])
        
    def covariance_loss(self,x): 
        cov = torch.tensor(np.cov(x.detach().numpy()))
        ind = np.diag_indices(cov.shape[0])
        cov[ind[0],ind[1]] = torch.zeros(cov.shape[0]).double()
        return (cov**2).sum()/cov.shape[0]
        
    def distance_loss(self,x,y): 
        loss = torch.nn.MSELoss() 
        return loss(x,y)
    
    def __call__(self,x,y): 
        return self.lam*self.distance_loss(x,y)+self.mu*(self.variance_loss(x)+self.variance_loss(y)
                                                 )+self.nu*(self.covariance_loss(x)+self.covariance_loss(y))