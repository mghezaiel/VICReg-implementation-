from optuna import Trial
import optuna
from torch import optim
import numpy as np 
import time
import torch 
from loss import *
        

    
class Training(): 
    def __init__(self,**kwargs): 
        self.__dict__.update(kwargs)
         
    def metrics(self): 
        pass
    
    def sample_batch(self,batch_size): 
        indices = np.random.choice(np.arange(self.data.shape[0]),batch_size, replace = False)
        return self.data[indices,:]

    def training(self,trial):
        
        # Fixed parameters
        in_encoder_features = self.config["Encoder"]["in_features"]
        out_encoder_features = self.config["Encoder"]["out_features"]
        in_projector_features = self.config["Projector"]["in_features"]
        gamma = self.config["Model"]["loss"]["gamma"]
        epsilon = self.config["Model"]["loss"]["epsilon"] 
        nu = self.config["Model"]["loss"]["nu"] 
        epochs = self.config["Model"]["epochs"]
        
        # To optimize
        if not hasattr(self,'study'): 
            out_projector_features = trial.suggest_int("projector_out_features",*self.config["Projector"]["out_features"])
            lam = trial.suggest_int("lam",*self.config["Model"]["loss"]["lambda"])
            mu = trial.suggest_int("mu",*self.config["Model"]["loss"]["mu"])
            lr = trial.suggest_uniform("lr",*self.config["Optimizer"]["lr"])
            beta1 = trial.suggest_uniform("beta1",*self.config["Optimizer"]["beta1"])
            beta2 = trial.suggest_uniform("beta2",*self.config["Optimizer"]["beta2"])
            batch_size = trial.suggest_int("batch_size",*self.config["Model"]["batch_size"])
 
        else:
            assert not trial
            out_projector_features = self.study.best_params["projector_out_features"]
            lam = self.study.best_params["lam"]
            mu = self.study.best_params["mu"]
            lr = self.study.best_params["lr"]
            beta1 = self.study.best_params["beta1"]
            beta2 = self.study.best_params["beta2"]
            batch_size = self.study.best_params["batch_size"]
            
        # Instanciate the model
        model = self.model(in_encoder_features = in_encoder_features, 
                          out_encoder_features = out_encoder_features, 
                          in_projector_features = in_projector_features, 
                          out_projector_features = out_projector_features)
        
        if not trial: 
            print("N params = ",sum([p.numel() for p in model.parameters()]))
    
        # Define the loss
        criterion = VICLoss(gamma = gamma, epsilon = epsilon, lam = lam, mu = mu, nu = nu)
        
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, betas = (beta1,beta2))
        
        # Training loop
        losses = list()
        for epoch in range(epochs):
            
            batches = self.sample_batch(batch_size)
            optimizer.zero_grad()
            encoded_1,encoded_2,projected_1,projected_2 = model(batches)
            loss = criterion(projected_1,projected_2)
            loss.backward()
            optimizer.step()

            losses+=[loss]
                
            
        if trial:
            return min(losses)
        else: 
            self.trained_model = model
            return losses
            
class Optimize(): 
    def __init__(self,**kwargs): 
        self.__dict__.update(kwargs)
        
    def monitor_execution_time(func): 
        def wrapper(self,**kwargs):
            start = time.time()
            func(self,**kwargs)
            print("Elapsed time: ", round(time.time()-start,2), "seconds")
        return wrapper 
        
    @monitor_execution_time
    def optimize(self,data): 
        study = optuna.create_study(direction = "minimize")
        training = Training(config = self.config, model = self.model, data = data)
        study.optimize(training.training, n_trials=self.n_trials)
        training.study = study
        self.training = training 
    
    def training_on_best_params(self,data):
        # Train on data (could be new data)
        self.training.data = data
        self.losses = self.training.training(trial = False)
        self.trained_model = self.training.trained_model
        
    def predict(self,data): 
        return self.trained_model(data)

