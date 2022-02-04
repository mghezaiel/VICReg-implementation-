import torch


class Encoder(torch.nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__()
        self.__dict__.update(kwargs)
        self.linear = torch.nn.Linear(in_features = self.in_features, out_features=self.out_features)
        
    def forward(self,x): 
        x = self.linear(x)
        return x.relu()
      
class Projector(torch.nn.Module): 
    def __init__(self,**kwargs):
        super(Projector,self).__init__()
        self.__dict__.update(kwargs) 
        self.linear = torch.nn.Linear(in_features = self.in_features, out_features=self.out_features)
        
    def forward(self,x): 
        x = self.linear(x)
        return x.relu()
    
    
class EncoderProjectorModel(torch.nn.Module): 
    def __init__(self,**kwargs): 
        super(EncoderProjectorModel,self).__init__()
        self.__dict__.update(kwargs)
        self.encoder1 = Encoder(in_features = self.in_encoder_features, out_features = self.out_encoder_features)
        self.encoder2 = Encoder(in_features = self.in_encoder_features, out_features = self.out_encoder_features)
        self.projector1 = Projector(in_features = self.in_projector_features, out_features = self.out_projector_features)
        self.projector2 = Projector(in_features = self.in_projector_features, out_features = self.out_projector_features)
        
    def forward(self,x): 
        encoded_1 = self.encoder1(x)
        encoded_2 = self.encoder2(x)
        projected_1 = self.projector1(encoded_1)
        projected_2 = self.projector2(encoded_2)
        return encoded_1,encoded_2,projected_1,projected_2