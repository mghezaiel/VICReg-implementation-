# VICReg-implementation-
Implementation of the Variance Invariance Covariance regularization approach applied to dimensionality reduction

This is an example of a dimensionality reduction task following the paper:
<br>

<i>Bardes, A., Ponce, J., & LeCun, Y. (2021).<i/><br>
    
<b>VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.</b> ArXiv, abs/2105.04906.
   
I adapted the work to non image data (nxm matrices) for unsupervised dimensionality reduction. 
    
All modules were implemented in the local repository and following what as been described in the paper. 

    
1) The library contains:

        - A simple encoder consisting in a single dense layer followed by relu activation. 
        - A projector with the same architecture that aims to project the data in a higher dimensional space.
        - The Variance - Invariance - Covariance loss applied to the project data.
        - An optuna based optimization setting to optimize: loss hyperparameters (mu and lambda, cf paper) and training hypeparameters. 
        - A class to evaluate the quality of embeddings by comparing silhouette coefficient based clustering separability between:
            a) Input data
            b) Predicted embeddings. 
    
![alt text](http://url/to/img.png) 

2) Short explanation of VICReg: 
The work was initially developped to produce embeddings from image data that are subject to data augmentation using affine transformation or cropping.
The idea is about trying to learn similar embeddings between different view of the data (ie: augmented view). 
Inspired by this work, this can be used for dimensionality reduction. The architecture consists in: 
    1) Two siamese networks (encoders) that produces embeddings (Y and Y')
    2) Two other networks (decoders or projectors) that projects the embeddings to a higher dimensional space (Z and Z')


Consistency of the learned embeddings (Y and Y') with the input data (X and X') through regularization of the projection Z and Z'
    
Regularization is performed by defining the VICReg loss: 
    - Variance loss: That maintains variance inside a batch (for Z and Z' independantly)
    - Invariance loss: That maintains similarity between embeddings Z and Z' (MSE) 
    - Covariance loss: That enforces sparsity of the covariance matrix to maintain independance between dimensions (for Z and Z' independantly)
