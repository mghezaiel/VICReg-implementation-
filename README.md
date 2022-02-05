# VICReg-implementation-
Implementation of the Variance Invariance Covariance regularization method from the paper:

<i>Bardes, A., Ponce, J., & LeCun, Y. (2021).</i><br>
    
<b>VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.</b> ArXiv, abs/2105.04906.
   
I adapted the work to non image data (matrices) for self-supervised representation learning. 
    
All modules were implemented in the local repository and following what has been described in the paper. 

    
1) The library contains:

        - A simple encoder consisting in a single dense layer followed by relu activation. 
        - A projector with the same architecture that aims to project the data in a higher dimensional space.
        - The Variance - Invariance - Covariance loss applied to the projected data.
        - An optuna based optimization class to optimize: loss hyperparameters (mu and lambda, cf paper) and training hypeparameters. 
        - A class to evaluate embeddings quality by comparing silhouette coefficient based clustering separability between:
            a) Input data
            b) Predicted embeddings. 
    


2) Short explanation of VICReg:

The work was initially developped to produce embeddings from image data that are subject to data augmentation such as affine transformation or cropping.
The idea is about trying to learn similar embeddings between different views of the data (ie: augmented views). 
In the present implementation, we use it for representation learning of non image data.

According to the paper, the architecture consists in: 

    1) Two siamese networks (encoders) that produce embeddings (Y and Y')
    2) Two other networks (decoders or projectors) that projects the embeddings to a higher dimensional space (Z and Z')

![alt text](https://github.com/mghezaiel/VICReg-implementation-/blob/master/architecture.png)

Consistency of the learned embeddings (Y and Y') with the input data (X and X') is expected to be achieved through regularization of the projections (Z and Z').
    
Regularization is performed by defining the VICReg loss:

    - Variance loss: That maintains variance inside a batch (for Z and Z' independantly)
    - Invariance loss: That maintains similarity between embeddings Z and Z' (MSE) 
    - Covariance loss: That enforces covariance matrix sparsity to maintain independance between dimensions (for Z and Z' independantly)
