

'''

Notes on BN
https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/

Addresses the problem of Internal Covariate Shift which is a change in data distribution
as weights are updated

3 steps:
1 Compute mean, stdv along axix = 0
2 Normzliate (x-mean)/(std+eps) eps to handle div by 0
3 rescale n shift
 gamma * hnorm + beta

Benefits of Batch Normalization

Speeds up learning: Reducing internal covariate shift helps the model train faster.
Regularizes the model: It adds a little noise to your model, and in some cases, you might
 not even need to use dropout or other regularization techniques.
Allows higher learning rates: Gradient descent usually requires small learning rates
for the network to converge. Batch normalization helps us use much larger learning
rates, speeding up the training process.

'''


import numpy as np

def batch_norm(X, gamma, beta, eps = 1e-8):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mean) / np.sqrt(var + eps)
    out = gamma * X_norm + beta

    return out



def batch_norm2(X, eps=1e-8):
    if len(X.shape) not in (2, 4):
        raise ValueError("Input array should be 2D or 4D")

    if len(X.shape) == 2:
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
    else:
        mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        std_dev = np.std(X, axis=(0, 2, 3), keepdims=True)

    X_normalized = (X - mean) / (std_dev + eps)
    return X_normalized