import numpy as np

def minibatch_data(data, batch_size, random_seed=45):
    """
    Create minibatch samples from the dataset
    """
    n = data.shape[0]
    p = data.shape[1]
    if n % batch_size != 0:
        n = (n // batch_size) * batch_size
    ind = np.arange(n)
    np.random.shuffle(ind)
    n_minibatches = n // batch_size
    data = data[ind].reshape(batch_size, p, n_minibatches)
    return(data, n_minibatches)

def sghmc(gradU, eps, C, inv_M, theta_0, V_hat, data, batch_size, burn_in, n_iter=500):
    '''
    Define SGHMC as described in the paper Stochastic Gradient Hamilton Monte Carlo, 
    ICML, Beijing, China, 2014 by
    Tianqi Chen, Emily B. Fox, Carlos Guestrin.

    The inputs are:
    gradU = gradient of U
    eps = the learning rate
    C = user specified friction term
    inv_M = inverse of the mass matrix
    theta_0 = initial value of parameter sampling
    V_hat = estimated covariance matrix using empirical Fisher information
    batch_size = size of a minibatch in an iteration
    burn_in = number of iterations to drop
    n_iter = number of samples to generate

    The outpit is:
    theta_samples: a np.array of positions of theta.
    '''

    # parameter vector dimension
    p = theta_0.shape[0]
    # number of samples
    n = data.shape[0]
    # placeholder for theta samples
    theta_samples = np.zeros((p, n_iter))
    theta_samples[:, 0] = theta_0
    
    # fix beta_hat as described on pg. 6 of paper
    beta_hat = (V_hat * eps) / 2
    Sigma = np.linalg.cholesky(2 * (C - beta_hat) * eps)
    
    # data
    mini_data, n_batches = minibatch_data(data, batch_size)

    # assert batch size to be <= the amount of data
    if (batch_size > data.shape[0]): 
        print("Error: batch_size cannot be bigger than the number of samples")
        return
    
    # loop through algorithm to get n iteration samples
    for i in range(n_iter - 1):
        theta = theta_samples[:, i]
        # resample momentum at each new iteration
        M = np.linalg.cholesky(np.linalg.inv(inv_M))
        momen = M@np.random.randn(p).reshape(p, -1)
        
        # sghmc sampler
        for j in range(n_batches):
            theta = theta + (eps*inv_M@momen).flatten()
            gradU_batch = gradU(theta, mini_data[:,:,j], n, batch_size).reshape(p, -1)
            momen = momen - eps*gradU_batch - eps*C@inv_M@momen + Sigma@np.random.randn(p).reshape(p, -1)
            
        theta_samples[:, i+1] = theta
        
    return theta_samples[:, burn_in:]