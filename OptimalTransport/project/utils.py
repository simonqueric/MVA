import numpy as np
from scipy.linalg import sqrtm

## Optimal Transport functions ##

def ot_gauss(ma, mb, Sa, Sb) :
    """
    Compute O.T Wasserstein distance between 
    two gaussians a and b with closed form solution 
    
    input
    -----
    ma : 1d array mean of gaussian distribution a
    mb : 1d array mean of gaussian distribution b
    Sa : square matrix, covariance matrix of a
    Sb : square matrix, covariance matrix of b
    
    
    output
    ------
    W : Exact Wasserstein distance between a and b
    """
    B2 = np.trace(Sa + Sb - 2*sqrtm(sqrtm(Sa)@Sb@sqrtm(Sa)))
    W2 = np.linalg.norm(ma-mb)**2 + B2
    return W2

def ot_gaussian_regularized(ma, mb, Sa, Sb, eps) :
    """
    Compute regularized O.T Wasserstein distance between 
    two gaussians a and b with closed form solution 
    
    input
    -----
    ma : 1d array mean of gaussian distribution a
    mb : 1d array mean of gaussian distribution b
    Sa : square matrix, covariance matrix of a
    Sb : square matrix, covariance matrix of b
    eps : regularizer
    
    output
    ------
    W : Exact Wasserstein distance between a and b
    """
    d = len(ma)
    Id = np.eye(d)
    Deps = sqrtm( 4*(sqrtm(Sa)@Sb@sqrtm(Sa)) + eps**2/4*Id)
    
    _, logdet = np.linalg.slogdet(Deps + eps/2 * Id)
    B2eps = np.trace(Sa + Sb - Deps) + d*eps/2 * (1-np.log(eps)) + eps / 2 * logdet
    W2eps = np.linalg.norm(ma-mb)**2 + B2eps
    return W2eps

def ot_1d_uniform(a, b, c, d):
    """
    OT distance between 1D uniform distributions on (a, b) and (c, d) 
    """    
    r = (d-c)/(b-a)
    t = c - a*(d-c)/(b-a)
    return (1-r)**2/(b-a) *(b**3-a**3)/3 + (1-r)*t/(b-a)*(b**2-a**2) + t**2

def distmat(x,y):
    """
    Compute square euclidean distance matrix between two point clouds x and y
    
    input
    -----
    x : array of shape d x n
    y : array of shape d x n
    
    output
    ------
    C : matrix with entries ||xi - yj||^2_2
    """
    return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)

def mina_u(H,a,epsilon): return -epsilon*np.log( np.sum(a * np.exp(-H/epsilon),0) )
def minb_u(H,b,epsilon): return -epsilon*np.log( np.sum(b * np.exp(-H/epsilon),1) )
def mina(H,a,epsilon): return mina_u(H-np.min(H,0),a,epsilon) + np.min(H,0);
def minb(H,b,epsilon): return minb_u(H-np.min(H,1)[:,None],b,epsilon) + np.min(H,1);

def logsumexp(x):
    """
    Logsumexp function of x using the logsumpexp trick
    input
    -----
    x : array of size n
    
    output
    ------
    logsumexp(x) : float
    """
    c = np.max(x)
    return np.log(np.sum(np.exp(x-c))) + c

def cost(x, y):
    """
    Square euclidean distance between two vectors x and y
    """
    return np.linalg.norm(x - y)**2

def W(u, v, x, y, cost, eps) :
    """ 
    Compute wasserstein distance between point clouds 
    with sinkhorn potentials.  
    input
    -----
    x, y : point clouds 
    u, v : sinkhorn potentials
    cost : cost function
    eps : regularizer
    
    output
    ------
    wasserstein distance between 
    
    use case : 
    - draw x and y from proba distributions mu and nu
    - compute sinkhorn potentials u and v with Sinkhorn algorithm 
    - compute wasserstein distance W(mu, nu) 
    """
    n = len(u)
    c = np.zeros(n)
    for i in range(len(c)) :
        c[i] = cost(x[i], y[i])
    h = u+v-c
    return np.mean(u) + np.mean(v) - (eps/n) *logsumexp(h/eps)+eps

def sinkhorn(C, eps, a, b, niter=500):
    K = np.exp(-C/eps)
    n = len(a)
    m = len(b)
    v = np.ones(n)
    Err_p = []
    Err_q = []
    for i in range(niter):
        # sinkhorn step 1
        u = a / (np.dot(K,v))
        # error computation
        r = v*np.dot(np.transpose(K),u)
        Err_q = Err_q + [np.linalg.norm(r - b, 1)]
        # sinkhorn step 2
        v = b /(np.dot(np.transpose(K),u))
        s = u*np.dot(K,v)
        Err_p = Err_p + [np.linalg.norm(s - a,1)]

    return u, v, K

def log_sinkhorn(C,epsilon,f, a, b, niter = 500):    
    Err = np.zeros(niter)
    for it in range(niter):
        g = mina(C-f[:,None],a, epsilon)
        f = minb(C-g[None,:], b, epsilon)
        # generate the coupling
        P = a * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b
        # check conservation of mass
        Err[it] = np.linalg.norm(np.sum(P,0)-b,1)
    return f, g, P, Err


def approx_wasserstein(x, y, eps) :
    """
    Approximation of Wasserstein distance for empirical distribution x and y
    """
    n = len(x)
    a = np.ones(n)/n
    b = np.ones(n)/n    
    C = distmat(x, y)
    u, v, _, _ = log_sinkhorn(C,eps,np.zeros(n), a,b, niter = 500)
    return np.mean(u)+np.mean(v)

def approx_normalized_wasserstein(x, y, n, eps) :
    """
    Approximation of normalized Wasserstein distance for empirical distribution x and y
    """
    n = len(x)
    a = np.ones(n)/n
    b = np.ones(n)/n    
    C = distmat(x,y)
    u, v, _, _ = log_sinkhorn(C,eps,np.zeros(n), a,b, niter = 500)
    Cx = distmat(x,x)
    ux, vx, _, _ = log_sinkhorn(C,eps,np.zeros(n), a,b, niter = 500)
    Cy = distmat(x,y)
    uy, vy, _, _ = log_sinkhorn(C,eps,np.zeros(n), a,b, niter = 500)
    return np.mean(u)+np.mean(v) - (np.mean(ux)+np.mean(vx)+np.mean(uy) +np.mean(vy))/2

## Vizualisation ##

def plot(T, X, alpha=0.2):
    Xmean = X.mean(axis=1)
    Xstd = X.std(axis=1)
    plt.plot(T, Xmean)
    plt.fill_between(T, Xmean-Xstd, Xmean+Xstd, alpha=alpha, label='_nolegend_')