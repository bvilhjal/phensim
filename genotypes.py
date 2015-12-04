"""
Code for simulating genotypes
"""

import scipy as sp
from scipy import stats
from scipy import linalg


def get_sample_D(num_sim=100, conseq_r2=0.9, m=100, n=1000):
    """
    Simulates genotype regions, according to consecutive simulation scheme, and estimated the D matrix.
    """
    print 'Simulating genotypes for %d individuals and %d markers' % (n, m)
    D_avg = sp.zeros((m,m))
    for sim_i in range(num_sim):
        # Generating correlated training genotypes
        X = sp.empty((m, n))
        X[0] = stats.norm.rvs(size=n)        
        for j in range(1, m):
            X[j] = sp.sqrt(conseq_r2) * X[j - 1] + sp.sqrt(1 - conseq_r2) * stats.norm.rvs(size=n)
        
        X = sp.mat(X).T
        # Make sure that they have 0 mean and variance 1. (Normalizing)
        X_means = X - sp.mean(X, 0)
        X = X_means / sp.std(X_means, 0) 
        return_dict = {'X': X.T}
        # Calculating the marker correlation matrix (if needed)

        D_avg+= X.T * X / n
    D_avg = D_avg/num_sim
    return D_avg


def simulate_genotypes_w_ld(n_sample=100, m=50000, conseq_r2=0.9, m_ld_chunk_size=100, diploid=False, verbose=False):
    """
    Simulates genotype regions, according to consecutive simulation scheme, and estimated the D matrix.
    """
    if verbose:
        print 'Simulating genotypes for %d individuals and %d markers' % (n_sample, m)
        
    if diploid:
        print 'Simulating diploid dosages {0,1,2}'
        snps = sp.zeros((m,2*n_sample),dtype='single')
        assert m%m_ld_chunk_size==0,'WTF?'
        num_chunks = m/m_ld_chunk_size
        for chunk_i in range(num_chunks):
            # Generating correlated training genotypes
            X = sp.empty((m_ld_chunk_size, 2*n_sample))
            X[0] = stats.norm.rvs(size=2*n_sample)        
            for j in range(1, m_ld_chunk_size):
                X[j] = sp.sqrt(conseq_r2) * X[j - 1] + sp.sqrt(1 - conseq_r2) * stats.norm.rvs(size=2*n_sample)
                    
            start_i = chunk_i*m_ld_chunk_size
            stop_i = start_i+m_ld_chunk_size
            snps[start_i:stop_i]=X
        
        snps_means = sp.median(snps,axis=1)
        snps_means.shape = (m,1)
        bin_snps = sp.array(snps>snps_means,dtype='int8')
        snps = sp.array(bin_snps[:,:n_sample] + bin_snps[:,n_sample:],dtype='int8')
    else:
        snps = sp.zeros((m,n_sample),dtype='single')
        assert m%m_ld_chunk_size==0,'WTF?'
        num_chunks = m/m_ld_chunk_size
        for chunk_i in range(num_chunks):
            # Generating correlated training genotypes
            X = sp.empty((m_ld_chunk_size, n_sample))
            X[0] = stats.norm.rvs(size=n_sample)        
            for j in range(1, m_ld_chunk_size):
                X[j] = sp.sqrt(conseq_r2) * X[j - 1] + sp.sqrt(1 - conseq_r2) * stats.norm.rvs(size=n_sample)
                    
            start_i = chunk_i*m_ld_chunk_size
            stop_i = start_i+m_ld_chunk_size
            snps[start_i:stop_i]=X
        
        #Normalize SNPs
        snps_means = sp.mean(snps,axis=1)    
        snps_stds = sp.std(snps,axis=1)    
        snps_means.shape = (m,1)
        snps_stds.shape = (m,1)
        snps = (snps-snps_means)/snps_stds
        
    return snps

def simulate_genotypes_wo_ld(n_sample=100, m=50000, verbose=False):
    """
    Simulates gaussian distributed genotypes.
    
    Returns normalized genotypes.
    """
    if verbose:
        print 'Simulating genotypes for %d individuals and %d markers' % (n_sample, m)
    snps = stats.norm.rvs(size=(m,n_sample))
    
    # Make sure that they have 0 mean and variance 1. (Normalizing)
    snps_means = sp.mean(snps, 1)
    snps_stds = sp.std(snps,1)
    snps_means.shape = (m,1)
    snps_stds.shape = (m,1)
    snps = (snps - snps_means) / snps_stds
    
    return snps



def simulate_genotypes_w_ld_old(n=10000, m=100, ld=0.8, return_ne=False, ld_window_size=0, verbose=False):
    """
    Simulating genotypes w. LD
    
    m: number of causal variants
    n: number of individuals
    h2: heritability
    p: prior threshold
    
    if ld_window_size > 0, then LD in tiling windows are stored.
    """
    if verbose:
        print 'Simulating genotypes for %d individuals and %d markers' % (n, m)
    # Generating correlated training genotypes
    X = sp.empty((m, n))
    X[0] = stats.norm.rvs(size=n)        
    for j in range(1, m):
        X[j] = sp.sqrt(ld) * X[j - 1] + sp.sqrt(1 - ld) * stats.norm.rvs(size=n)
#         assert sp.corrcoef(X[j],X[j - 1])[0,1]<1, 'WTF?'
    X = sp.mat(X).T
    # Make sure that they have 0 mean and variance 1. (Normalizing)
    X_means = X - sp.mean(X, 0)
    X = X_means / sp.std(X_means, 0) 
    return_dict = {'X': X.T}
    # Calculating the marker correlation matrix (if needed)
    if ld_window_size == 0:
        D = X.T * X / n
#        D_inv = sp.mat(linalg.pinv2(D)) 
        return_dict['D'] = D
    
    elif ld_window_size > 0:
        assert m % ld_window_size == 0, 'm needs to be divisible by the ld_window_size'
        D = []
        # D_inv = []
        for i in sp.arange(0, m, ld_window_size):
            X_w = X[:, i:i + ld_window_size]
            D_w = X_w.T * X_w / n
            D.append(D_w)
            # D_inv_w = sp.mat(linalg.pinv2(D_w)) 
            # D_inv.append(D_inv_w)
        return_dict['D'] = D
  
    return return_dict


# def simulate_hdf5_genotypes_w_ld(ns=[1000, 5000, 10000, 20000], m=[5000, 10000, 20000, 50000],
#                                  ld=[0, 0.5, 0.8, 0.9], return_ne=False, ld_window_size=100):
#     """
#     Simulating genotypes w. LD
#     
#     m: number of causal variants
#     n: number of individuals
#     h2: heritability
#     p: prior threshold
#     
#     if ld_window_size > 0, then LD in tiling windows are stored.
#     """
#     print 'Simulating genotypes for %d individuals and %d markers' % (n, m)
#     # Generating correlated training genotypes
#     X = sp.empty((m, n))
#     X[0] = stats.norm.rvs(size=n)        
#     for j in range(1, m):
#         X[j] = sp.sqrt(ld) * X[j - 1] + sp.sqrt(1 - ld) * stats.norm.rvs(size=n)
#     X = sp.mat(X).T
#     # Make sure that they have 0 mean and variance 1. (Normalizing)
#     X_means = X - sp.mean(X, 0)
#     X = X_means / sp.std(X_means, 0) 
#     return_dict = {'X':X}
#     
#     # Calculating the marker correlation matrix (if needed)
#     if ld_window_size == 0:
#         D = X.T * X / n
# #        D_inv = sp.mat(linalg.pinv2(D)) 
#         return_dict['D'] = D
#     
#     elif ld_window_size > 0:
#         assert m % ld_window_size == 0, 'm needs to be divisible by the ld_window_size'
#         D = []
#         # D_inv = []
#         for i in sp.arange(0, m, ld_window_size):
#             X_w = X[:, i:i + ld_window_size]
#             D_w = X_w.T * X_w / n
#             D.append(D_w)
#             # D_inv_w = sp.mat(linalg.pinv2(D_w)) 
#             # D_inv.append(D_inv_w)
#         return_dict['D'] = D
#   
# 
#     return return_dict



def simulate_rare_genotypes(n=1000, m=10000, exp_scale=0.1):
    """
    Simulating unlinked rare genotypes, rounding up exponentially distributed values.
    
    m: number of causal variants
    n: number of individuals    
    """
    snps = stats.expon.rvs(0, scale=exp_scale, size=(m, n))
    snps = sp.round_(snps)
    snps[snps > 1] = 1
    snps = snps[sp.sum(snps, 1) > 0]
    
    print 'Simulated %d dimorphic SNPs' % len(snps)
    
    return snps


def simulate_k_tons(k=1, n=1000, m=10000):
    """    
    m: number of causal variants
    n: number of individuals    
    """
    snps = sp.zeros((m, n))
    for i in range(m):
        for j in range(k):
            snps[i, sp.random.randint(0, n)] = 1
    
    print 'Simulated %d  SNPs' % len(snps)
    return snps


def simulate_common_genotypes(n=1000, m=10000):    
    """
    Simulate genotypes
    """
    snps = sp.random.random((m, n))
    snps = sp.round_(snps)
    snps = sp.array(snps, dtype='int8')
    snps = snps[sp.sum(snps, 1) > 0]
    print 'Done Simulating SNPs.'
    return snps    


    

# def simulate_phenotypes_w_ld(n=1000, m=100, effect_prior='gaussian',
#                            p=1.0, h2=0.5, ld=0.8, return_ne=False, ** kwargs):
#    """
#    Simulating genotypes w. LD
#    
#    m: number of causal variants
#    n: number of individuals
#    h2: heritability
#    p: prior threshold
#    
#    Calls simulate genotypes 
#    """
#
#    #Generating the effects
#    if effect_prior == 'gaussian':
#        if p == 1.0:
#            betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
#        else:
#            mp = max(1, int(round(m * p)))
#            betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / mp), size=mp), sp.zeros(m - mp, dtype=float)))
#    elif effect_prior == 'laplace':
#        if p == 1.0:
#            betas = stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * m)), size=m)
#        else:
#            mp = max(1, int(round(m * p)))
#            betas = sp.concatenate((stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * mp)), size=mp), sp.zeros(m - mp, dtype=float)))
#    betas_var = sp.var(betas)
#    beta_scalar = sp.sqrt(h2 / (m * betas_var))
#    betas = betas * beta_scalar
#
#    #Generating the genotypes:
#    train_genotypes = simulate_genotypes_w_ld(n=n, m=m, ld=ld, return_ne=return_ne)
#    #Generating the phenotypes
#    phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
#    phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
#    genetic_part = sp.dot(train_genotypes['X'], betas)
#    genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
#    phen = genetic_part + phen_noise
#    betas_marg = (1. / n) * sp.dot(phen, train_genotypes['X']).T
#    
#    return_dict = {'betas':betas, 'phen':phen, 'betas_marg':betas_marg, 'D':train_genotypes['D'],
#            'D_inv':train_genotypes['D_inv']}
#    if return_ne:
#        return_dict['n_e'] = train_genotypes['n_e']
#
#    return return_dict
