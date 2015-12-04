"""
Code for simulating phenotypes
"""
import scipy as sp
from scipy import stats
from scipy import linalg
import h5py
import sys
import os
import genotypes


def simulate_traits(n=1000, m=100, hdf5_file_prefix=None, hdf5_group=None,
                    num_traits=1000, h2=0.5, effect_prior='gaussian', p=1.0,
                    conseq_ld=0, overwrite_hdf5=False, test_n=1000, simulate_validation_traits=True):
    """
    Simluate traits:
    First simulate SNPs, then simulate the traits
    
    """
    
    print "Using %d SNPs to simulate traits for %d individuals." % (m, n)
    
    genotype_dict = genotypes.simulate_genotypes_w_ld(n=n, m=m, ld=conseq_ld, return_ne=False, ld_window_size=0)
    snps = genotype_dict['X']
    betas_list = []
    betas_marg_list = []
    phen_list = []
    for i in range(num_traits):
        if effect_prior == 'gaussian':
            if p == 1.0:
                betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M), sp.zeros(m - M, dtype=float)))
        elif effect_prior == 'laplace':
            if p == 1.0:
                betas = stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * m)), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * M)), size=M), sp.zeros(m - M, dtype=float)))
                    
        betas_var = sp.var(betas)
        beta_scalar = sp.sqrt(h2 / (m * betas_var))
        betas = betas * beta_scalar
        betas_list.append(betas)
        phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
        phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
        genetic_part = sp.dot(snps, betas)
        genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
        train_phen = genetic_part + phen_noise
        print 'Herit:', sp.var(genetic_part) / sp.var(train_phen)        
        phen_list.append(train_phen)
        betas_marg = (1. / n) * sp.dot(train_phen, snps)
        betas_marg_list.append(betas_marg)

        sys.stdout.write('\b\b\b\b\b\b\b%0.1f%%' % (100.0 * (float(i) / num_traits)))
        sys.stdout.flush()

    if hdf5_file_prefix != None:
        hdf5_file = '%s_p_%0.4f.hdf5' % (hdf5_file_prefix, p)
        if os.path.isfile(hdf5_file):
            print 'File already exists.'
            if overwrite_hdf5:
                print 'Overwriting %s' % hdf5_file
                os.remove(hdf5_file)
            else:
                print 'Attempting to continue.'
            
        h5f = h5py.File(hdf5_file)
        h5f.create_dataset('phenotypes', data=phen_list, compression='gzip')
        h5f.create_dataset('betas', data=betas_list, compression='gzip')
        h5f.create_dataset('betas_marg', data=betas_marg_list, compression='gzip')
    elif hdf5_group != None:
        hdf5_group.create_dataset('phenotypes', data=phen_list, compression='gzip')
        hdf5_group.create_dataset('betas', data=betas_list, compression='gzip')
        hdf5_group.create_dataset('betas_marg', data=betas_marg_list, compression='gzip')        
    else:
        print 'Warning: No storage file given!'
    print '.'
    print "Done simulating data."
    return phen_list


def simulate_beta_hats(h2=0.5, n=100000, n_sample=100, m=50000, model='gaussian', 
                   p=1.0, conseq_r2=0, m_ld_chunk_size=100, D_sample=None):
    """
    Implements an efficient simulation of correlated least square effects, etc.

    models:  Gaussian dist.
             Gaussian dist. with 0-threshold
             Laplace dist.
             
    """
    if conseq_r2>0:
        assert m%m_ld_chunk_size==0, 'The number of SNPs needs to be a multiple of the LD chunk size'
    
    if model == 'gaussian':
        if p == 1.0:
            betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
        else:
            M = int(round(m * p))
            betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M), sp.zeros(m - M, dtype=float)))
            sp.random.shuffle(betas)
    elif model == 'laplace':
        if p == 1.0:
            betas = stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * m)), size=m)
        else:
            M = int(round(m * p))
            betas = sp.concatenate((stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * M)), size=M), sp.zeros(m - M, dtype=float)))
            sp.random.shuffle(betas)
    elif model=='gaussian_clumped':
        assert conseq_r2>0, "Simulating phenotypes w clumped causal effects makes no sense with unlinked genotypes!"
        if p==1:
            betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
        else:
            betas = sp.zeros(m)
            M = int(round(m * p))
            for m_i in range(0,m,m_ld_chunk_size):
                p_reg = stats.beta.rvs(p,1-p)
                M_chunk = int(round(m_ld_chunk_size * p_reg))
                betas_chunk = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M_chunk), sp.zeros(m_ld_chunk_size - M_chunk, dtype=float)))
                sp.random.shuffle(betas_chunk)
                betas[m_i: m_i+m_ld_chunk_size] = betas_chunk
            
    betas_var = sp.var(betas)
    beta_scalar = sp.sqrt(h2 / (m * betas_var))
    betas = betas * beta_scalar
    ret_dict = {'betas':betas}
    noises = stats.norm.rvs(0,1,size=m)
    if conseq_r2==0:
        beta_hats = betas + sp.sqrt(1.0 / n)*noises
        
    else:
        assert D_sample !=None, 'D_sample is missing...'
        C = sp.sqrt(((1.0)/n))*linalg.cholesky(D_sample)
        D_I = linalg.pinv(D_sample)
        betas_ld = sp.zeros(m)
        noises_ld = sp.zeros(m)
        for m_i in range(0,m,m_ld_chunk_size):
            m_end = m_i+m_ld_chunk_size
            betas_ld[m_i:m_end] = sp.dot(D_sample,betas[m_i:m_end])
            noises_ld[m_i:m_end]  = sp.dot(C.T,noises[m_i:m_end])
        ret_dict['D_sample']=D_sample
        ret_dict['betas_ld']=betas_ld
        beta_hats = betas_ld + noises_ld
        
        betas_cojo = sp.zeros(m)
        for m_i in range(0,m,m_ld_chunk_size):
            m_end = m_i+m_ld_chunk_size
            betas_cojo[m_i:m_end] = sp.dot(D_I,beta_hats[m_i:m_end])
        ret_dict['betas_cojo']=betas_cojo

    ret_dict['beta_hats']=beta_hats
    return ret_dict


def simulate_betas(num_traits=1000, p=0.1, m=100, h2=0.5, effect_prior='gaussian', verbose=False):
    betas_list = []
    for i in range(num_traits):
        if effect_prior == 'gaussian':
            if p == 1.0:
                betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M), sp.zeros(m - M, dtype=float)))
        elif effect_prior == 'laplace':
            if p == 1.0:
                betas = stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * m)), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * M)), size=M), sp.zeros(m - M, dtype=float)))
                    
        betas_var = sp.var(betas)
        beta_scalar = sp.sqrt(h2 / (m * betas_var))
        betas = betas * beta_scalar
        sp.random.shuffle(betas)
        betas_list.append(betas)
    return sp.array(betas_list)


def generate_test_data_w_sum_stats(h2=0.5, n=100000, n_sample=100, m=50000, model='gaussian', 
                                         p=1.0, conseq_r2=0, m_ld_chunk_size=100):
    """
    Generate 
    """
    #Get LD sample matrix
    D_sample = genotypes.get_sample_D(200,conseq_r2=conseq_r2,m=m_ld_chunk_size)
    
    #Simulate beta_hats
    ret_dict = simulate_beta_hats(h2=h2, n=n, n_sample=n_sample, m=m, model=model, p=p, 
                                    conseq_r2=conseq_r2, m_ld_chunk_size=m_ld_chunk_size, D_sample=D_sample)
    
    #Simulate test genotypes
    test_snps = genotypes.simulate_genotypes_w_ld(n_sample=n_sample, m=m, conseq_r2=conseq_r2, 
                                                  m_ld_chunk_size=m_ld_chunk_size)
    ret_dict['test_snps'] = test_snps
    
    #Simulate test phenotypes
    phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n_sample) 
    phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
    genetic_part = sp.dot(test_snps.T, ret_dict['betas'])
    genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
    test_phen = genetic_part + phen_noise
    ret_dict['test_phen'] = test_phen
    return ret_dict


def generate_train_test_phenotypes(betas, train_snps, test_snps, h2=0.01):
    """
    Generate genotypes given betas and SNPs
    """
    (m, n) = train_snps.shape
    (test_m, test_n) = test_snps.shape
    assert len(betas) == m == test_m, 'WTF?'
    
    #Training phenotypes
    phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
    phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
    genetic_part = sp.dot(train_snps.T, betas)
    genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
    train_phen = genetic_part + phen_noise
#        print 'Herit:', sp.var(genetic_part) / sp.var(train_phen)        
    ret_dict = {}
    ret_dict['phen'] = train_phen
    betas_marg = (1. / n) * sp.dot(train_phen, train_snps.T)
    ret_dict['betas_marg'] = betas_marg
    
    #Testing phenotypes
    phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=test_n) 
    phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
    genetic_part = sp.dot(test_snps.T, betas)
    genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
    test_phen = genetic_part + phen_noise
    ret_dict['test_phen'] = test_phen
    return ret_dict


def simulate_traits_w_snps(snps, num_traits=1000, p=0.1, m=100, h2=0.5, effect_prior='gaussian', verbose=False, liability_thres=None):
    """
    Simluate traits with SNPs given.
    
    Assumes that the SNPs are normalized.
    """
    (m, n) = snps.shape
    print n, m
    print "Using %d SNPs to simulate %d traits for %d individuals." % (m, num_traits, n)
    
    # Check if SNPs are normalized, if not, then normalize!
    
    betas_list = []
    phen_list = []
#     test_phen_list = []
    if liability_thres!=None:
        cc_phen_list = []
    for i in range(num_traits):
        if effect_prior == 'gaussian':
            if p == 1.0:
                betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M), sp.zeros(m - M, dtype=float)))
                sp.random.shuffle(betas)

        elif effect_prior == 'laplace':
            if p == 1.0:
                betas = stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * m)), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * M)), size=M), sp.zeros(m - M, dtype=float)))
                sp.random.shuffle(betas)
        phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
        phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
        genetic_part = sp.dot(snps.T, betas)
        betas_scalar = (sp.sqrt(h2 / sp.var(genetic_part)))
        betas = betas * betas_scalar
        betas_list.append(betas)
        genetic_part = betas_scalar * (genetic_part - sp.mean(genetic_part))
        train_phen = genetic_part + phen_noise
        train_phen = (train_phen-sp.mean(train_phen))/sp.std(train_phen)
        if verbose:
            print 'Herit:', sp.var(genetic_part) / sp.var(train_phen)        
        phen_list.append(train_phen)
#         betas_marg = (1. / n) * sp.dot(train_phen, snps.T)
#         betas_marg_list.append(betas_marg)
#         phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
#         phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
#         test_phen = genetic_part + phen_noise            
#         test_phen_list.append(test_phen)
        if liability_thres!=None:
            cc_trait = sp.array(train_phen>liability_thres,'int8')
            cc_phen_list.append(cc_trait)
        if verbose:
            sys.stdout.write('\b\b\b\b\b\b\b%0.1f%%' % (100.0 * (float(i) / num_traits)))
            sys.stdout.flush()

    ret_dict = {}
    ret_dict['phenotypes'] = phen_list
    if liability_thres!=None:
        ret_dict['cc_phenotypes'] = cc_phen_list

#     ret_dict['test_phenotypes'] = test_phen_list
    assert len(betas_list) == num_traits, 'WTF?'
    ret_dict['betas'] = betas_list
    print '.'
    print "Done simulating data."
    return ret_dict




def simulate_traits_w_snps_to_hdf5(snps, hdf5_file_prefix='/Users/bjarnivilhjalmsson/tmp/test',
                    num_traits=1000, h2=0.5, effect_prior='gaussian', p=1.0):
    """
    Simluate traits w SNPs.
    
    Assumes that the SNPs are normalized.
    """
    (m, n) = snps.shape
    print "Using %d SNPs to simulate traits for %d individuals." % (m, n)
    
    # Check if SNPs are normalized, if not, then normalize!
    norm_snps = snps.T
    norm_snps = (norm_snps - sp.mean(norm_snps, 0)) / sp.std(norm_snps, 0)
    
    betas_list = []
    betas_marg_list = []
    phen_list = []
    for i in range(num_traits):
        if effect_prior == 'gaussian':
            if p == 1.0:
                betas = stats.norm.rvs(0, sp.sqrt(h2 / m), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.norm.rvs(0, sp.sqrt(h2 / M), size=M), sp.zeros(m - M, dtype=float)))
        elif effect_prior == 'laplace':
            if p == 1.0:
                betas = stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * m)), size=m)
            else:
                M = int(round(m * p))
                betas = sp.concatenate((stats.laplace.rvs(scale=sp.sqrt(h2 / (2 * M)), size=M), sp.zeros(m - M, dtype=float)))
                    
        betas_var = sp.var(betas)
        beta_scalar = sp.sqrt(h2 / (m * betas_var))
        betas = betas * beta_scalar
        betas_list.append(betas)
        phen_noise = stats.norm.rvs(0, sp.sqrt(1.0 - h2), size=n) 
        phen_noise = sp.sqrt((1.0 - h2) / sp.var(phen_noise)) * phen_noise
        genetic_part = sp.dot(norm_snps, betas)
        genetic_part = sp.sqrt(h2 / sp.var(genetic_part)) * genetic_part
        train_phen = genetic_part + phen_noise
        print 'Herit:', sp.var(genetic_part) / sp.var(train_phen)        
        phen_list.append(train_phen)
        betas_marg = (1. / n) * sp.dot(train_phen, norm_snps)
        betas_marg_list.append(betas_marg)

        sys.stdout.write('\b\b\b\b\b\b\b%0.1f%%' % (100.0 * (float(i) / num_traits)))
        sys.stdout.flush()

    hdf5_file = '%s_p_%0.4f.hdf5' % (hdf5_file_prefix, p)
    if os.path.isfile(hdf5_file):
        print 'Overwriting %s' % hdf5_file
        os.remove(hdf5_file)
    h5f = h5py.File(hdf5_file)
    h5f.create_dataset('phenotypes', data=phen_list, compression='gzip')
    h5f.create_dataset('betas', data=betas_list, compression='gzip')
    h5f.create_dataset('betas_marg', data=betas_marg_list, compression='gzip')
    print '.'
    
    ret_dict = {}
    ret_dict['phenotypes'] = phen_list
    assert len(betas_list) == num_traits, 'WTF?'
    ret_dict['betas'] = betas_list
    ret_dict['betas_marg'] = betas_marg_list
    print "Done simulating data."
    return ret_dict