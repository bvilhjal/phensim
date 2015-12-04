"""
Code for testing methods to ward against colliders.
"""

import scipy as sp
from scipy import linalg
from scipy import stats
import genotypes
import matplotlib
import matplotlib.pyplot as plt
import pylab


def test_null(n=100, num_traits=1000, corr=0.8, beta_c=0, beta_y=0, alpha_thres=0.05):
    """
    # 1. Generate test genotypes.
    # 2. Generate GE1 and GE2, w correlation.
    # 3. Construct phenotypes.
    # 4. Check direct association signal.
    # 5. Check covariate adjusted association signal.
    """
    snps = genotypes.simulate_common_genotypes(n=n,m=num_traits)
    #Normalize genotypes
    snps = snps.T
    snps = (snps - sp.mean(snps, 0)) / sp.std(snps, 0)
    snps = snps.T     
       
    V = sp.kron(sp.array([[1,corr],[corr,1]]),sp.eye(n))
    #print V
    GE = sp.random.multivariate_normal(sp.zeros(n*2), cov=V, size=num_traits)
    #print GE.shape
    GE1 = GE[:,:n]
    GE1_norm = GE1.T
    GE1_norm = (GE1_norm - sp.mean(GE1_norm, 0)) / sp.std(GE1_norm, 0)
    GE1_norm = GE1_norm.T
    
    GE2 = GE[:,n:]
    GE2_norm = GE2.T
    GE2_norm = (GE2_norm - sp.mean(GE2_norm, 0)) / sp.std(GE2_norm, 0)
    GE2_norm = GE2_norm.T

    gs = snps
    C = GE1_norm + beta_c*gs
#     print C[0], GE1_norm[0],gs[0]
#     print sp.var(C[0]), sp.var(GE1_norm[0]), sp.var(gs[0])
    #print C.shape, GE1.shape, gs.shape
    Y = GE2_norm + beta_y*gs

    #Normalize Y's and C's
    C_norm = C.T
    C_norm_factor = sp.std(C_norm, 0)
    C_norm = (C_norm - sp.mean(C_norm, 0))/ C_norm_factor
    C_norm = C_norm.T
    
    Y_norm = Y.T
    Y_norm = (Y_norm - sp.mean(Y_norm, 0)) / sp.std(Y_norm, 0)
    Y_norm = Y_norm.T
    
    est_corrs = sp.zeros(num_traits)
    for i in range(num_traits):
        est_corrs[i] = sp.corrcoef(C_norm[i],Y_norm[i])[0,1]
    print sp.mean(est_corrs)
    
    direct_betas_Y = sp.empty(num_traits)
    for i in range(num_traits):
        direct_betas_Y[i] = sp.dot(snps[i],Y[i])/n
    
    print 'Direct association on Y:'
#     print sp.mean(direct_betas_Y), sp.var(direct_betas_Y)
    rss0 = sp.sum(Y**2,1)
    res = Y-(snps.T*direct_betas_Y).T
#     print res.shape
    rss1 = sp.sum(res**2,1)
    f_stats = (rss0 - rss1)*(n-1)/rss1
    marg_Y_p_vals = stats.f.sf(f_stats, 1, n - 1)
    mean_marg_Y_pvals = sp.mean(marg_Y_p_vals)
    print 'Mean marginal Y p-value:', mean_marg_Y_pvals

    if beta_y==0:
        marg_Y_fdr = sp.sum(marg_Y_p_vals<alpha_thres)/float(num_traits)
        marg_Y_power = 0
    else:
        marg_Y_fdr = 0
        marg_Y_power = sp.sum(marg_Y_p_vals<alpha_thres)/float(num_traits)
    print 'Marginal Y power:', marg_Y_power    
    print 'Marginal Y FDR:', marg_Y_fdr    
        
    direct_betas_C = sp.empty(num_traits)
    for i in range(num_traits):
        direct_betas_C[i] = sp.dot(snps[i],C[i])/n
#     print len(direct_betas_C)
    print 'Direct association on C:'
    print sp.mean(direct_betas_C), sp.var(direct_betas_C)
    rss0 = sp.sum(C**2,1)
    C_SNP_resid = C-(snps.T*direct_betas_C).T
#     print res.shape
    rss1 = sp.sum(C_SNP_resid**2,1)
    f_stats = (rss0 - rss1)*(n-1)/rss1
    marg_C_p_vals = stats.f.sf(f_stats, 1, n - 1)
    mean_marg_C_pvals = sp.mean(marg_C_p_vals)
    print 'Mean marginal C p-value:', mean_marg_C_pvals

    if beta_c==0:
        marg_C_fdr = sp.sum(marg_C_p_vals<alpha_thres)/float(num_traits)
        marg_C_power = 0
    else:
        marg_C_fdr = 0
        marg_C_power = sp.sum(marg_C_p_vals<alpha_thres)/float(num_traits)
    print 'Marginal C power:', marg_C_power    
    print 'Marginal C FDR:', marg_C_fdr    
    
     
    adj_betas_Y = sp.empty((num_traits,2))
    for i in range(num_traits):
        X = sp.vstack([snps[i],C[i]])
        XX = sp.mat(sp.dot(X,X.T))
        XX_inv = linalg.inv(XX)
        adj_betas_Y[i] =sp.dot(XX_inv, sp.dot(X,Y[i].T).T)
#     print adj_betas_Y.shape
    print 'Adj association on Y:'
    print sp.mean(adj_betas_Y,0)
#     print adj_betas_Y
    adj_betas_Y_genot = adj_betas_Y[:,0]

    res0 = Y - (C.T*adj_betas_Y[:,1]).T
    rss0 = sp.sum(res0**2,1)
    res1 = Y-(snps.T*adj_betas_Y_genot).T - (C.T*adj_betas_Y[:,1]).T
#     print res.shape
    rss1 = sp.sum(res1**2,1)
    f_stats = (rss0 - rss1)*(n-2)/rss1
    adj_Y_p_vals = stats.f.sf(f_stats, 1, n - 2)
    mean_adj_pvals = sp.mean(adj_Y_p_vals)
    print 'Mean Adjust Y p-value:', mean_adj_pvals
    
    if beta_y==0:
        adj_Y_fdr = sp.sum(adj_Y_p_vals<alpha_thres)/float(num_traits)
        adj_Y_power = 0
    else:
        adj_Y_fdr = 0
        adj_Y_power = sp.sum(adj_Y_p_vals<alpha_thres)/float(num_traits)
    print 'Adj Y power:', adj_Y_power    
    print 'Adj Y FDR:', adj_Y_fdr    

    adj_Y_bias = sp.mean(beta_y-adj_betas_Y_genot)
 
    #Test for detecting a false association... 
    z_stats = sp.sqrt(n)*(adj_betas_Y_genot - direct_betas_Y)/est_corrs
    bias_p_vals = stats.f.sf(z_stats**2, 1, n - 1)
    mean_bias_p_val = sp.mean(bias_p_vals)
    
    if beta_c==0:
        bias_test_fdr = sp.sum(bias_p_vals<alpha_thres)/float(num_traits)
        bias_test_power = 0
    else:
        bias_test_fdr = 0
        bias_test_power = sp.sum(bias_p_vals<alpha_thres)/float(num_traits)
    print 'Adj bias-test power:', bias_test_power    
    print 'Adj bias-test FDR:', bias_test_fdr    
    
    
    #The conservative approach...
    
    
    ok_filter = sp.sign(direct_betas_Y)==sp.sign(adj_betas_Y_genot)
    ok_filter[ok_filter] = direct_betas_Y[ok_filter]**2 > adj_betas_Y_genot[ok_filter]**2
    print sp.sum(ok_filter)
    print len(ok_filter)
    
    ok_pvals = sp.copy(marg_Y_p_vals)
    ok_pvals[ok_filter]=adj_Y_p_vals[ok_filter]
    
    increased_power_frac = sp.sum(ok_pvals<marg_Y_p_vals)/float(num_traits)
    

    # Try regressing the genetic effect on C out of C and then perform the regression:
    #    Frequentists approach
    adj_C_resid_betas_Y = sp.empty((num_traits,2))
    for i in range(num_traits):
        X = sp.vstack([snps[i],C_SNP_resid[i]])
        XX = sp.mat(sp.dot(X,X.T))
        XX_inv = linalg.inv(XX)
        adj_C_resid_betas_Y[i] =sp.dot(XX_inv, sp.dot(X,Y[i].T).T)
#     print adj_betas_Y.shape
    print 'Adj association on Y C-g:'
    print sp.mean(adj_betas_Y,0)
#     print adj_betas_Y
    adj_C_resid_betas_Y_genot = adj_C_resid_betas_Y[:,0]
    adj_C_resid_betas_Y_C = adj_C_resid_betas_Y[:,1]
    C_effect = (C_SNP_resid.T*adj_C_resid_betas_Y_C).T
    
    res0 = Y - C_effect
    rss0 = sp.sum(res0**2,1)
    res1 = Y-(snps.T*adj_C_resid_betas_Y_genot).T - C_effect
#     print res.shape
    rss1 = sp.sum(res1**2,1)
    f_stats = (rss0 - rss1)*(n-2)/rss1
    adj_C_resid_Y_p_vals = stats.f.sf(f_stats, 1, n - 2)
    mean_adj_C_resid_Y_p_val = sp.mean(adj_C_resid_Y_p_vals)
    print 'Mean Adjust Y (C-g) p-value :', mean_adj_C_resid_Y_p_val
    print 'Mean Adjust Y (C-g) bias:',sp.mean(beta_y -adj_C_resid_betas_Y_genot)
    
    if beta_y==0:
        adj_C_resid_Y_fdr = sp.sum(adj_C_resid_Y_p_vals<alpha_thres)/float(num_traits)
        adj_C_resid_Y_power = 0
    else:
        adj_C_resid_Y_fdr = 0
        adj_C_resid_Y_power = sp.sum(adj_C_resid_Y_p_vals<alpha_thres)/float(num_traits)
    print 'Adj Y (C-g) power:', adj_C_resid_Y_power    
    print 'Adj Y (C-g) FDR:', adj_C_resid_Y_fdr    
        
    #Bayesian approach
    
    #1. Calculate Bayes C_snp_resid.
    p = 1 #Fraction of causal markers
    beta_c2 = beta_c**2
    h2 = (beta_c2)/(1+beta_c2)
    if beta_c==0:
        post_mean_direct_betas_C = 0
    else:
        post_mean_direct_betas_C = (1/(1+1/(n*h2)))*direct_betas_C
    
    bayes_C_SNP_resid = C-(snps.T*post_mean_direct_betas_C).T
    
    #2. Use to calculate the statistic, the usual way.
    bayes_adj_C_resid_betas_Y = sp.empty((num_traits,2))
    for i in range(num_traits):
        X = sp.vstack([snps[i],bayes_C_SNP_resid[i]])
        XX = sp.mat(sp.dot(X,X.T))
        XX_inv = linalg.inv(XX)
        bayes_adj_C_resid_betas_Y[i] =sp.dot(XX_inv, sp.dot(X,Y[i].T).T)
#     print adj_betas_Y.shape
    print 'Adj association on Y:'
    print sp.mean(adj_betas_Y,0)
#     print adj_betas_Y
    bayes_adj_C_resid_betas_Y_genot = bayes_adj_C_resid_betas_Y[:,0]
    bayes_adj_C_resid_betas_Y_C = bayes_adj_C_resid_betas_Y[:,1]
    bayes_C_effect = (bayes_C_SNP_resid.T*bayes_adj_C_resid_betas_Y_C).T
    
    res0 = Y - bayes_C_effect
    rss0 = sp.sum(res0**2,1)
    res1 = Y-(snps.T*bayes_adj_C_resid_betas_Y_genot).T - bayes_C_effect
#     print res.shape
    rss1 = sp.sum(res1**2,1)
    f_stats = (rss0 - rss1)*(n-2)/rss1
    bayes_adj_C_resid_Y_p_vals = stats.f.sf(f_stats, 1, n - 2)
    mean_bayes_adj_C_resid_Y_p_val = sp.mean(bayes_adj_C_resid_Y_p_vals)
    print 'Bayesian mean adjust Y (C-g) p-value :', mean_bayes_adj_C_resid_Y_p_val
    print 'Bayesian mean adjust Y (C-g) bias:',sp.mean(beta_y-adj_C_resid_betas_Y_genot)
    
    if beta_y==0:
        bayes_adj_C_resid_Y_fdr = sp.sum(bayes_adj_C_resid_Y_p_vals<alpha_thres)/float(num_traits)
        bayes_adj_C_resid_Y_power = 0
    else:
        bayes_adj_C_resid_Y_fdr = 0
        bayes_adj_C_resid_Y_power = sp.sum(bayes_adj_C_resid_Y_p_vals<alpha_thres)/float(num_traits)
    print 'Bayesian adj Y (C-g) power:', bayes_adj_C_resid_Y_power    
    print 'Bayesian adj Y (C-g) FDR:', bayes_adj_C_resid_Y_fdr    
   
    
    
    mean_ok_pvals= sp.mean(ok_pvals)
    print 'Mean OK p-value:', mean_ok_pvals
    print 'Mean bias p-value:', mean_bias_p_val
    print 'Increased power frac.:',increased_power_frac
    ret_dict =  {'mean_ok_pvals':mean_ok_pvals,'mean_adj_pvals':mean_adj_pvals, 
                'mean_marg_C_pvals':mean_marg_C_pvals, 'mean_marg_Y_pvals':mean_marg_Y_pvals,
                'increased_power_frac':increased_power_frac, 'mean_bias_p_vals':mean_bias_p_val,
                'marg_Y_power': marg_Y_power, 'marg_Y_fdr': marg_Y_fdr,
                'marg_C_power': marg_C_power, 'marg_C_fdr': marg_C_fdr,
                'adj_Y_power': adj_Y_power, 'adj_Y_fdr': adj_Y_fdr,
                'bias_test_power': bias_test_power, 'bias_test_fdr': bias_test_fdr,
                'adj_C_resid_Y_power':adj_C_resid_Y_power, 'adj_C_resid_Y_fdr':adj_C_resid_Y_fdr,
                'bayes_adj_C_resid_Y_power':bayes_adj_C_resid_Y_power, 'bayes_adj_C_resid_Y_fdr':bayes_adj_C_resid_Y_fdr,
                'adj_Y_bias':adj_Y_bias,
                }
    
    return ret_dict



def test_null_fig2(n=100, num_traits=1000, corr=0.8, beta_c=0, beta_y=0, alpha_thres=0.05):
    """
    # 1. Generate test genotypes.
    # 2. Generate GE1 and GE2, w correlation.
    # 3. Construct phenotypes.
    # 4. Check direct association signal.
    # 5. Check covariate adjusted association signal.
    """
    snps = genotypes.simulate_common_genotypes(n=n,m=num_traits)
    #Normalize genotypes
    snps = snps.T
    snps = (snps - sp.mean(snps, 0)) / sp.std(snps, 0)
    snps = snps.T     
       
    V = sp.kron(sp.array([[1,corr],[corr,1]]),sp.eye(n))
    #print V
    GE = sp.random.multivariate_normal(sp.zeros(n*2), cov=V, size=num_traits)
    #print GE.shape
    GE1 = GE[:,:n]
    GE1_norm = GE1.T
    GE1_norm = (GE1_norm - sp.mean(GE1_norm, 0)) / sp.std(GE1_norm, 0)
    GE1_norm = GE1_norm.T
    
    GE2 = GE[:,n:]
    GE2_norm = GE2.T
    GE2_norm = (GE2_norm - sp.mean(GE2_norm, 0)) / sp.std(GE2_norm, 0)
    GE2_norm = GE2_norm.T

    gs = snps
    C = GE1_norm + beta_c*gs
#     print C[0], GE1_norm[0],gs[0]
#     print sp.var(C[0]), sp.var(GE1_norm[0]), sp.var(gs[0])
    #print C.shape, GE1.shape, gs.shape
    Y = GE2_norm + beta_y*gs

#     #Normalize Y's and C's
#     C_norm = C.T
#     C_norm_factor = sp.std(C_norm, 0)
#     C_norm = (C_norm - sp.mean(C_norm, 0))/ C_norm_factor
#     C = C_norm.T
#     
#     Y_norm = Y.T
#     Y_norm = (Y_norm - sp.mean(Y_norm, 0)) / sp.std(Y_norm, 0)
#     Y = Y_norm.T
    
    adj_betas_Y = sp.empty((num_traits,2))
    for i in range(num_traits):
        X = sp.vstack([snps[i],C[i]])
        XX = sp.mat(sp.dot(X,X.T))
        XX_inv = linalg.inv(XX)
        adj_betas_Y[i] =sp.dot(XX_inv, sp.dot(X,Y[i].T).T)
#     print adj_betas_Y.shape
    print 'Adj association on Y:'
    print sp.mean(adj_betas_Y,0)
#     print adj_betas_Y
    adj_betas_Y_genot = adj_betas_Y[:,0]

    res0 = Y - (C.T*adj_betas_Y[:,1]).T
    rss0 = sp.sum(res0**2,1)
    res1 = Y-(snps.T*adj_betas_Y_genot).T - (C.T*adj_betas_Y[:,1]).T
#     print res.shape
    rss1 = sp.sum(res1**2,1)
    f_stats = (rss0 - rss1)*(n-2)/rss1
    adj_Y_p_vals = stats.f.sf(f_stats, 1, n - 2)
    mean_adj_pvals = sp.mean(adj_Y_p_vals)
    print 'Mean Adjust Y p-value:', mean_adj_pvals
    
    if beta_y==0:
        adj_Y_fdr = sp.sum(adj_Y_p_vals<alpha_thres)/float(num_traits)
        adj_Y_power = 0
    else:
        adj_Y_fdr = 0
        adj_Y_power = sp.sum(adj_Y_p_vals<alpha_thres)/float(num_traits)
    print 'Adj Y power:', adj_Y_power    
    print 'Adj Y FDR:', adj_Y_fdr    

    adj_Y_bias = sp.mean(beta_y-adj_betas_Y_genot)
    
    ret_dict =  {'mean_adj_pvals':mean_adj_pvals, 
                'adj_Y_power': adj_Y_power, 'adj_Y_fdr': adj_Y_fdr,
                'adj_Y_bias':adj_Y_bias,}
    
    return ret_dict



def generate_c_regression_plots(plot_file_prefix='/Users/bjarnivilhjalmsson/data/tmp/collider_plot_FDRs_y',
                  n=200,num_traits=1000, beta_y=0.0):
    corrs =[-0.95,-0.8,-0.65,-0.5,-0.35,-0.2,-0.05,
            0.05,0.2,0.35,0.5,0.65,0.8,0.95]
    pylab.Figure()
    for color, beta_c in zip(['r','g','b'], [0.0,0.02,0.04]):
        marg_Y_powers = []
        marg_C_powers = []
        adj_Y_powers = []
        bias_test_powers = []
        mean_ok_pvals = []
        mean_adj_pvals = []
        mean_bias_p_vals = []
        mean_marg_C_pvals = []
        mean_marg_Y_pvals = []
        increased_power_fracs = [] 
        adj_Y_fdrs = []
        adj_C_resid_Y_fdrs = []
        bayes_adj_C_resid_Y_fdrs = []
        adj_C_resid_Y_powers = []
        bayes_adj_C_resid_Y_powers = []
        for corr in corrs:
            d = test_null(n=n, num_traits=num_traits, corr=corr, beta_c=beta_c, beta_y=beta_y)
            marg_Y_powers.append(d['marg_Y_power'])
            marg_C_powers.append(d['marg_C_power'])
            adj_Y_powers.append(d['adj_Y_power'])
            bias_test_powers.append(d['bias_test_power'])
            mean_ok_pvals.append(d['mean_ok_pvals'])
            mean_adj_pvals.append(d['mean_adj_pvals'])
            mean_bias_p_vals.append(d['mean_bias_p_vals'])
            mean_marg_C_pvals.append(d['mean_marg_C_pvals'])
            mean_marg_Y_pvals.append(d['mean_marg_Y_pvals'])
            increased_power_fracs.append(d['increased_power_frac'])
            adj_Y_fdrs.append(d['adj_Y_fdr'])
            adj_C_resid_Y_fdrs.append(d['adj_C_resid_Y_fdr'])
            bayes_adj_C_resid_Y_fdrs.append(d['bayes_adj_C_resid_Y_fdr'])
            adj_C_resid_Y_powers.append(d['adj_C_resid_Y_power'])
            bayes_adj_C_resid_Y_powers.append(d['bayes_adj_C_resid_Y_power'])
        beta_str = 'beta_c=%0.2f'%beta_c
        if beta_y==0.0:
            pylab.plot(corrs,adj_Y_fdrs, label=beta_str, color=color, linestyle='-',alpha=0.6)
            pylab.plot(corrs,adj_C_resid_Y_fdrs, color=color, linestyle='-.', alpha=0.6)
            pylab.plot(corrs,bayes_adj_C_resid_Y_fdrs, color=color, linestyle=':', alpha=0.6)
        else:
            pylab.plot(corrs,adj_Y_powers, label=beta_str, color=color, linestyle='-',alpha=0.6)
            pylab.plot(corrs,adj_C_resid_Y_powers, color=color, linestyle='-.', alpha=0.6)
            pylab.plot(corrs,bayes_adj_C_resid_Y_powers, color=color, linestyle=':', alpha=0.6)
    pylab.axis([-1,1,0,1])
    if beta_y==0.0:
        pylab.plot([-2,-2],[-2,-2], label='Adj. Y FDR', color='k', linestyle='-',alpha=0.8)
        pylab.plot([-2,-2],[-2,-2], label='Adj. Y w C resid FDR', color='k', linestyle='-.',alpha=0.8)
        pylab.plot([-2,-2],[-2,-2], label='Adj. Y w Bayes C resid FDR', color='k', linestyle=':',alpha=0.8)
    else:
        pylab.plot([-2,-2],[-2,-2], label='Adj. Y power', color='k', linestyle='-',alpha=0.8)
        pylab.plot([-2,-2],[-2,-2], label='Adj. Y w C resid power', color='k', linestyle='-.',alpha=0.8)
        pylab.plot([-2,-2],[-2,-2], label='Adj. Y w Bayes C resid power', color='k', linestyle=':',alpha=0.8)
    pylab.legend()
    pylab.savefig(plot_file_prefix+'_n%d_%0.02f.png'%(n,beta_y))
    pylab.clf()



            

def generate_plot(plot_file_prefix='/Users/bjarnivilhjalmsson/data/tmp/power_plot_beta_y',
                  n=1000,num_traits=5000, beta_y=0.01):
    corrs =[-0.95,-0.85,-0.75,-0.65,-0.55,-0.45,-0.35,-0.25,-0.15,-0.05,
            0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
#    corrs =[-0.95,-0.8,-0.65,-0.5,-0.35,-0.2,-0.05,
#            0.05,0.2,0.35,0.5,0.65,0.8,0.95]
    pylab.Figure()
    colors = ['r','g','b','m']
    for color, beta_c in zip(colors, [0.0,0.02,0.04,0.08]):
        marg_Y_powers = []
        marg_C_powers = []
        adj_Y_powers = []
        bias_test_powers = []
        mean_ok_pvals = []
        mean_adj_pvals = []
        mean_bias_p_vals = []
        mean_marg_C_pvals = []
        mean_marg_Y_pvals = []
        increased_power_fracs = [] 
        for corr in corrs:
            d = test_null(n=n, num_traits=num_traits, corr=corr, beta_c=beta_c, beta_y=beta_y)
            marg_Y_powers.append(d['marg_Y_power'])
            marg_C_powers.append(d['marg_C_power'])
            adj_Y_powers.append(d['adj_Y_power'])
            bias_test_powers.append(d['bias_test_power'])
            mean_ok_pvals.append(d['mean_ok_pvals'])
            mean_adj_pvals.append(d['mean_adj_pvals'])
            mean_bias_p_vals.append(d['mean_bias_p_vals'])
            mean_marg_C_pvals.append(d['mean_marg_C_pvals'])
            mean_marg_Y_pvals.append(d['mean_marg_Y_pvals'])
            increased_power_fracs.append(d['increased_power_frac'])
#         beta_str = 'beta_c=%0.2f'%beta_c
        beta_str = '%0.2f'%beta_c
#         pylab.plot(corrs,marg_C_powers, label=beta_str, color=color, linestyle='-', linewidth=2,alpha=0.6)
#         pylab.plot(corrs,mean_adj_pvals, label=beta_str, color=color, linestyle='-',alpha=0.6)
#         pylab.plot(corrs,mean_ok_pvals, label='Avg. cons. p-value', alpha=0.6)
        pylab.plot(corrs,adj_Y_powers, color=color, label=r'$\beta_C=%s$'%(beta_str), linestyle='-', linewidth=2, alpha=0.4)
        #pylab.plot(corrs,bias_test_powers, color=color, linestyle=':', alpha=0.6)
#         pylab.plot(corrs,mean_bias_p_vals, color=color, linestyle='-.', alpha=0.6)
#         pylab.plot(corrs,increased_power_fracs, label='Cons. incr. power rate', alpha=0.6)
    pylab.axis([-1,1,0,1])
    pylab.ylabel('Power')
    pylab.xlabel(r'Correlation ($\rho_{Y,C}$)')
#     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. power', color='k', linestyle='-',alpha=0.8)
    #pylab.plot([-2,-2],[-2,-2], label='Avg. marg C power', color='k', linestyle='-.',alpha=0.8)
    #pylab.plot([-2,-2],[-2,-2], label='Bias testg power', color='k', linestyle=':',alpha=0.8)
#     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. p-value', color='k', linestyle='-',alpha=0.8)
#     pylab.plot([-2,-2],[-2,-2], label='Avg. bias p-value', color='k', linestyle='-.',alpha=0.8)
        
    pylab.legend(loc=9)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6,3.3) 
    pylab.tight_layout()
    pylab.savefig(plot_file_prefix+'%0.02f.pdf'%beta_y, format='pdf')
    pylab.clf()
        
    
def generate_fig2a(plot_file_prefix='/Users/bjarnivilhjalmsson/data/tmp/fdr_plot',
                  n=2000,num_traits=5000, beta_y=0.0):
    corrs =[-0.95,-0.85,-0.75,-0.65,-0.55,-0.45,-0.35,-0.25,-0.15,-0.05,
            0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
#     corrs =[-0.95,-0.8,-0.65,-0.5,-0.35,-0.2,-0.05,
#             0.05,0.2,0.35,0.5,0.65,0.8,0.95]
    pylab.Figure()
    colors = ['r','g','b','m']
    for color, beta_c in zip(colors, [0.0,0.005,0.01,0.02]):
        marg_Y_powers = []
        marg_C_powers = []
        adj_Y_powers = []
        bias_test_powers = []
        mean_ok_pvals = []
        mean_adj_pvals = []
        mean_bias_p_vals = []
        mean_marg_C_pvals = []
        mean_marg_Y_pvals = []
        increased_power_fracs = [] 
        adj_Y_fdrs = []

        for corr in corrs:
            d = test_null(n=n, num_traits=num_traits, corr=corr, beta_c=beta_c, beta_y=beta_y)
            marg_Y_powers.append(d['marg_Y_power'])
            marg_C_powers.append(d['marg_C_power'])
            adj_Y_powers.append(d['adj_Y_power'])
            bias_test_powers.append(d['bias_test_power'])
            mean_ok_pvals.append(d['mean_ok_pvals'])
            mean_adj_pvals.append(d['mean_adj_pvals'])
            mean_bias_p_vals.append(d['mean_bias_p_vals'])
            mean_marg_C_pvals.append(d['mean_marg_C_pvals'])
            mean_marg_Y_pvals.append(d['mean_marg_Y_pvals'])
            increased_power_fracs.append(d['increased_power_frac'])
            adj_Y_fdrs.append(d['adj_Y_fdr'])

#         beta_str = 'beta_c=%0.2f'%beta_c
        beta_str = '%0.3f'%beta_c
#         pylab.plot(corrs,marg_C_powers, label=beta_str, color=color, linestyle='-', linewidth=2,alpha=0.6)
#         pylab.plot(corrs,mean_adj_pvals, label=beta_str, color=color, linestyle='-',alpha=0.6)
#         pylab.plot(corrs,mean_ok_pvals, label='Avg. cons. p-value', alpha=0.6)
        pylab.plot(corrs,adj_Y_fdrs, color=color, label=r'$\beta_C=%s$'%(beta_str), linestyle='-', linewidth=2, alpha=0.4)
        #pylab.plot(corrs,bias_test_powers, color=color, linestyle=':', alpha=0.6)
#         pylab.plot(corrs,mean_bias_p_vals, color=color, linestyle='-.', alpha=0.6)
#         pylab.plot(corrs,increased_power_fracs, label='Cons. incr. power rate', alpha=0.6)
    pylab.axis([-1,1,-0.05,1])
    pylab.ylabel('False discovery rate')
    #pylab.xlabel(r'Correlation ($\rho_{Y,C}$)')
    pylab.text(-0.88,0.88,'A)',fontsize=14, fontweight='bold')
#     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. power', color='k', linestyle='-',alpha=0.8)
    #pylab.plot([-2,-2],[-2,-2], label='Avg. marg C power', color='k', linestyle='-.',alpha=0.8)
    #pylab.plot([-2,-2],[-2,-2], label='Bias testg power', color='k', linestyle=':',alpha=0.8)
#     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. p-value', color='k', linestyle='-',alpha=0.8)
#     pylab.plot([-2,-2],[-2,-2], label='Avg. bias p-value', color='k', linestyle='-.',alpha=0.8)
        
    pylab.legend(loc=9)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6,3.3)
    pylab.tight_layout()
    pylab.savefig(plot_file_prefix+'%0.02f.pdf'%beta_y, format='pdf')
    pylab.clf()
    


def generate_fig2b(plot_file_prefix='/Users/bjarnivilhjalmsson/data/tmp/power_plot_beta_c',
                  n=2000,num_traits=5000, beta_c=0.01):
    corrs =[-0.95,-0.85,-0.75,-0.65,-0.55,-0.45,-0.35,-0.25,-0.15,-0.05,
            0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
#     corrs =[-0.95,-0.8,-0.65,-0.5,-0.35,-0.2,-0.05,
#             0.05,0.2,0.35,0.5,0.65,0.8,0.95]
    pylab.Figure()
    colors = ['r','g','b','m']
    for color, beta_y in zip(colors, [0.0,0.005,0.01,0.02]):
        marg_Y_powers = []
        marg_C_powers = []
        adj_Y_powers = []
        bias_test_powers = []
        mean_ok_pvals = []
        mean_adj_pvals = []
        mean_bias_p_vals = []
        mean_marg_C_pvals = []
        mean_marg_Y_pvals = []
        increased_power_fracs = [] 
        for corr in corrs:
            d = test_null(n=n, num_traits=num_traits, corr=corr, beta_c=beta_c, beta_y=beta_y)
            marg_Y_powers.append(d['marg_Y_power'])
            marg_C_powers.append(d['marg_C_power'])
            adj_Y_powers.append(d['adj_Y_power'])
            bias_test_powers.append(d['bias_test_power'])
            mean_ok_pvals.append(d['mean_ok_pvals'])
            mean_adj_pvals.append(d['mean_adj_pvals'])
            mean_bias_p_vals.append(d['mean_bias_p_vals'])
            mean_marg_C_pvals.append(d['mean_marg_C_pvals'])
            mean_marg_Y_pvals.append(d['mean_marg_Y_pvals'])
            increased_power_fracs.append(d['increased_power_frac'])
#         beta_str = 'beta_c=%0.2f'%beta_c
        beta_str = '%0.3f'%beta_y
#         pylab.plot(corrs,marg_C_powers, label=beta_str, color=color, linestyle='-', linewidth=2,alpha=0.6)
#         pylab.plot(corrs,mean_adj_pvals, label=beta_str, color=color, linestyle='-',alpha=0.6)
#         pylab.plot(corrs,mean_ok_pvals, label='Avg. cons. p-value', alpha=0.6)
        pylab.plot(corrs,adj_Y_powers, color=color, label=r'$\beta_Y=%s$'%(beta_str), linestyle='-', linewidth=2, alpha=0.4)
        #pylab.plot(corrs,bias_test_powers, color=color, linestyle=':', alpha=0.6)
#         pylab.plot(corrs,mean_bias_p_vals, color=color, linestyle='-.', alpha=0.6)
#         pylab.plot(corrs,increased_power_fracs, label='Cons. incr. power rate', alpha=0.6)
    pylab.axis([-1,1,-0.05,1])
    pylab.ylabel('Power')
    #pylab.xlabel(r'Correlation ($\rho_{Y,C}$)')
    pylab.text(-0.88,0.88,'B)',fontsize=14, fontweight='bold')
#     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. power', color='k', linestyle='-',alpha=0.8)
    #pylab.plot([-2,-2],[-2,-2], label='Avg. marg C power', color='k', linestyle='-.',alpha=0.8)
    #pylab.plot([-2,-2],[-2,-2], label='Bias testg power', color='k', linestyle=':',alpha=0.8)
#     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. p-value', color='k', linestyle='-',alpha=0.8)
#     pylab.plot([-2,-2],[-2,-2], label='Avg. bias p-value', color='k', linestyle='-.',alpha=0.8)
        
    pylab.legend(loc=9)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6,3.3) 
    pylab.tight_layout()
    pylab.savefig(plot_file_prefix+'%0.02f.pdf'%beta_c, format='pdf')
    pylab.clf()
        
        
        

def generate_fig2(plot_file_prefix='/Users/bjarnivilhjalmsson/data/tmp/plot_fig2',
                  n=2000,num_traits=10000, beta_y=0.0):
    corrs =[-0.95,-0.85,-0.75,-0.65,-0.55,-0.45,-0.35,-0.25,-0.15,-0.05,
            0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
#     corrs =[-0.95,-0.8,-0.65,-0.5,-0.35,-0.2,-0.05,
#             0.05,0.2,0.35,0.5,0.65,0.8,0.95]
    f = plt.figure(figsize=(7, 7))
    ax1 = f.add_axes([0.12, 0.08 + 0.46, 0.85, 0.43 ])
    ax2 = f.add_axes([0.12, 0.08, 0.85, 0.43 ])
    colors = ['y','g','b','m','r']
    for color, beta_c in zip(colors, [0.0,0.005,0.01,0.02,0.03]):
        adj_Y_bias = []
        exp_Y_bias = []
        adj_Y_fdrs = []
        for corr in corrs:
            d = test_null_fig2(n=n, num_traits=num_traits, corr=corr, beta_c=beta_c, beta_y=beta_y)
            adj_Y_fdrs.append(d['adj_Y_fdr'])
            adj_Y_bias.append(d['adj_Y_bias'])
            exp_Y_bias.append(beta_c*corr)
        beta_str = '%0.3f'%beta_c
        ax1.plot(corrs,adj_Y_fdrs, color=color, label=r'$\beta_C=%s$'%(beta_str), linestyle='-', linewidth=2, alpha=0.4)
        ax2.plot(corrs,adj_Y_bias, color=color, linestyle='-', linewidth=2, alpha=0.4)
        ax2.plot(corrs,exp_Y_bias, color=color, linestyle='--', linewidth=1, alpha=0.7)
    ax1.axis([-1,1,-0.05,1])
    ax1.set_ylabel('False discovery rate')
    #pylab.xlabel(r'Correlation ($\rho_{Y,C}$)')
    ax1.text(-0.88,0.88,'A)',fontsize=14, fontweight='bold')
    ax1.xaxis.set_ticklabels([])
    ax1.legend(loc=9)

    y_max = 0.03
    y_min = -0.03
    y_range = y_max-y_min
    ax2.text(-0.88,y_max-y_range*0.08,'B)',fontsize=14, fontweight='bold')
    ax2.axis([-1,1,y_min-y_range*0.05,y_max+y_range*0.05])
    ax2.set_ylabel(r'Average observed bias ($\beta_Y-\hat{\beta}_Y$)')
    ax2.set_xlabel(r'Correlation ($\rho_{Y,C}$)')

    pylab.savefig(plot_file_prefix+'.pdf', format='pdf')
    pylab.clf()
        


# 
# def generate_fig2a(plot_file_prefix='/Users/bjarnivilhjalmsson/data/tmp/fdr_plot',
#                   n=2000,num_traits=5000, beta_y=0.0):
#     corrs =[-0.95,-0.85,-0.75,-0.65,-0.55,-0.45,-0.35,-0.25,-0.15,-0.05,
#             0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
# #     corrs =[-0.95,-0.8,-0.65,-0.5,-0.35,-0.2,-0.05,
# #             0.05,0.2,0.35,0.5,0.65,0.8,0.95]
#     pylab.Figure()
#     colors = ['r','g','b','m']
#     for color, beta_c in zip(colors, [0.0,0.005,0.01,0.02]):
#         marg_Y_powers = []
#         marg_C_powers = []
#         adj_Y_powers = []
#         bias_test_powers = []
#         mean_ok_pvals = []
#         mean_adj_pvals = []
#         mean_bias_p_vals = []
#         mean_marg_C_pvals = []
#         mean_marg_Y_pvals = []
#         increased_power_fracs = [] 
#         adj_Y_fdrs = []
# 
#         for corr in corrs:
#             d = test_null(n=n, num_traits=num_traits, corr=corr, beta_c=beta_c, beta_y=beta_y)
#             marg_Y_powers.append(d['marg_Y_power'])
#             marg_C_powers.append(d['marg_C_power'])
#             adj_Y_powers.append(d['adj_Y_power'])
#             bias_test_powers.append(d['bias_test_power'])
#             mean_ok_pvals.append(d['mean_ok_pvals'])
#             mean_adj_pvals.append(d['mean_adj_pvals'])
#             mean_bias_p_vals.append(d['mean_bias_p_vals'])
#             mean_marg_C_pvals.append(d['mean_marg_C_pvals'])
#             mean_marg_Y_pvals.append(d['mean_marg_Y_pvals'])
#             increased_power_fracs.append(d['increased_power_frac'])
#             adj_Y_fdrs.append(d['adj_Y_fdr'])
# 
# #         beta_str = 'beta_c=%0.2f'%beta_c
#         beta_str = '%0.3f'%beta_c
# #         pylab.plot(corrs,marg_C_powers, label=beta_str, color=color, linestyle='-', linewidth=2,alpha=0.6)
# #         pylab.plot(corrs,mean_adj_pvals, label=beta_str, color=color, linestyle='-',alpha=0.6)
# #         pylab.plot(corrs,mean_ok_pvals, label='Avg. cons. p-value', alpha=0.6)
#         pylab.plot(corrs,adj_Y_fdrs, color=color, label=r'$\beta_C=%s$'%(beta_str), linestyle='-', linewidth=2, alpha=0.4)
#         #pylab.plot(corrs,bias_test_powers, color=color, linestyle=':', alpha=0.6)
# #         pylab.plot(corrs,mean_bias_p_vals, color=color, linestyle='-.', alpha=0.6)
# #         pylab.plot(corrs,increased_power_fracs, label='Cons. incr. power rate', alpha=0.6)
#     pylab.axis([-1,1,-0.05,1])
#     pylab.ylabel('False discovery rate')
#     #pylab.xlabel(r'Correlation ($\rho_{Y,C}$)')
#     pylab.text(-0.88,0.88,'A)',fontsize=14, fontweight='bold')
# #     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. power', color='k', linestyle='-',alpha=0.8)
#     #pylab.plot([-2,-2],[-2,-2], label='Avg. marg C power', color='k', linestyle='-.',alpha=0.8)
#     #pylab.plot([-2,-2],[-2,-2], label='Bias testg power', color='k', linestyle=':',alpha=0.8)
# #     pylab.plot([-2,-2],[-2,-2], label='Avg. adj. p-value', color='k', linestyle='-',alpha=0.8)
# #     pylab.plot([-2,-2],[-2,-2], label='Avg. bias p-value', color='k', linestyle='-.',alpha=0.8)
#         
#     pylab.legend(loc=9)
#     fig = matplotlib.pyplot.gcf()
#     fig.set_size_inches(6,3.3)
#     pylab.tight_layout()
#     pylab.savefig(plot_file_prefix+'%0.02f.pdf'%beta_y, format='pdf')
#     pylab.clf()
#     

    