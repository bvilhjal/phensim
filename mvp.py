"""
Rare variant permutations.
"""
import scipy as sp
import sys
#sys.path.append('./../atgwas/src/')
sys.path.append('./../mixmogam/')
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt   
import linear_models as lm
import kinship
import analyze_gwas_results as agr
import phenotypes
import genotypes
import os


def _test_():
    singleton_snps = genotypes.simulate_k_tons(n=500, m=1000)
    doubleton_snps = genotypes.simulate_k_tons(k=2, n=500, m=1000)
    common_snps = genotypes.simulate_common_genotypes(500, 1000) 
    
    snps = sp.vstack([common_snps, singleton_snps, doubleton_snps])
    print snps
    snps = snps.T
    snps = (snps - sp.mean(snps, 0)) / sp.std(snps, 0)
    snps = snps.T
    print snps, snps.shape
    file_prefix = os.environ['HOME'] + '/tmp/test'
    phen_list = phenotypes.simulate_traits_w_snps_to_hdf5(snps, hdf5_file_prefix=file_prefix,
                                           num_traits=30, p=0.1)
    
    singletons_thres = []
    doubletons_thres = []
    common_thres = []
    for i, y in enumerate(phen_list['phenotypes']):
        
        K = kinship.calc_ibd_kinship(snps)
        K = kinship.scale_k(K)
        lmm = lm.LinearMixedModel(y)
        lmm.add_random_effect(K)
        r1 = lmm.get_REML()
        print 'pseudo_heritability:', r1['pseudo_heritability']

        ex_res = lm.emmax(snps, y, K)
        plt.figure()
        plt.hist(y, 50)
        plt.savefig('%s_%d_phen.png' % (file_prefix, i))
        plt.clf()
        
        
        agr.plot_simple_qqplots_pvals('%s_%d' % (file_prefix, i),
                                      [ex_res['ps'][:1000], ex_res['ps'][1000:2000], ex_res['ps'][2000:]],
                                      result_labels=['Common SNPs', 'Singletons', 'Doubletons'],
                                      line_colors=['b', 'r', 'y'],
                                      num_dots=200, max_neg_log_val=3)
        
        # Cholesky permutations..
        res = lm.emmax_perm_test(singleton_snps, y, K, num_perm=1000)
        print 1.0 / (20 * 1000.0), res['threshold_05']
        singletons_thres.append(res['threshold_05'][0])
        res = lm.emmax_perm_test(doubleton_snps, y, K, num_perm=1000)
        print 1.0 / (20 * 1000.0), res['threshold_05']
        doubletons_thres.append(res['threshold_05'][0])
        res = lm.emmax_perm_test(common_snps, y, K, num_perm=1000)
        print 1.0 / (20 * 1000.0), res['threshold_05']
        common_thres.append(res['threshold_05'][0])
        
        #ATT permutations (Implement)
        
        #PC permutations (Implement)
        

    print sp.mean(singletons_thres), sp.std(singletons_thres)
    print sp.mean(doubletons_thres), sp.std(doubletons_thres)
    print sp.mean(common_thres), sp.std(common_thres)
        

def _test_scz_():
    # Load Schizophrenia data
    
    singleton_snps = genotypes.simulate_k_tons(n=500, m=1000)
    doubleton_snps = genotypes.simulate_k_tons(k=2, n=500, m=1000)
    common_snps = genotypes.simulate_common_genotypes(500, 1000) 
    
    snps = sp.vstack([common_snps, singleton_snps, doubleton_snps])
    test_snps = sp.vstack([singleton_snps, doubleton_snps])
    print snps
    phen_list = phenotypes.simulate_traits(snps, hdf5_file_prefix='/home/bv25/tmp/test', num_traits=30, p=1.0)
    
    singletons_thres = []
    doubletons_thres = []
    common_thres = []
    for i, y in enumerate(phen_list):
        
        K = kinship.calc_ibd_kinship(snps)
        K = kinship.scale_k(K)
        lmm = lm.LinearMixedModel(y)
        lmm.add_random_effect(K)
        r1 = lmm.get_REML()
        print 'pseudo_heritability:', r1['pseudo_heritability']

        ex_res = lm.emmax(snps, y, K)
        plt.figure()
        plt.hist(y, 50)
        plt.savefig('/home/bv25/tmp/test_%d_phen.png' % i)
        plt.clf()
        agr.plot_simple_qqplots_pvals('/home/bv25/tmp/test_%d' % i,
                                      [ex_res['ps'][:1000], ex_res['ps'][1000:2000], ex_res['ps'][2000:]],
                                      result_labels=['Common SNPs', 'Singletons', 'Doubletons'],
                                      line_colors=['b', 'r', 'y'],
                                      num_dots=200, max_neg_log_val=3)
        
        # Now permutations..
        res = lm.emmax_perm_test(singleton_snps, y, K, num_perm=1000)
        print 1.0 / (20 * 1000.0), res['threshold_05']
        singletons_thres.append(res['threshold_05'][0])
        res = lm.emmax_perm_test(doubleton_snps, y, K, num_perm=1000)
        print 1.0 / (20 * 1000.0), res['threshold_05']
        doubletons_thres.append(res['threshold_05'][0])
        res = lm.emmax_perm_test(common_snps, y, K, num_perm=1000)
        print 1.0 / (20 * 1000.0), res['threshold_05']
        common_thres.append(res['threshold_05'][0])
    print sp.mean(singletons_thres), sp.std(singletons_thres)
    print sp.mean(doubletons_thres), sp.std(doubletons_thres)
    print sp.mean(common_thres), sp.std(common_thres)
        
