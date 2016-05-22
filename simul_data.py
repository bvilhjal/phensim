"""
Methods for simulating GWAS datasets
"""

import scipy as sp
from scipy import stats
import sys
import os
import h5py
import genotypes as gt
import phenotypes as pt
import plinkio
from plinkio import plinkfile
import itertools as it



def write_fake_plink_file(snps, plink_file_prefix, positions, cc_phenotypes=None, phenotypes=None):
    """
    Writes out a PLINK formatted file
    """
    num_snps,num_indivs = snps.shape

    #Filling in the blanks
    print 'Creating fake SNP IDs.'
    sids = ['sid_%d'%i for i in range(num_snps)]
        
    print 'Creating fake nucleotides.'
    nts = [['A','G']]*num_snps

    if os.path.isfile(plink_file_prefix+'.bed'):
        print "PLINK files already exist."
    else:
        print 'Generating PLINK formatted files'
        print 'Creating fake chromosome names.'
        chromosomes = [1]*num_snps
        
        print 'Writing PLINK bed file: %s'%plink_file_prefix
        #Create samples
        samples = []
        for i in range(num_indivs):
            iid = 'i%d'%i
            fid = iid
            sex = 0 # 
            if cc_phenotypes!=None:
                phenotype = 0
                affection = cc_phenotypes[i] # 0: control, 1: case, -9: missing, any other value: continuous phenotype
            elif phenotypes!=None:
                affection = 2  
                phenotype = phenotypes[i]
            else:
                #Phenotype is missing.
                phenotype = 0
                affection = 2
            indiv = plinkfile.Sample(fid, iid, iid, iid, sex, affection, phenotype)
            samples.append(indiv)
        
        #Creating plinkfile
        wpf = plinkfile.WritablePlinkFile(plink_file_prefix, samples)
        
        #Now iterate over SNPs.
        for name, row, bp_position, chromosome, nt in it.izip(sids, snps, positions, chromosomes, nts):
            position = 0 #map position
            allele1 = nt[0]
            allele2 = nt[1]
            locus = plinkfile.Locus(chromosome, name, position, bp_position, allele1, allele2)
            wpf.write_row(locus, row) 
            del locus   
            del row
        wpf.close()
        del wpf
        del samples
    


def simulate_plink_train_test_datasets(num_traits=1, n_sample=1000, p=0.001, m=10000, h2=0.1, adj_r2=0.9, m_ld_chunk_size=100, 
                            effect_prior='gaussian', out_prefix='/Users/bjarnivilhjalmsson/data/tmp/LDpred_data'):
    
    #First simulate SNPs (w LD)
    snps = gt.simulate_genotypes_w_ld(n_sample=n_sample, m=m, conseq_r2=adj_r2, m_ld_chunk_size=m_ld_chunk_size, diploid=True, verbose=True)
    positions = range(m)

    print snps[0], snps[100], snps[200]

    #Simulate traits
    phen_dict = pt.simulate_traits_w_snps(snps, num_traits=num_traits, p=p, m=m, h2=h2, effect_prior=effect_prior, verbose=True, 
                                          liability_thres=None)
    
    #Partition into training and test data
    part_i = int(n_sample/5.0)
    train_snps = snps[:,part_i:]
    test_snps = snps[:,:part_i]
    
    
    #Write out Plink files
    for t_i in range(num_traits):
        train_plink_prefix = '%s_p%0.3f_train_%d'%(out_prefix, p, t_i)
        test_plink_prefix = '%s_p%0.3f_test_%d'%(out_prefix, p, t_i)
        
        
        train_phens = phen_dict['phenotypes'][t_i][part_i:]
        test_phens = phen_dict['phenotypes'][t_i][:part_i]
        write_fake_plink_file(train_snps, train_plink_prefix, positions, phenotypes=train_phens)
        print 'Done w Training file'
        write_fake_plink_file(test_snps, test_plink_prefix, positions, phenotypes=test_phens)
        print 'Done w Testing file'
    

    #Conduct GWAS, and write out results.
    print 'Normalizing genotypes'
    snps_stds = sp.std(train_snps,axis=1)
    snps_means = sp.mean(train_snps,axis=1)
    snps_stds.shape = (len(snps_stds),1)
    snps_means.shape = (len(snps_means),1)
    snps = (train_snps - snps_means)/snps_stds
            
    for t_i in range(num_traits):
        ss_filename = '%s_p%0.3f_ss_%d.txt'%(out_prefix, p, t_i)
        train_phens = phen_dict['phenotypes'][t_i][part_i:]
        #Normalize phenotypes
        n_training = len(train_phens)
        beta_hats = sp.dot(snps, train_phens) / n_training
        b2s = beta_hats ** 2
        f_stats = (n_training - 2) * b2s / (1 - b2s)
        pvals = stats.f.sf(f_stats, 1, n_training - 2)
        print 'Median p-value is %0.3f, and mean p-value is %0.3f'%(sp.median(pvals),sp.mean(pvals))
        
        """
        chr     pos     ref     alt     reffrq  info    rs       pval    effalt
        chr1    1020428 C       T       0.85083 0.98732 rs6687776    0.0587  -0.0100048507289348
        chr1    1020496 G       A       0.85073 0.98751 rs6678318    0.1287  -0.00826075392985992
        """
        with open(ss_filename,'w') as f:
            f.write('chr     pos     ref     alt     reffrq  info    rs       pval    effalt\n')
            i = 0
            for eff, pval in it.izip(beta_hats,pvals):
                f.write('chr1    %d    A    G    0.5    1    sid_%d    %0.6e    %0.6e\n'%(i,i,pval,eff))
                i += 1
            


if __name__=='__main__':
    pass
#     simulate_plink_datasets()
    
    
    