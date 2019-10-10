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
import json
import random

def write_fake_plink_file(snps, plink_file_prefix, chromosomes=None, 
                          snp_ids=None, nts = None, positions=None, 
                          cc_phenotypes=None, phenotypes=None):
    """
    Writes out a PLINK formatted file
    """
    num_snps,num_indivs = snps.shape

    #Filling in the blanks
    if snp_ids is None:
        print ('Creating fake SNP IDs.')
        snp_ids = ['sid_%d'%i for i in range(num_snps)]
        
    if nts is None:
        print ('Creating fake nucleotides.')
        nts = [['A','G']]*num_snps

    if os.path.isfile(plink_file_prefix+'.bed'):
        print ("PLINK files already exist.")
    else:
        print ('Generating PLINK formatted files')
        
        if chromosomes is None:
            print ('Creating fake chromosome names.')
            chromosomes = [1]*num_snps
        if positions is None:
            print ('Creating fake positions.')
            positions = range(num_snps)
        
        print ('Writing PLINK bed file: %s'%plink_file_prefix)
        #Create samples
        samples = []
        for i in range(num_indivs):
            iid = 'i%d'%i
            fid = iid
            sex = 0 # 
            if cc_phenotypes is not None:
                phenotype = 0
                affection = cc_phenotypes[i] # 0: control, 1: case, -9: missing, any other value: continuous phenotype
            elif phenotypes is not None:
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
        for name, row, bp_position, chromosome, nt in zip(snp_ids, snps, positions, chromosomes, nts):
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
    

def simulate_plink_train_test_datasets_simple(num_traits=1, h2=0.1, m=10000, n=15000, p=0.1, 
                                              adj_r2=0.9, m_ld_chunk_size=100, diploid = True,
                                              verbose=True,effect_prior='gaussian', 
                                              out_prefix='/Users/au507860/REPOS/ldpred/test_data/LDpred_data'):
    chrom_dict = {1:{'h2':h2, 'm':m, 'n':n, 'p':p, 'adj_r2':adj_r2, 'm_ld_chunk_size':m_ld_chunk_size}}
    simulate_plink_train_test_datasets(chrom_dict,num_traits=num_traits,diploid=diploid,verbose=verbose,
                                       effect_prior=effect_prior,out_prefix=out_prefix)

def simulate_plink_train_test_datasets_complex(num_traits=1, diploid = True,
                                              verbose=True,effect_prior='gaussian', 
                                              out_prefix='/Users/au507860/REPOS/ldpred/test_data/LDpred_data'):
    chrom_dict = {1:{'h2':0.05, 'm':5000, 'n':10000, 'p':0.01, 'adj_r2':0.9, 'm_ld_chunk_size':100},
                  2:{'h2':0.05, 'm':5000, 'n':10000, 'p':0.1, 'adj_r2':0.9, 'm_ld_chunk_size':100},
                  3:{'h2':0.05, 'm':5000, 'n':10000, 'p':1, 'adj_r2':0.9, 'm_ld_chunk_size':100}}
    simulate_plink_train_test_datasets(chrom_dict,num_traits=num_traits,diploid=diploid,verbose=verbose,
                                       effect_prior=effect_prior,out_prefix=out_prefix)


def simulate_plink_train_test_datasets(sim_dict,out_prefix='/Users/au507860/REPOS/ldpred/test_data/LDpred_data',verbose=True):
    
    
    """
    CHR     POS     SNP_ID    REF     ALT     REF_FRQ    PVAL    BETA    SE    N
    chr1    1020428    rs6687776    C       T       0.85083    0.0587  -0.0100048507289348    0.0100    8000
    chr1    1020496    rs6678318    G       A       0.85073    0.1287  -0.00826075392985992    0.0100    8000
    """
    
    chrom_dict = sim_dict['chrom_parameters']
    num_traits = sim_dict['num_traits']
    diploid = sim_dict['diploid']
    liability_thres = sim_dict['liability_thres']
    n_sample = sim_dict['n']
    test_frac = sim_dict['test_frac']
    train_n_fracs = sim_dict['train_n_fracs']
    frac_0_pval = sim_dict['frac_0_pval']
    
    #First simulate SNPs (w LD)
    all_train_snps = []
    all_test_snps = []
    all_snp_ids = []
    all_chr =[]
    all_pos =[]
    for chrom in chrom_dict:
        d = chrom_dict[chrom]
        m = d['m']
        adj_r2 = d['adj_r2']
        m_ld_chunk_size = d['m_ld_chunk_size']
        snps = gt.simulate_genotypes_w_ld(n_sample=n_sample, m=m, conseq_r2=adj_r2, 
                                               m_ld_chunk_size=m_ld_chunk_size, 
                                               diploid=diploid, verbose=verbose)

        #Partition into training and test data
        part_i = int(n_sample*test_frac)
        train_snps = snps[:,part_i:]
        test_snps = snps[:,:part_i]                
        
        snp_ids = ['SNP_ID_%s_%d'%(chrom,i+1) for i in range(m)]  
        positions = range(m)
        chromosomes = [chrom]*m
        
        #Add genotypes and other info to chrom dict..
        d['snps'] = snps
        d['train_snps'] = train_snps
        d['test_snps'] = test_snps
        d['positions'] = positions
        d['snp_ids'] = snp_ids
        d['chromosomes'] = chromosomes

        
        #Store information for plink file
        all_chr.extend(d['chromosomes'])
        all_pos.extend(d['positions'])
        all_train_snps.extend(train_snps)
        all_test_snps.extend(test_snps)
        all_snp_ids.extend(snp_ids)

    
    #Simulate traits
    phen_dict = pt.simulate_traits_w_snps(chrom_dict, num_traits=num_traits, verbose=verbose, 
                                          liability_thres=liability_thres)


    for t_i in range(num_traits):
        ss_filename = '%s_%d_ss.txt'%(out_prefix, t_i)

        with open(ss_filename,'w') as f:
            f.write('CHR     POS     SNP_ID    REF     ALT     REF_FRQ    PVAL    BETA    SE    N\n')

            for chrom in chrom_dict:
                d = chrom_dict[chrom]
                train_snps = d['train_snps']
                positions = d['positions']
                snp_ids = d['snp_ids']  
                num_snps = len(snp_ids)
                
                #Conduct GWAS, and write out results.
                train_phens = phen_dict['phenotypes'][t_i][part_i:]
                n_training = len(train_phens)
                
                if train_n_fracs is None:
                    print ('Normalizing genotypes')
                    snps_stds = sp.std(train_snps,axis=1)
                    snps_means = sp.mean(train_snps,axis=1)
                    snps_stds.shape = (len(snps_stds),1)
                    snps_means.shape = (len(snps_means),1)
                    snps = (train_snps - snps_means)/snps_stds
                                        
                    #Normalize phenotypes
                    beta_hats = sp.dot(snps, train_phens) / n_training
                    ses = [1/sp.sqrt(n_training)]*num_snps
                    b2s = beta_hats ** 2
                    f_stats = (n_training - 2) * b2s / (1 - b2s)
                    pvals = stats.f.sf(f_stats, 1, n_training - 2)
                    ns = [n_training]*num_snps
                
                else:  #Varying sample sizes
                    print ('Normalizing genotypes')
                    assert m == len(train_snps), 'Something wrong with SNP matrix lenght'
                    beta_hats = sp.zeros(m)
                    pvals = sp.zeros(m)
                    ses = sp.zeros(m)
                    ns= sp.zeros(m)
                    for j in range(num_snps):
                        train_n_frac = random.choice(train_n_fracs)
                        nt = int(train_n_frac*n_training)
                        random_selection = random.sample(range(n_training),nt)
                        tsnp = train_snps[j,random_selection]
                        tsnp = (tsnp - sp.mean(tsnp))/sp.std(tsnp)
                                        
                        tphens = train_phens[random_selection]
                        assert len(tphens)==len(tsnp)==nt,'Problems with random sample selection'
                        ns[j]=nt
                        
                        #Normalize phenotypes
                        beta_hats[j] = sp.dot(tsnp, tphens) / nt
                        ses[j] = 1/sp.sqrt(nt)
                        b2 = beta_hats[j] ** 2
                        f_stat = (nt - 2) * b2 / (1 - b2)
                        pvals[j] = stats.f.sf(f_stat, 1, nt - 2)
                    
                    if frac_0_pval>0:
                        zero_pvals_index = random.sample(range(num_snps),int(frac_0_pval*num_snps))
                        pvals[zero_pvals_index]= 0
                
                print ('Median p-value is %0.3f, and mean p-value is %0.3f'%(sp.median(pvals),sp.mean(pvals)))
                
                i = 0
                for pos, snp_id, eff, se, pval, n in zip(positions, snp_ids, beta_hats, ses, pvals, ns):
                    f.write('%s    %d    %s    A    G    0.5    %0.6e    %0.6e    %0.6e    %d\n'%
                                (chrom, pos, snp_id, pval, eff, se, n))
                    i += 1
                    
        #Write out Plink files
        train_plink_prefix = '%s_%d_train'%(out_prefix, t_i)
        test_plink_prefix = '%s_%d_test'%(out_prefix, t_i)
        
        train_phens = phen_dict['phenotypes'][t_i][part_i:]
        test_phens = phen_dict['phenotypes'][t_i][:part_i]
        
        all_train_snps = sp.stack( all_train_snps, axis=0 )
        all_test_snps = sp.stack( all_test_snps, axis=0 )
        write_fake_plink_file(all_train_snps, train_plink_prefix, chromosomes=all_chr, snp_ids=all_snp_ids,
                              positions=all_pos, phenotypes=train_phens)
        print ('Done w Training file')
        write_fake_plink_file(all_test_snps, test_plink_prefix, chromosomes=all_chr, snp_ids=all_snp_ids,
                              positions=all_pos, phenotypes=test_phens)
        print ('Done w Testing file')


def dict2json(sim_setup,json_file):
    json.dump(sim_setup, open(json_file,'w'))
    

def complex_simulations(out_prefix='/Users/au507860/REPOS/ldpred/test_data/'):
    sim1 = {'chrom_parameters':{1:{'h2':0.1, 'm':2000, 'p':0.5, 'adj_r2':0.9, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'}},
            'num_traits':5,
            'liability_thres':None,
            'n':10000,
            'test_frac':0.2,
            'train_n_fracs':None,
            'frac_0_pval':0,
            'diploid':True}
    dict2json(sim1,out_prefix+'sim1_parameters.json')
    simulate_plink_train_test_datasets(sim1,out_prefix=out_prefix+'sim1')

    sim2 = {'chrom_parameters':{1:{'h2':0.1, 'm':5000, 'p':0.005, 'adj_r2':0.9, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'}},
            'num_traits':5,
            'liability_thres':None,
            'n':5000,
            'test_frac':0.2,
            'train_n_fracs':None,
            'frac_0_pval':0,
            'diploid':True}
    dict2json(sim2,out_prefix+'sim2_parameters.json')
    simulate_plink_train_test_datasets(sim2,out_prefix=out_prefix+'sim2')

    sim3 = {'chrom_parameters':{1:{'h2':0.03, 'm':2000, 'p':0.01, 'adj_r2':0.95, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'},
           2:{'h2':0.01, 'm':2000, 'p':0.1, 'adj_r2':0.9, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'},
           3:{'h2':0.05, 'm':2000, 'p':1, 'adj_r2':0.9, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'},
           4:{'h2':0.02, 'm':2000, 'p':0.01, 'adj_r2':0.8, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'}},
            'num_traits':5,
            'liability_thres':None,
            'n':10000,
            'test_frac':0.2,
            'train_n_fracs':None,
            'frac_0_pval':0,
            'diploid':True}
    dict2json(sim3,out_prefix+'sim3_parameters.json')
    simulate_plink_train_test_datasets(sim3,out_prefix=out_prefix+'sim3')

    sim4 = {'chrom_parameters':{1:{'h2':0.02, 'm':2000, 'p':0.01, 'adj_r2':0.95, 'm_ld_chunk_size':400, 'effect_prior':'gaussian'},
           2:{'h2':0.02, 'm':2000, 'p':0.1, 'adj_r2':0.9, 'm_ld_chunk_size':200, 'effect_prior':'gaussian'},
           3:{'h2':0.02, 'm':2000, 'p':1, 'adj_r2':0.9, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'},
           4:{'h2':0.02, 'm':2000, 'p':1, 'adj_r2':0.8, 'm_ld_chunk_size':50, 'effect_prior':'gaussian'}},
            'num_traits':5,
            'liability_thres':None,
            'n':10000,
            'test_frac':0.2,
            'train_n_fracs':[0.5,0.6,0.8,0.9],
            'frac_0_pval':0.001,
            'diploid':True}
    dict2json(sim4,out_prefix+'sim4_parameters.json')
    simulate_plink_train_test_datasets(sim4,out_prefix=out_prefix+'sim4')
    
    sim5 = {'chrom_parameters':{1:{'h2':0.05, 'm':2000, 'p':0.01, 'adj_r2':0.95, 'm_ld_chunk_size':400, 'effect_prior':'gaussian'},
           2:{'h2':0.02, 'm':2000, 'p':0.1, 'adj_r2':0.9, 'm_ld_chunk_size':200, 'effect_prior':'laplace'},
           3:{'h2':0.02, 'm':2000, 'p':1, 'adj_r2':0.9, 'm_ld_chunk_size':100, 'effect_prior':'gaussian'},
           4:{'h2':0.03, 'm':2000, 'p':1, 'adj_r2':0.9, 'm_ld_chunk_size':50, 'effect_prior':'laplace'}},
            'num_traits':5,
            'liability_thres':None,
            'n':10000,
            'test_frac':0.2,
            'train_n_fracs':[0.4,0.8,0.9],
            'frac_0_pval':0.001,
            'diploid':True}
    dict2json(sim5,out_prefix+'sim5_parameters.json')
    simulate_plink_train_test_datasets(sim5,out_prefix=out_prefix+'sim5')

if __name__=='__main__':    
    complex_simulations()    