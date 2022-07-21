import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
import random
# from src.populations import populations
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def draw(run):
    run = '/mnt/c/Users/malac/Stochastic_LISA/storage/dur_lmax_pwr/'+run
    with open(run +'/config.pickle','rb') as paramfile:
        params = pickle.load(paramfile)
        inj = pickle.load(paramfile)
        parameters = pickle.load(paramfile)
    post = np.loadtxt(run +"/post_samples.txt")
    return params, post, parameters, inj

def quickMapmaker(params, sample, parameters, inj, nside, saveto=None):
    
    if type(parameters) is dict:
        blm_start = len(parameters['noise']) + len(parameters['signal'])
        ## deal with extra parameter in broken_powerlaw:
        if 'spectrum_model' in params.keys():
            if params['spectrum_model']=='broken_powerlaw':
                blm_start = blm_start - 1
        
    elif type(parameters) is list:
        print("Warning: using a depreciated parameter format. Number of non-b_lm parameters is unknown, defaulting to n=4.")
        blm_start = 4
    else:
        raise TypeError("parameters argument is not dict or list.")

    # size of the blm array
    blm_size = Alm.getsize(params['lmax'])

    # nside = 2*params['nside']

    blmax = params['lmax']  

    #### ------------ Now plot median value
    # median values of the posteriors
    med_vals = sample #np.median(post, axis=0)

    ## blms.
    blms_median = np.append([1], med_vals[blm_start:])
    # print(len(post))

    # Omega at 1 mHz
    # handle various spectral models, but default to power law
    ## include backwards compatability check (to be depreciated later)
    if 'spectrum_model' in params.keys():
        if params['spectrum_model']=='broken_powerlaw':
            alpha_1 = med_vals[2]
            log_A1 = med_vals[3]
            alpha_2 = med_vals[2] - 0.667
            log_A2 = med_vals[4]
            Omega_1mHz_median= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
        else:
            if params['spectrum_model'] != 'powerlaw':
                print("Unknown spectral model. Defaulting to power law...")
            alpha = med_vals[2]
            log_Omega0 = med_vals[3]
            Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
    else:
        print("Warning: running on older output without specification of spectral model.")
        print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
        alpha = med_vals[2]
        log_Omega0 = med_vals[3]
        Omega_1mHz_median = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)

    ## Complex array of blm values for both +ve m values
    blm_median_vals = np.zeros(blm_size, dtype='complex')

    ## this is b00, alsways set to 1
    blm_median_vals[0] = 1
    cnt = 1

    for lval in range(1, blmax + 1):
        for mval in range(lval + 1):

            idx = Alm.getidx(blmax, lval, mval)

            if mval == 0:
                blm_median_vals[idx] = blms_median[cnt]
                cnt = cnt + 1
            else:
                ## prior on amplitude, phase
                blm_median_vals[idx] = blms_median[cnt] * np.exp(1j * blms_median[cnt+1])
                cnt = cnt + 2

    norm = np.sum(blm_median_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(blmax + 1):])**2)

    Omega_median_map  =  Omega_1mHz_median * (1.0/norm) * (hp.alm2map(blm_median_vals, nside))**2

    return Omega_median_map  

def FWxM(skymap,x,ns):
    peak_value = max(skymap)
    peak_index = list(skymap).index(peak_value)
    signalBlob = {peak_index}
    border = set(hp.pixelfunc.get_all_neighbours(ns,peak_index))
    count = 0
    full = False
    while not full:
        full = True
        for pxl_index in border:
            if skymap[pxl_index] > x*peak_value:
                signalBlob = signalBlob.union({pxl_index})
                border = border.union(set(hp.pixelfunc.get_all_neighbours(ns,pxl_index))).difference(signalBlob)
                count+=1
                full = False       
    return signalBlob, peak_value

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    return list_object

def getSignalBlob(Omega_median_map):
    nside=32
    # pointSize = fracOfMax(Omega_median_map,.5)
    # pointSize = sumNeighborsFracOfInj(Omega_median_map,np.sum(Omega_median_map),.75,nside)
    signalBlob, peak_val = FWxM(Omega_median_map, .5, nside)
    return signalBlob, peak_val

def getMapStatus(params, sample, parameters, inj):
    skymap = list(quickMapmaker(params, sample, parameters, inj, 32))
    signalBlob, peak_val = getSignalBlob(skymap)

    condition_1 = True
    for pxl_val in delete_multiple_element(skymap,list(signalBlob)):
        if pxl_val > .5*peak_val:
            condition_1 = False
            break
    
    condition_2 = False
    if hp.pixelfunc.ang2pix(32,1.5708, -1.5708) in signalBlob:
        condition_2 = True

    if condition_1 and condition_2:
        return True
    else:
        return False

def getQuality(run):
    params, post, parameters, inj = draw(run)

    medianStatus = getMapStatus(params, np.median(post, axis=0), parameters, inj)
    meanStatus = getMapStatus(params, np.average(post, axis=0), parameters, inj)
    print(medianStatus)

    random.shuffle(post)

    count,good,r=0,0,0
    print("There are " ,len(post)," samples for this run")
    for sample in post:
        if getMapStatus(params, sample, parameters, inj):
            good+=1

        if count <= r*len(post) < count+1:
            print(str(int(r*100+.1)) + '%')
            r+=.01
        count+=1

    recovery_quality = good/len(post)
        # if 0.017 <= A <= 0.018:
        #     print(sample)

    print('100%')
    print(recovery_quality)

    with open('/mnt/c/Users/malac/Stochastic_LISA/storage/recovery_quality/FWxM_5e-1_'+ run + '.txt','w') as f:
        f.write("median: " + str(medianStatus) +  '\n')
        f.write("mean: " + str(meanStatus) +  '\n')
        f.write(str(recovery_quality))

def main():
    runs = ['6mo_2_5e-8','6mo_2_6e-8','3mo_4_7e-8']
    for run in runs:
        print()
        print(run)
        getQuality(run)

if __name__ == '__main__':
    main()