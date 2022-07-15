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

def sumNeighborsFracOfInj(skymap,inj_max,frac,ns):
    peak = list(skymap).index(max(skymap))
    signalBlob = {peak}
    border = set(hp.pixelfunc.get_all_neighbours(ns,peak))
    count = 0
    Sum = 0
    while Sum <= frac*inj_max:
        maxVal = 0
        for pxl in list(border):
            if skymap[pxl] > maxVal:
                maxVal = skymap[pxl]
                maxIndex = pxl
        signalBlob = signalBlob.union({maxIndex})
        border = border.union(set(hp.pixelfunc.get_all_neighbours(ns,maxIndex))).difference(signalBlob)
        count += 1
        Sum += maxVal
    return count/len(skymap)

def getPointSize(params, sample, parameters, inj):
    nside=32
    Omega_median_map = quickMapmaker(params, sample, parameters, inj, nside)
    pointSize = sumNeighborsFracOfInj(Omega_median_map,np.sum(Omega_median_map),.75,nside) 
    return pointSize

def getInterval(lst, confidence):
    lowToHigh = np.sort(lst)
    highToLow = np.sort(lst)[::-1]
    quantile = (1-confidence)/2
    for i in range(len(lowToHigh)):
        if i+1 >= quantile*len(lowToHigh):
            lowerBound = lowToHigh[i]
            break
    for i in range(len(highToLow)):
        if i+1 >= quantile*len(highToLow):
            upperBound = highToLow[i]
            break
    return [lowerBound,upperBound]

def getAreas(run):
    params, post, parameters, inj = draw(run)

    medianPointSize = getPointSize(params, np.median(post, axis=0), parameters, inj)
    meanPointSize = getPointSize(params, np.average(post, axis=0), parameters, inj)

    random.shuffle(post)
    areas=[]
    count=0
    r=0
    for sample in post:
        A = getPointSize(params, sample, parameters, inj)
        areas.append(A)
        if count <= r*len(post) < count+1:
            print(str(int(r*100+.1)) + '%')
            r+=.01
        count+=1
    print('100%')

    condidence68 = getInterval(areas, .68)
    condidence90 = getInterval(areas, .9)
    condidence95 = getInterval(areas, .95)
    with open('/mnt/c/Users/malac/Stochastic_LISA/storage/error_bars/'+ run + '.txt','w') as f:
        f.write("median: " + str(medianPointSize) +  '\n')
        f.write("mean: " + str(meanPointSize) +  '\n')
        f.write("68'%' confidence interval: " + str(condidence68) +  '\n')
        f.write("90'%' confidence interval: " + str(condidence90) + '\n')
        f.write("95'%' confidence interval: " + str(condidence95) + '\n')
        f.write(str(areas))

def main():
    run = "3mo_2_2e-7"
    getAreas(run)

if __name__ == '__main__':
    main()