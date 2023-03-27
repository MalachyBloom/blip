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

def draw(run):
    run = '/home/vuk/mbloom/storage/tp_dur_lmax_pwr/' + run
    with open(run +'/config.pickle','rb') as paramfile:
        params = pickle.load(paramfile)
        inj = pickle.load(paramfile)
        parameters = pickle.load(paramfile)
    post = np.loadtxt(run +"/post_samples.txt")
    return params, post, parameters, inj

def quickMapmaker(params, sample, parameters, inj, nside=32, saveto=None):
    
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


    blmax = params['lmax']  

    #### ------------ Now plot median value
    # median values of the posteriors
    # med_vals = np.median(post, axis=0)
    med_vals = sample

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

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    return list_object

def FWxM(skymap,x,ns=32,ommission=[]):
    ommittedSkymap = delete_multiple_element(list(skymap),list(ommission))
    peak_value = max(set(ommittedSkymap))
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
    return count/len(skymap),signalBlob,border #count/len(skymap)

def getDist(x,y,nside=32):
    th1,ph1 = hp.pixelfunc.pix2ang(nside,x)
    th2,ph2 = hp.pixelfunc.pix2ang(nside,y)
    d = float(hp.rotator.angdist([th1,ph1], [th2, ph2]))
    return d

def tpMetric(params, sample, parameters, inj, nside=32,D=[]):
    skymap = quickMapmaker(params, sample, parameters, inj, nside)
    pointSize_1, signalBlob_1, border_1 = FWxM(skymap,.5,nside)
    pointSize_2, signalBlob_2, border_2 = FWxM(skymap,.5,nside,signalBlob_1)
    ps = pointSize_1 + pointSize_2
    if pointSize_2 ==0:
        dist = 0
    else:
        for x in border_1:
            for y in border_2:
                d = getDist(x,y)
                D.append(d)
        dist = min(D)
    return dist/ps, border_1.union(border_2)

def getDistOverArea(run):
    params, post, parameters, inj = draw(run)

    median_blm_val,b1 = tpMetric(params, np.median(post, axis=0), parameters, inj)
    mean_blm_val,b2 = tpMetric(params, np.average(post, axis=0), parameters, inj)
    print(len(post),median_blm_val)

    random.shuffle(post)
    vals = []
    count,r = 0,0
    for sample in post:
        val,b = tpMetric(params, sample, parameters, inj)
        vals.append(val)
        # if count <= r*100 < count+1:
        #     print(str(int(r*100+.1)) + '%')
        #     r+=.01
        count+=1
        print(count)

    print('100%')

    confidence68 = [np.quantile(vals, .16),np.quantile(vals, .84)]
    confidence90 = [np.quantile(vals, .05),np.quantile(vals, .95)] 
    confidence95 = [np.quantile(vals, .025),np.quantile(vals, .975)]

    print(str(np.mean(vals)), "+", confidence95[1], "-", confidence95[0])

    with open('/home/vuk/mbloom/storage/tp_dur_lmax_pwr/error_bars/DistOverArea_'+ run + '.txt','w') as f:
        f.write("b_lm median: " + str(median_blm_val) +  '\n')
        f.write("b_lm mean: " + str(mean_blm_val) +  '\n')
        f.write("distribution median: " + str(np.median(vals)) +  '\n')
        f.write("distribution mean: " + str(np.mean(vals)) +  '\n')
        f.write("68'%' confidence interval: " + str(confidence68) +  '\n')
        f.write("90'%' confidence interval: " + str(confidence90) + '\n')
        f.write("95'%' confidence interval: " + str(confidence95) + '\n')
        f.write(str(vals))

def main():
    nside = 32
    runs = ['tp_dur3_blmax2_pwr1e-7_sep8']
    for run in runs:
        print(run)
        getDistOverArea(run)
    
if __name__ == '__main__':
    main()