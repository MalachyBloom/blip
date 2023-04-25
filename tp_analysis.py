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
from tools.localization import quickMapmaker, FWxM, getAngRes, FWxM_contour, getLocalizationSummary,loadRunDir

def getDist(x,y,nside=32):
    th1,ph1 = hp.pixelfunc.pix2ang(nside,x)
    th2,ph2 = hp.pixelfunc.pix2ang(nside,y)
    d = float(hp.rotator.angdist([th1,ph1], [th2, ph2]))
    return d

def tpMetric(params, sample, parameters, inj, nside=32,D=[]):
    skymap = quickMapmaker(params, sample, parameters, inj, nside)
    pointSize_1, signalBlob_1, peak_value_1, border_1 = FWxM(skymap,fracMax=.5,nside=nside)
    pointSize_2, signalBlob_2, peak_value_2, border_2 = FWxM(skymap,fracMax=.5,nside=nside,ommission=signalBlob_1)
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

def getDistOverArea(run,outdir=None,summary_filename='tp_localization_summary',FWxM_filename='tp_val_list'):
    if outdir is None:
        outdir = run
    params, post, parameters, inj = loadRunDir(run)

    median_blm_val,b1 = tpMetric(params, np.median(post, axis=0), parameters, inj)
    mean_blm_val,b2 = tpMetric(params, np.average(post, axis=0), parameters, inj)
    print(len(post),median_blm_val)

    random.shuffle(post)
    vals = []
    count,r = 0,0
    for sample in post:
        val,b = tpMetric(params, sample, parameters, inj)
        vals.append(val)
        if count <= r*100 < count+1:
            print(str(int(r*100+.1)) + '%')
            r+=.01
        count+=1
        # print(count)

    print('100%')

    confidence68 = [np.quantile(vals, .16),np.quantile(vals, .84)]
    confidence90 = [np.quantile(vals, .05),np.quantile(vals, .95)] 
    confidence95 = [np.quantile(vals, .025),np.quantile(vals, .975)]

    print(str(np.mean(vals)), "+", confidence95[1], "-", confidence95[0])

    # with open('/home/vuk/mbloom/storage/tp_dur_lmax_pwr/error_bars/DistOverArea_'+ run + '.txt','w') as f:
    #     f.write("b_lm median: " + str(median_blm_val) +  '\n')
    #     f.write("b_lm mean: " + str(mean_blm_val) +  '\n')
    #     f.write("distribution median: " + str(np.median(vals)) +  '\n')
    #     f.write("distribution mean: " + str(np.mean(vals)) +  '\n')
    #     f.write("68'%' confidence interval: " + str(confidence68) +  '\n')
    #     f.write("90'%' confidence interval: " + str(confidence90) + '\n')
    #     f.write("95'%' confidence interval: " + str(confidence95) + '\n')
    #     f.write(str(vals))

    with open(outdir + summary_filename + '.txt','w') as f:
        f.write("distribution mean: " + str(np.mean(vals)) +  '\n')
        f.write("95'%' confidence interval: " + str(confidence95) + '\n')
        f.write('\n')
        f.write("b_lm median: " + str(median_blm_val) +  '\n')
        f.write("b_lm mean: " + str(mean_blm_val) +  '\n')
        f.write("distribution median: " + str(np.median(vals)) +  '\n')
        f.write("68'%' confidence interval: " + str(confidence68) +  '\n')
        f.write("90'%' confidence interval: " + str(confidence90) + '\n')

        
    with open(outdir + FWxM_filename + '.txt','w') as f:
        f.write(str(vals))

def main():
    nside = 32
    runs = ['tp_dur3_blmax2_pwr1e-7_sep2/','tp_dur3_blmax2_pwr1e-7_sep4/','tp_dur3_blmax2_pwr1e-7_sep6/','tp_dur3_blmax2_pwr1e-7_sep8/','tp_dur3_blmax2_pwr1e-7_sep10/','tp_dur3_blmax2_pwr1e-7_sep2/','tp_dur3_blmax2_pwr1e-7_sep4/']
    pathto = '/home/vuk/mbloom/storage/tp_dur_lmax_pwr/'
    for run in runs:
        print(run)
        getDistOverArea(pathto + run,outdir='/home/vuk/mbloom/storage/tp_dur_lmax_pwr/error_bars/',summary_filename=run,FWxM_filename='full'+run)
    
if __name__ == '__main__':
    main()