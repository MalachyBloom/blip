import pickle
import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import sys
import configparser
import subprocess
from src.makeLISAdata import LISAdata
from src.bayes import Bayes
from tools.plotmaker import plotmaker
import matplotlib.pyplot as plt
import scipy.signal as sg


class LISA(LISAdata, Bayes):

    '''
    Generic class for getting data and setting up the prior space
    and likelihood. This is tuned for ISGWB analysis at the moment
    but it should not be difficult to modify for other use cases.
    '''

    def __init__(self,  params, inj):
        
        # set up the LISAdata class
        LISAdata.__init__(self, params, inj)

        # Make noise spectra
        self.which_noise_spectrum()
        self.which_astro_signal()

        # Generate or get mldc data
        if self.params['mldc']:
            self.read_mldc_data()
            print("Data read.")
        else:
            self.makedata()

        # Set up the Bayes class
        # Bayes.__init__(self)
        
        # Figure out which response function to use for recoveries
        self.which_response()
        # import pdb; pdb.set_trace()
        

    def makedata(self):

        '''
        Just a wrapper function to use the methods the LISAdata class
        to generate data. Return Frequency domain data.
        '''

        # Generate TDI noise (check here again)
        times, self.h1, self.h2, self.h3 = self.gen_noise_spectrum()
        
        
        delt = times[1] - times[0]

        # Cut to required size
        N = int((self.params['dur'])/delt)
        self.h1, self.h2, self.h3 = self.h1[0:N], self.h2[0:N], self.h3[0:N]

        # Generate TDI isotropic signal
        if self.inj['doInj']:
            if self.inj['injtype'] == 'primordial':
                h1_gw, h2_gw, h3_gw, times = self.add_earlygw_data(dur=self.params['dur'])

            h1_gw, h2_gw, h3_gw = h1_gw[0:N], h2_gw[0:N], h3_gw[0:N]

            # Add gravitational-wave time series to noise time-series
            self.h1 = self.h1 + h1_gw
            self.h2 = self.h2 + h2_gw
            self.h3 = self.h3 + h3_gw

        self.timearray = times[0:N]
        if delt != (times[1] - times[0]):
            raise ValueError('The noise and signal arrays are at different sampling frequencies!')

        # Desample if we increased the sample rate for time-shifts.
        if self.params['fs'] != 1.0/delt:
            self.params['fs'] = 1.0/delt

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, dummy, self.tsegstart, self.tsegmid = self.tser2fser(self.h1, self.h2, self.h3, self.timearray)
        
        df = 1/self.params['dur']
        self.fdata = np.arange(self.params['fmin'], self.params['fmax']+df, df)
        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)
        
        # Modelled Noise PSD
        C_noise = self.instr_noise_spectrum(self.fdata, self.f0)
        # Extract noise auto-power
        S1, S2, S3 = C_noise[0, 0, :], C_noise[1, 1, :], C_noise[2, 2, :] 
        h1_sens, h2_sens, h3_sens = S1**2, S2**2, S3**2
        
        
        H0 = 2.2e-18
        
        ## Choosing S1
        # omega_sens1 = 4/3 * (np.pi/H0)**2 * self.fdata**3 * h1_sens

        # import pdb; pdb.set_trace()
        
        
    def which_noise_spectrum(self):

        # Figure out which instrumental noise spectra to use
        if self.params['tdi_lev'] == 'aet':
            self.instr_noise_spectrum = self.aet_noise_spectrum
            self.gen_noise_spectrum = self.gen_aet_noise
        elif self.params['tdi_lev'] == 'xyz':
            self.instr_noise_spectrum = self.xyz_noise_spectrum
            self.gen_noise_spectrum = self.gen_xyz_noise
        elif self.params['tdi_lev'] == 'michelson':
            self.instr_noise_spectrum = self.mich_noise_spectrum
            self.gen_noise_spectrum = self.gen_michelson_noise

    def which_response(self):
        # Figure out which antenna patterns to use
        
        # Stationary LISA case:
        if self.params['lisa_config'] == 'stationary':
            if self.params['modeltype'] == 'primordial' and self.params['tdi_lev'] == 'xyz':
                self.response_mat = self.isgwb_xyz_response(self.f0, self.tsegmid)
            elif self.params['modeltype'] == 'primordial' and self.params['tdi_lev'] == 'aet':
                self.response_mat = self.isgwb_aet_response(self.f0, self.tsegmid)
            elif self.params['modeltype'] == 'primordial' and self.params['tdi_lev'] == 'michelson':
                self.response_mat = self.isgwb_mich_response(self.f0, self.tsegmid)
            
            elif self.params['modeltype'] == 'noise_only':
                print('Noise only model chosen ...')
            else:
                raise ValueError('Unknown recovery model selected')

    def which_astro_signal(self):
        # Figure out which antenna patterns to use
        if self.inj['injtype'] == 'primordial' and self.params['tdi_lev'] == 'xyz':
            self.add_astro_signal = self.isgwb_xyz_response
        elif self.inj['injtype'] == 'primordial' and self.params['tdi_lev'] == 'aet':
            self.add_astro_signal = self.isgwb_aet_response
        elif self.inj['injtype'] == 'primordial' and self.params['tdi_lev'] == 'michelson':
            self.add_astro_signal = self.isgwb_mich_response

        else:
            raise ValueError('Unknown recovery model selected')
        
def blip(paramsfile='params.ini'):
    '''
    The main workhorse of the bayesian pipeline.

    Input:
    Params File

    Output: Files containing evidence and pdfs of the parameters
    '''

    #  --------------- Read the params file --------------------------------

    # Initialize Dictionaries
    params = {}
    inj = {}

    config = configparser.ConfigParser()
    config.read(paramsfile)

    # Params Dict
    params['fmin'] = float(config.get("params", "fmin"))
    params['fmax'] = float(config.get("params", "fmax"))
    params['dur'] = float(config.get("params", "duration"))
    params['seglen'] = float(config.get("params", "seglen"))
    params['fs'] = float(config.get("params", "fs"))
    params['Shfile'] = config.get("params", "Shfile")
    params['mldc'] = int(config.get("params", "mldc"))
    params['datatype'] = str(config.get("params", "datatype"))
    params['loadResponse'] = int(config.get("params", "loadResponse"))
    params['loadCustom'] = int(config.get("params", "loadCustom"))
    params['responsefile1'] = str(config.get("params", "responsefile1"))
    params['responsefile2'] = str(config.get("params", "responsefile2"))
    params['responsefile3'] = str(config.get("params", "responsefile3"))
    params['datafile'] = str(config.get("params", "datafile"))
    params['fref'] = float(config.get("params", "fref"))
    params['modeltype'] = str(config.get("params", "modeltype"))
    params['tdi_lev'] = str(config.get("params", "tdi_lev"))
    params['lisa_config'] = str(config.get("params", "lisa_config"))
    params['nside'] = int(config.get("params", "nside"))
    params['lmax'] = int(config.get("params", "lmax"))

    # Injection Dict
    inj['doInj'] = int(config.get("inj", "doInj"))
    inj['injtype'] = str(config.get("inj", "injtype"))
    inj['ln_omega0'] = np.log10(float(config.get("inj", "omega0")))
    inj['alpha'] = float(config.get("inj", "alpha"))
    inj['log_Np'] = np.log10(float(config.get("inj", "Np")))
    inj['log_Na'] = np.log10(float(config.get("inj", "Na")))

    if inj['injtype'] == 'primordial':
        inj['nHat'] = float(config.get("inj", "nHat"))
        inj['wHat'] = float(config.get("inj", "wHat"))
        inj['rts'] = float(config.get("inj", "rts"))

    # some run parameters
    params['out_dir'] = str(config.get("run_params", "out_dir"))
    params['doPreProc'] = int(config.get("run_params", "doPreProc"))
    params['input_spectrum'] = str(config.get("run_params", "input_spectrum"))
    params['FixSeed'] = str(config.get("run_params", "FixSeed"))
    params['seed'] = int(config.get("run_params", "seed"))
    verbose = int(config.get("run_params", "verbose"))
    nlive = int(config.get("run_params", "nlive"))
    nthread = int(config.get("run_params", "Nthreads"))

    if params['FixSeed']:
        from tools.SetRandomState import SetRandomState as setrs
        seed = params['seed']
        randst = setrs(seed)
    else:
        randst = None
    # Initialize lisa class
    lisa = LISA(params, inj)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide (only) the params file as an argument')
    else:
        blip(sys.argv[1])
