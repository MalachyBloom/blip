import numpy as np
import numpy.linalg as LA
from scipy.special import lpmn, sph_harm
import types
import healpy as hp
from src.sph_geometry import sph_geometry

class geometry(sph_geometry):

    '''
    Module containing geometry methods. The methods here include calculation of antenna patters 
    for a single doppler channel, for the three michelson channels or for the AET TDI channels 
    and calculation of noise power spectra for various channel combinations.
    '''

    def __init__(self):

        if self.params['modeltype'] == 'sph_sgwb' or self.inj['injtype'] == 'sph_sgwb':
            sph_geometry.__init__(self)



    def lisa_orbits(self, tsegmid):


        '''
        Define LISA orbital positions at the midpoint of each time integration segment using analytic MLDC orbits.

        Parameters
        -----------

        tsegmid  :  array
            A numpy array of the tsegmid for each time integration segment.

        Returns
        -----------
        rs1, rs2, rs3  :  array
            Arrays of satellite positions for each segment midpoint in timearray. e.g. rs1[1] is [x1,y1,z1] at t=midpoint[1]=timearray[1]+(segment length)/2.
        '''
        ## Branch orbiting and stationary cases; compute satellite position in stationary case based off of first time entry in data.
        if self.params['lisa_config'] == 'stationary':
            # Calculate start time from tsegmid
            tstart = tsegmid[0] - (tsegmid[1] - tsegmid[0])/2
            # Fill times array with just the start time
            times = np.empty(len(tsegmid))
            times.fill(tstart)
        elif self.params['lisa_config'] == 'orbiting':
            times = tsegmid
        else:
            raise ValueError('Unknown LISA configuration selected')


        ## Semimajor axis in m
        a = 1.496e11


        ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
        betaphase = 0
        alphaphase = 0

        ## Orbital angle alpha(t)
        at = (2*np.pi/31557600)*times + alphaphase

        ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
        e = self.armlength/(2*a*np.sqrt(3))

        ## Initialize arrays
        beta_n = (2/3)*np.pi*np.array([0,1,2])+betaphase

        ## meshgrid arrays
        Beta_n, Alpha_t = np.meshgrid(beta_n, at)

        ## Calculate inclination and positions for each satellite.
        x_n = a*np.cos(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.sin(Beta_n) - (1+(np.sin(Alpha_t))**2)*np.cos(Beta_n))
        y_n = a*np.sin(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.cos(Beta_n) - (1+(np.cos(Alpha_t))**2)*np.sin(Beta_n))
        z_n = -np.sqrt(3)*a*e*np.cos(Alpha_t - Beta_n)


        ## Construct position vectors r_n
        rs1 = np.array([x_n[:, 0],y_n[:, 0],z_n[:, 0]])
        rs2 = np.array([x_n[:, 1],y_n[:, 1],z_n[:, 1]])
        rs3 = np.array([x_n[:, 2],y_n[:, 2],z_n[:, 2]])

        return rs1, rs2, rs3



    def isgwb_mich_response(self, f0, tsegmid):
        '''
        Calculate the Antenna pattern/detector transfer function for an isotropic SGWB using basic michelson channels.
        Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array.
        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        tsegstart  :  float
            A numpy array of segment start times

        tsegmid  :  float
            A numpy array of segment midpoints

        Returns
        ---------

        response_tess   :   float
            4D array of covariance matrices for antenna patterns of the three channels, integrated over sky direction
            and averaged over polarization, across all frequencies and times.

        '''

        npix = hp.nside2npix(self.params['nside'])

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        # Angular coordinates of pixel indices
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = np.array([np.sqrt(1-ctheta**2)*np.cos(phi),np.sqrt(1-ctheta**2)*np.sin(phi),ctheta])

        # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
        rs1, rs2, rs3 = self.lisa_orbits(tsegmid)

        ## Calculate directional unit vector dot products
        ## Dimensions of udir is time-segs x sky-pixels
        udir = np.einsum('ij,ik',(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :],omegahat)
        vdir = np.einsum('ij,ik',(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],omegahat)
        wdir = np.einsum('ij,ik',(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],omegahat)


        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''

        mhat = np.array([np.sin(phi),-np.cos(phi),np.zeros(len(phi))])
        nhat = np.array([np.cos(phi)*ctheta,np.sin(phi)*ctheta,-np.sqrt(1-ctheta**2)])

        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        Fplus_u = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :], (rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        Fplus_v = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        Fplus_w = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        # 1/2 u x u : ecross
        Fcross_u = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :],(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))

        Fcross_v = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))

        Fcross_w = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))



        # Initlize arrays for the detector reponse
        R1 = np.zeros((f0.size,  tsegmid.size), dtype='complex')
        R2 = np.zeros((f0.size,  tsegmid.size), dtype='complex')
        R3 = np.zeros((f0.size,  tsegmid.size), dtype='complex')
        R12 = np.zeros((f0.size, tsegmid.size), dtype='complex')
        R13 = np.zeros((f0.size, tsegmid.size), dtype='complex')
        R23 = np.zeros((f0.size, tsegmid.size), dtype='complex')

        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):

            # Calculate GW transfer function for the michelson channels
            gammaU_plus    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV_plus    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW_plus    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))


            # Calculate GW transfer function for the michelson channels
            gammaU_minus    =    1/2 * (np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(3 - udir)) + \
                             np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(1 - udir)))

            gammaV_minus    =    1/2 * (np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - vdir)) + \
                             np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - vdir)))

            gammaW_minus    =    1/2 * (np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - wdir)) + \
                             np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir)))


            ## Michelson antenna patterns
            ## Calculate Fplus
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus  - Fcross_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3))
            Fcross2 = 0.5*(Fcross_w*gammaW_plus  - Fcross_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3))
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3))

            ## Detector response summed over polarization and integrated over sky direction
            ## The travel time phases for the which are relevent for the cross-channel are
            ## accounted for in the Fplus and Fcross expressions above.
            R1[ii, :]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus1))**2 + (np.absolute(Fcross1))**2, axis=1 )
            R2[ii, :]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus2))**2 + (np.absolute(Fcross2))**2, axis=1 )
            R3[ii, :]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus3))**2 + (np.absolute(Fcross3))**2, axis=1 )
            R12[ii, :] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus2 + np.conj(Fcross1)*Fcross2, axis=1)
            R13[ii, :] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus3 + np.conj(Fcross1)*Fcross3, axis=1)
            R23[ii, :] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus2)*Fplus3 + np.conj(Fcross2)*Fcross3, axis=1)

        response_mat = np.array([ [R1, R12, R13] , [np.conj(R12), R2, R23], [np.conj(R13), np.conj(R23), R3] ])

        return response_mat


    def isgwb_xyz_response(self, f0, tsegmid):

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using X, Y and Z TDI
        channels. Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------
        f0   : float
            A numpy array of scaled frequencies (see above for def)

        Returns
        ---------
        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        mich_response_mat = self.isgwb_mich_response(f0,tsegmid)
        xyz_response_mat = 4 * mich_response_mat * (np.sin(2*f0[None, None, :, None]))**2

        return xyz_response_mat


    def isgwb_aet_response(self, f0, tsegmid):

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using A, E and T TDI channels.
        Note that since this is the response to an isotropic background, the response function is integrated over sky direction
        and averaged over polarozation. The angular integral is a linear and rectangular in the cos(theta) and phi space.  Note
        that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------
        f0   : float
            A numpy array of scaled frequencies (see above for def)


        Return      s
        ---------
        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        xyz_response_mat = self.isgwb_xyz_response(f0,tsegmid)

        ## Upnack xyz matrix to make assembling the aet matrix easier
        RXX, RYY, RZZ = xyz_response_mat[0, 0], xyz_response_mat[1, 1], xyz_response_mat[2, 2]
        RXY, RXZ, RYZ = xyz_response_mat[0, 1], xyz_response_mat[0, 2], xyz_response_mat[1, 2]


        ## construct AET matrix elements
        RAA = (1/9) * (4*RXX + RYY + RZZ - 2*RXY - 2*np.conj(RXY) - 2*RXZ - 2*np.conj(RXZ) + \
                        RYZ  + np.conj(RYZ))

        REE = (1/3) * (RZZ + RYY - RYZ - np.conj(RYZ))

        RTT = (1/9) * (RXX + RYY + RZZ + RXY + np.conj(RXY) + RXZ + np.conj(RXZ) + RYZ + np.conj(RYZ))

        RAE = (1/(3*np.sqrt(3))) * (RYY - RZZ - RYZ + np.conj(RYZ) + 2*RXZ - 2*RXY)

        RAT = (1/9) * (2*RXX - RYY - RZZ + 2*RXY - np.conj(RXY) + 2*RXZ - np.conj(RXZ) - RYZ - np.conj(RYZ))

        RET = (1/(3*np.sqrt(3))) * (RZZ - RYY - RYZ + np.conj(RYZ) + np.conj(RXZ) - np.conj(RXY))

        aet_response_mat = np.array([ [RAA, RAE, RAT] , \
                                    [np.conj(RAE), REE, RET], \
                                    [np.conj(RAT), np.conj(RET), RTT] ])

        return aet_response_mat

    def ps_mich_response(self, f0, theta_inj, phi_inj, tsegmid):
        '''
        Calculate the Antenna pattern/detector transfer function for an point source using basic michelson channels.
        Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array.
        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        tsegstart  :  float
            A numpy array of segment start times

        tsegmid  :  float
            A numpy array of segment midpoints

        Returns
        ---------

        response_tess   :   float
            4D array of covariance matrices for antenna patterns of the three channels, integrated over sky direction
            and averaged over polarization, across all frequencies and times.

        '''

        npix = hp.nside2npix(self.params['nside'])

        inj_map = np.zeros(npix)
                
        # identify the pixel with the point source
        ps_id = hp.ang2pix( self.params['nside'] , theta_inj, phi_inj)
#        inj_map[ps_id-1:ps_id+1] = 1
        inj_map[ps_id] = 1

        # Array of pixel indices
        pix_idx  = np.arange(npix)

        # Angular coordinates of pixel indices
        theta, phi = hp.pix2ang(self.params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(self.params['nside'])

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = np.array([np.sqrt(1-ctheta**2)*np.cos(phi),np.sqrt(1-ctheta**2)*np.sin(phi),ctheta])

        # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
        rs1, rs2, rs3 = self.lisa_orbits(tsegmid)

        ## Calculate directional unit vector dot products
        ## Dimensions of udir is time-segs x sky-pixels
        udir = np.einsum('ij,ik',(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :],omegahat)
        vdir = np.einsum('ij,ik',(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],omegahat)
        wdir = np.einsum('ij,ik',(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],omegahat)


        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''

        mhat = np.array([np.sin(phi),-np.cos(phi),np.zeros(len(phi))])
        nhat = np.array([np.cos(phi)*ctheta,np.sin(phi)*ctheta,-np.sqrt(1-ctheta**2)])

        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        Fplus_u = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :], (rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        Fplus_v = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        Fplus_w = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) - np.einsum("ik,jk -> ijk",nhat,nhat))

        # 1/2 u x u : ecross
        Fcross_u = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :],(rs2-rs1)/LA.norm(rs2-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))

        Fcross_v = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :],(rs3-rs1)/LA.norm(rs3-rs1,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))

        Fcross_w = 0.5*np.einsum("ijk,ijl", \
                              np.einsum("ik,jk -> ijk",(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :],(rs3-rs2)/LA.norm(rs3-rs2,axis=0)[None, :]), \
                              np.einsum("ik,jk -> ijk",mhat,mhat) + np.einsum("ik,jk -> ijk",nhat,nhat))



        # Initlize arrays for the detector reponse
        R1 = np.zeros((f0.size,  tsegmid.size), dtype='complex')
        R2 = np.zeros((f0.size,  tsegmid.size), dtype='complex')
        R3 = np.zeros((f0.size,  tsegmid.size), dtype='complex')
        R12 = np.zeros((f0.size, tsegmid.size), dtype='complex')
        R13 = np.zeros((f0.size, tsegmid.size), dtype='complex')
        R23 = np.zeros((f0.size, tsegmid.size), dtype='complex')

        # Calculate the detector response for each frequency
        for ii in range(0, f0.size):

            # Calculate GW transfer function for the michelson channels
            gammaU_plus    =    1/2 * (np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(3+udir)) + \
                             np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(1+udir)))

            gammaV_plus    =    1/2 * (np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(3+vdir)) + \
                             np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(1+vdir)))

            gammaW_plus    =    1/2 * (np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(3+wdir)) + \
                             np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(1+wdir)))


            # Calculate GW transfer function for the michelson channels
            gammaU_minus    =    1/2 * (np.sinc((f0[ii])*(1 + udir)/np.pi)*np.exp(-1j*f0[ii]*(3 - udir)) + \
                             np.sinc((f0[ii])*(1 - udir)/np.pi)*np.exp(-1j*f0[ii]*(1 - udir)))

            gammaV_minus    =    1/2 * (np.sinc((f0[ii])*(1 + vdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - vdir)) + \
                             np.sinc((f0[ii])*(1 - vdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - vdir)))

            gammaW_minus    =    1/2 * (np.sinc((f0[ii])*(1 + wdir)/np.pi)*np.exp(-1j*f0[ii]*(3 - wdir)) + \
                             np.sinc((f0[ii])*(1 - wdir)/np.pi)*np.exp(-1j*f0[ii]*(1 - wdir)))


            ## Michelson antenna patterns
            ## Calculate Fplus
            Fplus1 = 0.5*(Fplus_u*gammaU_plus - Fplus_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3)) * inj_map[None, :]
            Fplus2 = 0.5*(Fplus_w*gammaW_plus - Fplus_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3)) * inj_map[None, :]
            Fplus3 = 0.5*(Fplus_v*gammaV_minus - Fplus_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3)) * inj_map[None, :]

            ## Calculate Fcross
            Fcross1 = 0.5*(Fcross_u*gammaU_plus  - Fcross_v*gammaV_plus)*np.exp(-1j*f0[ii]*(udir + vdir)/np.sqrt(3)) * inj_map[None, :]
            Fcross2 = 0.5*(Fcross_w*gammaW_plus  - Fcross_u*gammaU_minus)*np.exp(-1j*f0[ii]*(-udir + vdir)/np.sqrt(3)) * inj_map[None, :]
            Fcross3 = 0.5*(Fcross_v*gammaV_minus - Fcross_w*gammaW_minus)*np.exp(1j*f0[ii]*(vdir + wdir)/np.sqrt(3)) * inj_map[None, :]

            ## Detector response summed over polarization and integrated over sky direction
            ## The travel time phases for the which are relevent for the cross-channel are
            ## accounted for in the Fplus and Fcross expressions above.
            R1[ii, :]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus1))**2 + (np.absolute(Fcross1))**2, axis=1 )
            R2[ii, :]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus2))**2 + (np.absolute(Fcross2))**2, axis=1 )
            R3[ii, :]  = dOmega/(8*np.pi)*np.sum( (np.absolute(Fplus3))**2 + (np.absolute(Fcross3))**2, axis=1 )
            R12[ii, :] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus2 + np.conj(Fcross1)*Fcross2, axis=1)
            R13[ii, :] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus1)*Fplus3 + np.conj(Fcross1)*Fcross3, axis=1)
            R23[ii, :] = dOmega/(8*np.pi)*np.sum( np.conj(Fplus2)*Fplus3 + np.conj(Fcross2)*Fcross3, axis=1)

        response_mat = np.array([ [R1, R12, R13] , [np.conj(R12), R2, R23], [np.conj(R13), np.conj(R23), R3] ])
        return response_mat
   
    def ps_xyz_response(self, f0, theta_inj, phi_inj, tsegmid):

        '''
        Calcualte the Antenna pattern/ detector transfer function functions to an isotropic SGWB using X, Y and Z TDI
        channels. Note that since this is the response to an isotropic background, the response function is integrated
        over sky direction and averaged over polarozation. The angular integral is a linear and rectangular in the
        cos(theta) and phi space.  Note also that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------
        f0   : float
            A numpy array of scaled frequencies (see above for def)

        Returns
        ---------
        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged over polarization.
        '''

        mich_response_mat = self.ps_mich_response(f0, theta_inj, phi_inj, tsegmid)
        xyz_response_mat = 4 * mich_response_mat * (np.sin(2*f0[None, None, :, None]))**2

        return xyz_response_mat   


    def ps_aet_response(self, f0, theta_inj, phi_inj, tsegmid, tsegstart):
    
        '''
        Calculate the Antenna pattern/ detector transfer function functions to a GW originating in the direction of (theta, phi) 
        for the A, E and T TDI channels, and using a spherical harmonic decomposition. Note that the response function to power 
        is integrated over sky direction with the appropriate spherical harmonics, and averaged over polarozation. The angular
        integral is numerically done by divvying up the sky into a healpix grid.

        Note that f0 is (pi*L*f)/c and is input as an array



        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)



        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient.
        '''

        xyz_response_mat = self.ps_xyz_response(f0, theta_inj, phi_inj, tsegmid)

        ## Upnack xyz matrix to make assembling the aet matrix easier
        RXX, RYY, RZZ = xyz_response_mat[0, 0], xyz_response_mat[1, 1], xyz_response_mat[2, 2]
        RXY, RXZ, RYZ = xyz_response_mat[0, 1], xyz_response_mat[0, 2], xyz_response_mat[1, 2]

        ## construct AET matrix elements
        RAA = (1/9) * (4*RXX + RYY + RZZ - 2*RXY - 2*np.conj(RXY) - 2*RXZ - 2*np.conj(RXZ) + \
                        RYZ  + np.conj(RYZ))

        REE = (1/3) * (RZZ + RYY - RYZ - np.conj(RYZ))

        RTT = (1/9) * (RXX + RYY + RZZ + RXY + np.conj(RXY) + RXZ + np.conj(RXZ) + RYZ + np.conj(RYZ))

        RAE = (1/(3*np.sqrt(3))) * (RYY - RZZ - RYZ + np.conj(RYZ) + 2*RXZ - 2*RXY)

        RAT = (1/9) * (2*RXX - RYY - RZZ + 2*RXY - np.conj(RXY) + 2*RXZ - np.conj(RXZ) - RYZ - np.conj(RYZ))

        RET = (1/(3*np.sqrt(3))) * (RZZ - RYY - RYZ + np.conj(RYZ) + np.conj(RXZ) - np.conj(RXY))

        aet_response_mat = np.array([ [RAA, RAE, RAT] , \
                                    [np.conj(RAE), REE, RET], \
                                    [np.conj(RAT), np.conj(RET), RTT] ])

        return aet_response_mat
