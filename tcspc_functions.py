import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import scipy.constants as physics
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import least_squares

def R_si_ebc(musp, mua, rho, t):
    """Calculate and return the reflectance
    in a semi-infinite medium using the extrapolated boundary condition.
    
    Based on the following literature:
        [1] Hielscher et al. (1995) doi:10.1088/0031-9155/40/11/013
        [2] Cubeddu et al. (1999) doi:10.1364/AO.38.003670
        [3] Groenhuis et al. (1983) doi:10.1364/AO.22.002456
    
    "In the time domain, according to diffusion theory, the probability 
    of detecting a photon on the surface of a turbid semi-infinite medium 
    at adistance rho from the injection point after a time t can be 
    expressed as R = [...]" citing from [2]

    Keyword arguments: [pysical units]
    musp -- reduced scattering coefficient [/m]
    mua -- absorption coefficient [/m]
    rho -- source-detector separation [/m]
    t -- array of timesteps in [/s]
    
    Note that this always takes inputs in SI units.
    (i.e. /m instead of /cm or /mm).
    """
    # Refractive index for experiments on phantoms:
    n  = 1.34
    # Speed of light in vacuum [m/s]:
    c0 = physics.speed_of_light 
    # Speed of light in medium [m/s]:
    v  = c0/n
    # Effective mean free path in the medium [m]:
    z0 = 1/(musp+mua)
    # Diffusion coefficient [m]:
    D  = z0/3 
    # Internal reflectance approximated by [3]:
    rd = -1.44/n**2 + 0.71/n + 0.668 + 0.0636*n
    # Extrapolated distance:
    ze = 2*D*(1 + rd)/(1 - rd)
    
    R = (0.5*(4*np.pi*v*D)**(-3/2)*t**(-5/2) 
            * np.exp(-mua*v*t) 
            * (z0*np.exp(-(rho**2+z0**2)/(4*D*v*t))
               + (z0+2*ze)*np.exp(-(rho**2 + (z0+2*ze)**2)/(4*D*v*t))
              )
           )
    R[np.isnan(R)] = 0
    return R

def model_musp(wavelength_in_nm, coeffs):
    """
    Parameter estimation for the scattering model used by 
    mcxyz as the DKFZ simulation framework uses this approximation

    Keyword arguments: [pysical units]
    wavelength_in_nm -- wavelengths in numpy array [/nm]
    coeffs[] -- for fitting, including: 
        0: musp500 -- reduced scattering coeff. at 500 nm [/cm]
        1: fray -- fraction of Rayleigh scattering at 500 nm
        2: bmie  -- scatter power for mie scattering
    """
    musp = coeffs[0]*(coeffs[1]*(wavelength_in_nm/500)**(-4) 
                    + (1 - coeffs[1])*(wavelength_in_nm/500)**(-coeffs[2]))
    return musp*100.

def residuals_musp(coeffs, musprime_measured, wavelength_in_nm):
    return np.asarray(musprime_measured) - model_musp(wavelength_in_nm, coeffs)

def get_mu_from_DTOF(wavelength_in_nm, measurement_duration, irf_rho, 
                     irf_name, medium_rho, data_name, folder, verbose=False,
                     file_suffix = ".asc", irf_prefix = "/irf", 
                     medium_prefix = "/", 
                     initial_guess_musp = 10/physics.centi, 
                     initial_guess_mua = 0,
                     signal_start = 0.5, signal_stop = 0.01, load_set = 0):
    """Returns estimate for reduced scattering coefficien and 
    absorption coefficient (musp, mua) from a measured distribution of 
    times of flight (DTOF) and instrument response function (irf).
    The DTOF data is assumed to be saved in a csv(.asc) format default by 
    our SPC-160 software.
    Adjust the pd.read_csv() for differently formated input data.

    Literature:
        [1] Hielscher et al. (1995) doi:10.1088/0031-9155/40/11/013
        [2] Cubeddu et al. (1999) doi:10.1364/AO.38.003670
    
    Note on naming convention: 
        The file naming convention is arbitrary and should better be moved
        to an external method. I realize that "folder + medium_prefix 
        + wavelength_in_nm + data_name + file_suffix" is ugly but it is how 
        we structure our data but it is also easily adapted (sorry).
    
    Keyword arguments: [pysical units]
    wavelength_in_nm -- wavelength of the illumination pulse [/nm]
    t_spacing -- temporal spacing of the data [/s]
    irf_rho -- source-target-detector separation during irf measurement [/m]
    irf_name -- (file) name of the irf [/s]
    medium_rho -- source-detector separation during medium measurement [/m]
    data_name -- name of the data in folder
    folder -- path to the folder
    
    Optional keyword arguments: [pysical units]
    verbose -- printing and plotting fit results (default: False)
    file_suffix -- suffix of all csv DTOF files (default: ".asc")
    irf_prefix -- prefix to the irf DTOF file path (default: "/irf")
    medium_prefix -- prefix to the medium DTOF file path (default: "/")
    initial_guess_musp -- initial guess for the reduced scattering coefficient
                          in the model fitting [/m] (default: 10e2)
    initial_guess_mua -- initial guess for the absorption coefficient
                         in the model fitting [/m] (default: 0)
    signal_start -- signal threshold on the outgoing flank of DTOF 
                    measurement relative to peak (default: 0.8)
    signal_stop -- signal threshold on the outgoing flank of DTOF measurement
                   relative to its peak (default: 0.01)[2]
    load_set -- summing a whole numbered set with load_set items 
                if set to >0  (default: 0)
    """
    
    irf_path = str(folder) + irf_prefix + str(wavelength_in_nm) \
        + str(irf_name) + file_suffix
    df_irf = pd.read_csv(
        irf_path, header=None, skiprows=10, delim_whitespace=True, 
        names=["p"], skipfooter=1, engine='python')
    
    len_df = len(df_irf["p"])
    _t_spacing = measurement_duration/len_df
    _t = np.arange(0, len_df, 1)*_t_spacing

    if(verbose):
        peaks, _ = find_peaks(df_irf["p"], threshold=10, height=500, width=5)
        FWHM = peak_widths(df_irf["p"], peaks, rel_height=0.5)
        print(str(wavelength_in_nm) + " nm: irf FWHM =", \
              np.round(FWHM[0]*_t_spacing/physics.pico, 1), "ps")
    
    if (load_set > 0):
        for i in np.arange(1, load_set+1, 1):
            medium_path = str(folder) + medium_prefix + str(wavelength_in_nm) \
            + str(data_name) + str(i).zfill(2) + file_suffix
            if i == 1:
                df_medium = pd.read_csv(
                    medium_path, header=None, skiprows=10, delim_whitespace=True, 
                    names=["p"], skipfooter=1, engine='python')
            else:
                df_medium_ = pd.read_csv(
                    medium_path, header=None, skiprows=10, delim_whitespace=True, 
                    names=["p"], skipfooter=1, engine='python')
                df_medium["p"] = df_medium["p"] + df_medium_["p"]
    else:
        medium_path = str(folder) + medium_prefix + str(wavelength_in_nm) \
            + str(data_name) + file_suffix
        df_medium = pd.read_csv(
            medium_path, header=None, skiprows=10, delim_whitespace=True, 
            names=["p"], skipfooter=1, engine='python')

    peaks_medium, _ = find_peaks(df_medium["p"], height=100, width=20)
    FWHM_medium = peak_widths(df_medium["p"], peaks_medium, rel_height=0.5)
    if(verbose):
        plt.semilogy(_t, df_medium)
        plt.semilogy(_t, df_irf)
        plt.ylim(1e0,1e5)
        plt.show()
        print(str(wavelength_in_nm) + " nm: measured FWHM =", 
              np.round(FWHM[0]*_t_spacing/physics.pico, 1), "ps", 
              str(np.sum(df_medium["p"])), "photons in DTOF")
        print(peaks_medium)
        print(FWHM_medium)
        print(np.max(peaks_medium))

    measured_response = np.array(df_medium["p"])
    i_start = 0
    i_stop = 0
    i = np.argmax(measured_response)

    # Select time interval of useful signal with
    # signal_start (default: 80%) on the outgoing slope and 
    # signal_stop (default: 1%) on the outgoing side.
    while i < measured_response.size:
        if (i_start == 0 and 
            measured_response[i] < signal_start*np.max(measured_response)):
                i_start = i
        if (i_start > 0 and 
            measured_response[i] < signal_stop*np.max(measured_response)):
                i_stop = i
                i = measured_response.size
        i = i + 1
    
    # norming the measured response before fit
    measured_response = (measured_response[i_start:i_stop]\
                         /np.sum(measured_response[i_start:i_stop]))
    measured_time = (np.arange(i_start, i_stop, 1)*_t_spacing)
    irf_delay_index = int(np.round(irf_rho/physics.speed_of_light
                                   /_t_spacing, 0))
    
    def model_si_ecb(t, coeffs):
        irf  = np.asarray(df_irf["p"][irf_delay_index:])
        irt = _t[:-irf_delay_index]
        R_wo_conv = R_si_ebc(coeffs[0], coeffs[1], medium_rho, irt)
        conv_response = np.convolve(R_wo_conv, irf)
        model = (conv_response[i_start:i_stop]
               /np.sum(conv_response[i_start:i_stop]))
        if(verbose):
            plt.plot(model)
        return model

    x0 = np.array([initial_guess_musp, initial_guess_mua], dtype=float)

    def residuals_si_ecb(coeffs, measured_response, _t):
        return measured_response - model_si_ecb(_t, coeffs)

    fit = least_squares(residuals_si_ecb, x0, 
                        args=(measured_response, measured_time), 
                        bounds=((0, 1e3/physics.centi)), 
                            # max coefficients set to 1000/cm
                        ftol=1e-14 # increases precision in fitting
                       )
    if(verbose):
        plt.plot(measured_response, lw=3, alpha=0.6)
        plt.ylim(1e-5, np.max(measured_response)*1.3)
        plt.show()
        print("reduced scattering coefficient is "
              + str(fit.x[0]*physics.centi) + " /cm")
        print("absorption coefficient is "
              + str(fit.x[1]*physics.centi) + " /cm")

    return (fit.x[0], fit.x[1])