from angle_spectral_refl_tran import compute_angular_spectral_tran_refl
from autograd import numpy as np
import meep as mp


n_SiO2 = 1.46
n_Al2O3 = 1.68
n_Ta2O5 = 2.09
n_TiO2 = 2.35
SiO2 = mp.Medium(epsilon=n_SiO2**2)
Al2O3 = mp.Medium(epsilon=n_Al2O3**2)
Ta2O5 = mp.Medium(epsilon=n_Ta2O5**2)
TiO2 = mp.Medium(epsilon=n_TiO2**2)

def make_design(n_layers, lambda_0, chirp_param):
    # design qw stack to block wavelength of lambda_0 microns

    fcen_0 = 1/lambda_0
    print('Center frequency', fcen_0)

    # set up qw stack

    n1 = n_SiO2
    n2 = n_TiO2
    material1 = SiO2
    material2 = TiO2
    a = (n1+n2)/(4*n1*n2*fcen_0)
    d1 = a*n2/(n1+n2)
    d2 = a*n1/(n1+n2)

    D = []
    for i in range(n_layers):
        chirp_factor = (1+chirp_param/(1-chirp_param))**(1/(n_layers-1))
        D.append(d1*chirp_factor)
        D.append(d2*chirp_factor)

    return D

def compute_refl_tran_from_params(n_layers, lambda_0, chirp_param):
    D = make_design(n_layers, lambda_0, chirp_param)
    all_refl, all_tran = compute_angular_spectral_tran_refl(D)
    return all_refl, all_tran

    
def integrate_over_hemisphere(input_array):
    '''
    input array should go be (nfreq by nangle), with the angles range from 0 to 70 degrees
    '''
    n_angles = 69
    angles = np.arange(0, n_angles)
    angles = np.reshape(angles, (1, n_angles))
    scaled_array = input_array * np.sin(angles) * np.cos(angles)
    normalization = np.sin(angles)*np.cos(angles)
    return np.trapz(scaled_array, axis=1)/np.trapz(normalization, axis=1)

with open('save_tungsten_emissivity.npy', 'rb') as f:
    emissivity = np.load(f)
with open('save_planck_spectral.npy', 'rb') as f:
    planck_spectrum = np.load(f)
with open('save_luminosity_function.npy', 'rb') as f:
    V_luminosity = np.load(f)

n_layers = 25
lambda_0_all = np.arange(.4, 2, 0.1)
chirp_param = 0

for lambda_0 in lambda_0_all:
    all_refl, all_tran = compute_refl_tran_from_params(n_layers, lambda_0, chirp_param)
    print("Computing hemispherical values")
    # compute hemispherical values
    all_refl_integrated = integrate_over_hemisphere(all_refl)
    all_tran_integrated = integrate_over_hemisphere(all_tran)
    emissivity_integrated = integrate_over_hemisphere(emissivity)
    
    # use planck to find actual radiant emittance
    measured_spectral_emittance = planck_spectrum * emissivity_integrated
    
    print("integrating to find efficiency")
    # integrate to find efficiency
    eta = np.trapz(measured_spectral_emittance * V_luminosity)/np.trapz(measured_spectral_emittance)

    print('lam_0', lambda_0)
    print('chirp_param', chirp_param)
    print('n_layers', n_layers)