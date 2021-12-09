import meep as mp
from autograd import numpy as np
from angle_spectral_refl_tran import compute_angular_spectral_tran_refl

n_SiO2 = 1.46
n_Al2O3 = 1.68
n_Ta2O5 = 2.09
n_TiO2 = 2.35
SiO2 = mp.Medium(epsilon=n_SiO2**2)
Al2O3 = mp.Medium(epsilon=n_Al2O3**2)
Ta2O5 = mp.Medium(epsilon=n_Ta2O5**2)
TiO2 = mp.Medium(epsilon=n_TiO2**2)

material_spec = [4,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,3,4,2,1,2,4,2,1,4,1,2,3,4,2,1,2,4,2,1,2,4,2,
1,2,4,2,1,2,4,2,1,2,4,2,1,4,3,2,1,2,4,1,2,4,3,2,1,4,2,1,2,4,2,1,2,4,3,1,2,4,2,1,2,4,2,1,4,2,1,2,4,1,4,3,1,2,4,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,
3,4,2,1,2,3,4,2,1,2,4,2,1,2,4,3,2,3,4,2,1,4,2,1,2,4,2,1,4,3,2,1,2,4,2,1,4,1,2,4,2,1,2,4,2,1,2,4,3,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,1,2,4,1,4,1,2,4,1,
4,3,1,4,2,1,4,1,4,1,2,4,2,1,4,2,1,4,1,4,1,2,4,1,4,1,4,1,2,4,1,4,1,4,1,4,2,1,4,1,4,1,2,4,1,2,4,1,2,4,1,4,1,4,1,4,1,2,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,
4,1,4,1,4,1,4,1,4,1,4,1,4,1]


D = np.array([494,500,94,490,118,500,117,500,115,500,123,382,123,500,123,383,113,500,114,380,100,500,98,380,112,378,100,254,73,406,109,369,69,500,84,376,87,500,
60,243,122,328,79,9,336,71,500,41,359,133,197,237,500,93,130,243,100,304,111,241,106,273,143,259,123,461,84,252,152,62,141,256,75,500,92,280,143,56,283,
92,36,436,112,245,162,38,343,1,80,500,247,36,38,170,259,135,332,56,137,96,312,52,228,173,219,135,227,43,344,234,59,247,31,246,458,230,34,274,45,232,335,
54,214,64,244,104,154,127,241,36,241,92,189,78,126,89,325,54,10,219,40,100,52,8,218,69,313,21,114,3,362,69,236,17,114,44,199,85,309,132,9,296,111,192,
76,114,210,21,50,263,23,125,126,123,247,152,103,147,94,168,105,124,69,172,27,212,11,20,85,67,187,46,159,78,121,56,242,31,142,22,210,12,189,203,29,166,
256,148,220,56,131,203,164,70,163,137,104,147,129,238,128,199,48,131,4,230,128,64,155,130,231,109,183,49,137,209,119,210,119,74,126,138,169,110,167,114,
189,117,52,145,120,157,121,123,33,116,99,65,103,152,41,99,200,81,177,91,148,123,90,35,120,116,120,161,108,132,101,128,95,153,98,105,116,90,102,82,111,128,
80,198,6,203,59,165,98,89,116,76,259,147], dtype=np.float64)
D /= 1000

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
emissivity = emissivity[:, :69]
with open('save_planck_spectral.npy', 'rb') as f:
    planck_spectrum = np.load(f)
with open('save_luminosity_function.npy', 'rb') as f:
    V_luminosity = np.load(f)

all_refl, all_tran = compute_angular_spectral_tran_refl(D)
all_refl_integrated = integrate_over_hemisphere(all_refl)
all_tran_integrated = integrate_over_hemisphere(all_tran)
emissivity_integrated = integrate_over_hemisphere(emissivity)

measured_spectral_emittance = planck_spectrum * emissivity_integrated

with open('save_optimized_stack_emissive_power.npy', 'wb') as f:
    np.save(f, measured_spectral_emittance)