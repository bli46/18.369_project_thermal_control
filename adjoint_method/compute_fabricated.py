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

material_spec = [3 if i%2==0 else 1 for i in range(90)]


D = np.array([18,47,156,212,178,23,51,224,150,205,258,187,243,190,266,215,153,227,154,226,152,245,24,
     229,263,190,257,200,260,224,27,229,154,219,274,198,405,211,166,233,47,66,17,125,153,
     237,151,225,147,193,127,214,135,173,112,165,130,223,130,163,112,164,114,167,121,378,114,
     160,113,174,117,211,23,221,261,399,266,390,28,18,367,198,302,28,33,426,31,15,222,96], dtype=np.float64)
D /= 1000

def integrate_over_hemisphere(input_array):
    '''
    input array should go be (nfreq by nangle), with the angles range from 0 to 70 degrees
    '''
    n_angles = 20
    angles = np.arange(0, n_angles)*3*np.pi/180
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

all_refl, all_tran = compute_angular_spectral_tran_refl(D, material_spec)
with open('save_reflectance_spectrum.npy', 'wb') as f:
    np.save(f, all_refl)
all_refl_integrated = integrate_over_hemisphere(all_refl)
all_tran_integrated = integrate_over_hemisphere(all_tran)
emissivity_integrated = integrate_over_hemisphere(emissivity)
F = 0.95

emissivity_eff = emissivity_integrated * (F * all_tran_integrated / (1-F**2 * all_refl_integrated*(1-emissivity_integrated))
                                         + (1-F)*(1-F*all_refl_integrated)/(1 - F**2 * all_refl_integrated*(1-emissivity_integrated))
                                         )


#TODO : code the function to find one from the other using eff emissivity

measured_spectral_emittance = planck_spectrum * emissivity_eff

luminous_efficiency = np.trapz(measured_spectral_emittance*V_luminosity)/np.trapz(measured_spectral_emittance)

print('Luminous efficiency', luminous_efficiency)

with open('save_optimized_stack_emissive_power.npy', 'wb') as f:
    np.save(f, measured_spectral_emittance)