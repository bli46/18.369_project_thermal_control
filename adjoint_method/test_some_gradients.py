import adjoint_2d
import meep as mp
from autograd import numpy as np
import nlopt
import timeout
from autograd.differential_operators import value_and_grad
import sys
#np.set_printoptions(threshold=sys.maxsize)
np.random.seed(168)


n_SiO2 = 1.46
n_Al2O3 = 1.68
n_Ta2O5 = 2.09
n_TiO2 = 2.35
SiO2 = mp.Medium(epsilon=n_SiO2**2)
Al2O3 = mp.Medium(epsilon=n_Al2O3**2)
Ta2O5 = mp.Medium(epsilon=n_Ta2O5**2)
TiO2 = mp.Medium(epsilon=n_TiO2**2)

##################
# Set parameters #
##################
# Set minimum and maximum epsilon we can hit
eps_low = n_SiO2**2
eps_high = n_TiO2**2

# set angle and wavelength
nfreq = 36
wvl_min = 0.40
wvl_max = 2.25
wavelengths = np.linspace(wvl_min, wvl_max, num=nfreq)

###############
# Import data #
###############
with open('save_tungsten_emissivity.npy', 'rb') as f:
    bare_emissivity = np.load(f)
assert bare_emissivity.shape[0] == nfreq
with open('save_planck_spectral.npy', 'rb') as f:
    planck_spectrum = np.load(f)
assert bare_emissivity.shape[0] == nfreq
with open('save_luminosity_function.npy', 'rb') as f:
    V_luminosity = np.load(f)
assert V_luminosity.shape[0] == nfreq

###################
# Write functions #
###################
def refl_tran_from_ez(ez, ez_norm):
    '''
    Computes reflect and transmitted flux given a field
    '''
    nx, ny = ez.shape
    dpml = 1.
    pad = 2.
    res = 40
    npml = dpml * res
    npad = pad * res
    refl_norm_field = ez_norm[int(npml + npad/2), ny//2]
    tran_norm_field = ez_norm[int(nx - npml - npad/2), ny//2]
    refl_field = ez[int(npml + npad/2), ny//2]
    tran_field = ez[int(nx - npml - npad/2), ny//2]
    refl_field -= refl_norm_field
    reflectance = np.abs(refl_field)**2/np.abs(tran_norm_field)**2
    transmittance = np.abs(tran_field)**2/np.abs(tran_norm_field)**2
    return reflectance, transmittance


def luminous_efficiency_from_ez(ez_collected):
    all_refl = []
    all_tran = []
    for i in range(nfreq):
        reflectance, transmittance = refl_tran_from_ez(ez_collected[:, :, i], collected_ez_normalize[:,:,i])
        all_refl.append(reflectance)
        all_tran.append(transmittance)
    all_refl = np.array(all_refl)
    all_tran = np.array(all_refl)
    all_refl_integrated = all_refl
    all_tran_integrated = all_tran
    emissivity_integrated = bare_emissivity[:, 0]
    F = 0.95
    emissivity_eff = emissivity_integrated * (F * all_tran_integrated / (1-F**2 * all_refl_integrated*(1-emissivity_integrated))
                     + (1-F)*(1-F*all_refl_integrated)/(1 - F**2 * all_refl_integrated*(1-emissivity_integrated)))

    measured_spectral_emittance = planck_spectrum * emissivity_eff

    luminous_efficiency = np.sum(measured_spectral_emittance*V_luminosity)/np.sum(measured_spectral_emittance)
    return luminous_efficiency

res = 40
n = mp.divide_parallel_processes(nfreq)
dpml = 1.
pad = 2.
wvl = wavelengths[n]
#angle = angle_num * 3
fcen = 1./wvl
#theta_in = 10 * np.pi/180
structure_length = 2
structure_width = 1
ny = int((structure_width + 2*dpml)*res)
nx = int((structure_length + 2*dpml + 2*pad)*res)
xndgn = (structure_length * res) // 2
dgn = np.zeros((nx, ny))
#yndgn = (structure_width*res)//2
dgn[nx//2-xndgn:nx//2+xndgn,:]=1.
npml = int(dpml*res)
npad = int(pad*res)

#####################
# Normalization run #
#####################
#print((nx, ny))
eps_normalize = np.ones((nx, ny))
ez_normalize, _, _ = adjoint_2d.compute_fields(eps_normalize, fcen)
collected_ez_normalize = mp.merge_subgroup_data(ez_normalize)
with open('save_norm_fields.npy', 'wb') as f:
    np.save(f, collected_ez_normalize)

ez = np.ones((nx, ny, nfreq))

value_and_grad_lum = value_and_grad(luminous_efficiency_from_ez)
eff, eff_grad = value_and_grad_lum(ez)

dp = .000001
tmp = np.zeros((nx, ny, nfreq))
tmp[:, :, :] = ez[:, :, :]
tmp[int(npml + npad/2), ny//2, 0] += dp
eff_2 = luminous_efficiency_from_ez(tmp)
cent_diff = (eff_2 - eff)/dp
grad_val = eff_grad[int(npml + npad/2), ny//2, 0]

print(cent_diff, grad_val, (cent_diff - grad_val)/cent_diff)