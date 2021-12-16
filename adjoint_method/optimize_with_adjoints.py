import adjoint_2d
import meep as mp
from autograd import numpy as np
import nlopt
import timeout
from autograd.differential_operators import value_and_grad
import sys
import matplotlib.pyplot as plt
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

class AdamOptim():
    def __init__(self, eta1=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8, crop=1., m_dw=0, v_dw=0):
        self.m_dw, self.v_dw = m_dw, v_dw
        #self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta1 = eta1
        self.crop = crop
        #self.eta2 = eta2
    def update(self, t, w, dw):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        # *** biases *** #
        #self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        # *** biases *** #
        #self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        #m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        #v_db_corr = self.v_db/(1-self.beta2**t)

        ## update weights and biases
        w = w - self.eta1*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))*self.crop
        #b = b - self.eta2*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return w#, b
    
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
    with open('save_adjoint_optimized_spectral_emittance.npy', 'wb') as f:
        np.save(f, measured_spectral_emittance)
    luminous_efficiency = np.sum(measured_spectral_emittance*V_luminosity)/np.sum(measured_spectral_emittance)
    return luminous_efficiency


def luminous_efficiency(x, grad):
    '''
    Function we use for nlopt to compute luminous efficiency. returns our gradient.
    '''
    global global_iter
    global global_efficiency
    pad_width = (nx - 2*xndgn)//2
    eps = np.broadcast_to(np.expand_dims(np.pad(x, (pad_width, pad_width),'constant', constant_values=(1, 1)), axis=1), (nx, ny))
    plt.imshow(eps)
    plt.savefig('eps_fig.png')
    plt.close()
    n = mp.divide_parallel_processes(nfreq)
    wvl = wavelengths[n]
    fcen = 1./wvl
    ez, dt, df = adjoint_2d.compute_fields(eps, fcen)
    ez_collected = mp.merge_subgroup_data(ez)
    
    
    global_iter += 1
    print('Iteration', global_iter)
    if grad.size > 0:
        value_and_grad_lum_eff = value_and_grad(luminous_efficiency_from_ez)
        luminous_efficiency, grad_lum_eff = value_and_grad_lum_eff(ez_collected)
        wvl = wavelengths[n]
        fcen = 1./wvl
        grad_ez = grad_lum_eff
        grad_for_one_run = adjoint_2d.adjoint_compute(grad_ez[:,:,n], fcen, df, dt, nx, ny, eps, ez_collected[:, :, n])
        grad_for_all = mp.merge_subgroup_data(grad_for_one_run)
        grad_summed = np.sum(grad_for_all, axis=-1)
        grad[:] = np.sum(grad_summed[:, :], axis=1)[nonzero_positions][:]
        print('Norm of gradient:', np.linalg.norm(grad))
    else:
        luminous_efficiency = luminous_efficiency_from_ez(ez_collected)
    
    global_efficiency.append(luminous_efficiency)
    
    print('Luminous efficiency:', luminous_efficiency)
    
    with open('save_5_lam_run_2_adam_iter_' + str(global_iter) + '.npy', 'wb') as f:
        np.save(f, global_efficiency)
        np.save(f, eps)
    return luminous_efficiency

global_efficiency = []

res = 40
n = mp.divide_parallel_processes(nfreq)
dpml = 3.
pad = 2.
wvl = wavelengths[n]
#angle = angle_num * 3
fcen = 1./wvl
#theta_in = 10 * np.pi/180
structure_length = 5
structure_width = 7
ny = int((structure_width)*res)
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
print((nx, ny))
eps_normalize = np.ones((nx, ny))
#ez_normalize, _, _ = adjoint_2d.compute_fields(eps_normalize, fcen)
#collected_ez_normalize = mp.merge_subgroup_data(ez_normalize)
with open('save_norm_fields.npy', 'rb') as f:
    collected_ez_normalize=np.load(f)

############
# Optimize #
############

#eps_initial = dgn*np.ones((nx,ny)) * ((eps_low+eps_high-2)/2) + 1.
with open('save_5_lam_run_2_adam_iter_14.npy', 'rb') as f:
    np.load(f)
    eps_initial =  np.load(f)
nonzero_positions = np.nonzero(dgn[:, ny//2])
x_initial = eps_initial[:, ny//2][nonzero_positions]
global_iter = 0
luminous_efficiency(x_initial, np.array([]))
exit()
############################
# Check finite differences #
############################
#dp = 0.0001
#grad = np.zeros(2*xndgn)
#lum_eff = luminous_efficiency(x_initial, np.array([]))
#print(grad)
#temp = np.zeros(2*xndgn)
#temp[:] = x_initial[:]
#temp[0] += dp
#lum_eff_2 = luminous_efficiency(temp, np.array([]))
#print((lum_eff_2 - lum_eff)/(dp))
#temp[:] = x_initial[:]
#temp[1] += dp
#lum_eff_3 = luminous_efficiency(temp, np.array([]))
#print((lum_eff_3 - lum_eff)/(dp))

#print((lum_eff_3 - lum_eff)/(dp*grad[1] ))

############
# Optimize #
############
global_iter = 0
#opt = nlopt.opt(nlopt.GN_DIRECT_L, x_initial.size)
#opt.set_lower_bounds(eps_low*np.ones(x_initial.size))
#opt.set_upper_bounds(eps_high*np.ones(x_initial.size))
#opt.set_max_objective(luminous_efficiency)
#opt.set_maxeval(100)
#x = opt.optimize(x_initial)
#maxf = opt.last_optimum_value()
#print("optimum at ", x[0], x[1])
#print("maximum value = ", maxf)
#print("result code = ", opt.last_optimize_result())
adam = AdamOptim(eta1=1e-2)
x = x_initial
for t in range(1, 400):
    grad = np.zeros((2*xndgn,))
    luminous_efficiency(x, grad)
    grad = -grad # negate for maximization rather than minimization
    x = adam.update(t,w=x,dw=grad)
    x = np.clip(x, eps_low, eps_high)

with open('save_design_region.npy', 'wb') as f:
    np.save(f, x)