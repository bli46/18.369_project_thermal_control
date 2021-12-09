import meep as mp
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
#from timeout import timeout

##############################################
# Set materials and structure specifications #
##############################################

# indices of refraction at 610 nm from supplementary material of ilic et al
# values validated on the site https://refractiveindex.info/
# for now, assume dispersionless materials
# TODO: check how good of an assumption that is
# in future iterations, use the values on that site for dispersion relations

n_SiO2 = 1.46
n_Al2O3 = 1.68
n_Ta2O5 = 2.09
n_TiO2 = 2.35
SiO2 = mp.Medium(epsilon=n_SiO2**2)
Al2O3 = mp.Medium(epsilon=n_Al2O3**2)
Ta2O5 = mp.Medium(epsilon=n_Ta2O5**2)
TiO2 = mp.Medium(epsilon=n_TiO2**2)

# material specifications

material_spec = [4,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,3,4,2,1,2,4,2,1,4,1,2,3,4,2,1,2,4,2,1,2,4,2,
1,2,4,2,1,2,4,2,1,2,4,2,1,4,3,2,1,2,4,1,2,4,3,2,1,4,2,1,2,4,2,1,2,4,3,1,2,4,2,1,2,4,2,1,4,2,1,2,4,1,4,3,1,2,4,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,
3,4,2,1,2,3,4,2,1,2,4,2,1,2,4,3,2,3,4,2,1,4,2,1,2,4,2,1,4,3,2,1,2,4,2,1,4,1,2,4,2,1,2,4,2,1,2,4,3,2,1,2,4,2,1,2,4,2,1,2,4,2,1,2,4,1,2,4,1,4,1,2,4,1,
4,3,1,4,2,1,4,1,4,1,2,4,2,1,4,2,1,4,1,4,1,2,4,1,4,1,4,1,2,4,1,4,1,4,1,4,2,1,4,1,4,1,2,4,1,2,4,1,2,4,1,4,1,4,1,4,1,2,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,
4,1,4,1,4,1,4,1,4,1,4,1,4,1]

######################
# Set up simulations #
######################

resolution = 80 # pixels/um

# send in light with wavelength from 450 nm to 2250 nm
wvl_min = 0.40
wvl_max = 0.80
#frq_min = 1/wvl_max
#frq_max = 1/wvl_min
#fcen = (frq_min + frq_max)/2
#df = frq_max - frq_min
#angles = np.arange(35, 81, 5)
n_angles = 69
nfreq = 10
wavelengths = np.linspace(wvl_min, wvl_max, num=nfreq)
#frequencies = 1./wavelengths

#@timeout(300)
def perform_tasks(n, wvl, D):
    fcen = 1./wvl
    df = .01*fcen
    # set pml
    dpml = 3.
    pml_layers = [mp.PML(dpml, direction=mp.X)]

    # pad up structure and make the cell
    pad = 4.
    structure_length = sum(D)
    y_length = 0.
    sx = structure_length + 2*dpml + 2*pad
    sy = y_length
    cell = mp.Vector3(sx,sy)

    # make the geometry according to the specifications
    geometry_no_stack = []
    geometry_stack = [None]*len(D)
    location_in_cell = -structure_length/2 
    for i, element_width in enumerate(D):
        if material_spec[i] == 1:
            material = SiO2
        elif material_spec[i] == 2:
            material = Al2O3
        elif material_spec[i] == 3:
            material = Ta2O5
        elif material_spec[i] == 4:
            material = TiO2
        geometry_stack[i] = mp.Block(mp.Vector3(element_width, mp.inf),
                                     center=mp.Vector3(location_in_cell + element_width/2, 0),
                                     material=material)
        location_in_cell += element_width

    # set up the oblique-angle sources in parallel
    src_pos = -0.5*sx + dpml
    src_pt = mp.Vector3(src_pos, 0)
    #n_angles = 45
    #n = mp.divide_parallel_processes(n_angles)
    #n = n_angles - n - 1 # we use reverse order to see the print output of the slowest simulation

    #n = mp.divide_parallel_processes(n_angles)
    theta_in = n*np.pi/180
    n0 = 1. # air
    k = mp.Vector3(fcen*n0).rotate(mp.Vector3(z=1), theta_in)
    def pw_amp(k,x0):
        def _pw_amp(x):
            return cmath.exp(1j*2*math.pi*k.dot(x+x0))
        return _pw_amp
    sources = [ mp.Source(mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
                          # need is_integrated=True because source extends through pml
                          component=mp.Ez, 
                          center=mp.Vector3(src_pos,0),
                          size=mp.Vector3(0, sy), 
                          amp_func=pw_amp(k,src_pt)) ]

    ######################################################
    # First run for normalization: no interference stack #
    ######################################################

    # make simulation object with stack-free geometry

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry_no_stack,
                        sources=sources,
                        resolution=resolution,
                        k_point=k)

    # set up fluxes for reflection subtraction and normalization
    refl_pos = -0.5*sx + dpml + pad/2
    refl_fr = mp.FluxRegion(center=mp.Vector3(refl_pos,0), size=mp.Vector3(0, y_length/2))
    refl = sim.add_flux(fcen, 0, 1, refl_fr)
    tran_pos = 0.5*sx - dpml - pad/2
    tran_fr = mp.FluxRegion(center=mp.Vector3(tran_pos,0), size=mp.Vector3(0,y_length/2,0))
    tran = sim.add_flux(fcen, 0, 1, tran_fr)

    # normalization run
    pt = mp.Vector3(-0.5*sx+dpml+pad/4, 0)
    dT = 10.*1/fcen
    decay = 1e-6
    sim.run(until_after_sources=mp.stop_when_fields_decayed(dT,mp.Ez,pt,decay))
    norm_refl_data = sim.get_flux_data(refl)
    norm_tran_flux = mp.get_fluxes(tran)
    sim.reset_meep()
    ########################################
    # Second run: with interference stack #
    #######################################

    # make simulation object with interference stack
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry_stack,
                        sources=sources,
                        resolution=resolution,
                        k_point=k)


    # set up flux to find the reflectance
    refl = sim.add_flux(fcen, 0, 1, refl_fr)
    tran = sim.add_flux(fcen, 0, 1, tran_fr)
    sim.load_minus_flux_data(refl, norm_refl_data)

    # second run
    sim.run(until_after_sources=mp.stop_when_fields_decayed(dT,mp.Ez, pt, decay))
    stack_refl_flux = mp.get_fluxes(refl)
    stack_tran_flux = mp.get_fluxes(tran)
    flux_freqs = mp.get_flux_freqs(refl)

    # normalize flux and convert output to numpy arrays
    normalized_refl_flux = -np.array(stack_refl_flux)/np.array(norm_tran_flux)
    normalized_tran_flux = np.array(stack_tran_flux)/np.array(norm_tran_flux)
    flux_wvl = 1./np.array(flux_freqs)
    sim.reset_meep()
    ##########################################
    # Merge data from the parallel processes #
    ##########################################

    #flux_wvl_merged = mp.merge_subgroup_data(flux_wvl)
    #normalized_refl_flux_merged = mp.merge_subgroup_data(normalized_refl_flux)

    # move the axes so they're in angle, frequency order and flip the angles to be ascending
    #flux_wvl_final = np.flip(np.moveaxis(flux_wvl_merged, -1, 0), axis=0)
    #normalized_refl_flux_final = np.flip(np.moveaxis(normalized_refl_flux_merged, -1, 0), axis=0)
    return normalized_refl_flux[0], normalized_tran_flux[0]

def compute_angular_spectral_tran_refl(D):
    n = mp.divide_parallel_processes(n_angles*nfreq)
    wvl = wavelengths[n//n_angles]
    angle = n - n_angles * (n//n_angles)
    refl_by_wvl, tran_by_wvl = perform_tasks(angle, wvl, D)
    all_refl = mp.merge_subgroup_data(refl_by_wvl)
    all_tran = mp.merge_subgroup_data(tran_by_wvl)
    all_refl = np.reshape(all_refl, (nfreq, n_angles), order='C')
    all_tran = np.reshape(all_tran, (nfreq, n_angles), order='C')
    return all_refl, all_tran

