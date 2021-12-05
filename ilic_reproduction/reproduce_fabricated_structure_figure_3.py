import meep as mp
import math
import cmath
import numpy as np

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
n1 = Ta2O5
n2 = SiO2
D = [18,47,156,212,178,23,51,224,150,205,258,187,243,190,266,215,153,227,154,226,152,245,24,
     229,263,190,257,200,260,224,27,229,154,219,274,198,405,211,166,233,47,66,17,125,153,
     237,151,225,147,193,127,214,135,173,112,165,130,223,130,163,112,164,114,167,121,378,114,
     160,113,174,117,211,23,221,261,399,266,390,28,18,367,198,302,28,33,426,31,15,222,96]

######################
# Set up simulations #
######################

resolution = 40 # pixels/um

# send in light with wavelength from 450 nm to 2250 nm
wvl_min = 0.450
wvl_max = 2.250
frq_min = 1/wvl_max
frq_max = 1/wvl_min
fcen = (frq_min + frq_max)/2
df = frq_max - frq_min
nfreq = 100

# set pml
dpml = 2.
pml_layers = [mp.PML(dpml, direction=mp.X)]

# pad up structure and make the cell
pad = 4.
structure_length = sum(D)/1000
y_length = 5.
sx = structure_length + 2*dpml + 2*pad
sy = y_length + 2*dpml
cell = mp.Vector3(sx,sy)

# make the geometry according to the specifications
geometry_no_stack = []
geometry_stack = [None]*len(D)
location_in_cell = -0.5*sx + dmpl + pad 
for i, element_width in enumerate(D):
    if i%2 == 0:
        material = Ta2O5
    else:
        material = SiO2
    geometry_stack[i] = mp.Block(mp.Vector3(element_width, mp.inf),
                                 center=mp.Vector3(location_in_stack + element_width/2, 0),
                                 material=material)
    location_in_cell += element_width

# set up the oblique-angle sources in parallel
src_pos = -0.5*sx + dpml
src_pt = mp.Vector3(src_pos, 0)
n = mp.divide_parallel_processes(80)
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
nfreq = 100
refl_pos = -0.5*sx + dpml + pad/2
refl_fr = mp.FluxRegion(center=mp.Vector3(refl_pos,0), size=mp.Vector3(0, y_length/2))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran_pos = 0.5*sx - dpml - pad/2
tran_fr = mp.FluxRegion(center=mp.Vector3(tran_pos,0), size=mp.Vector3(0,y_length/2,0))
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# normalization run
pt = mp.Vector3(0.5*sx-dpml-pad/4, 0)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ez,pt,1e-6))
norm_refl_data = sim.get_flux_data(refl)
norm_tran_flux = mp.get_fluxes(tran)

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
refl = sim.add_flux(fcen, df, nfreq, refl_fr)
sim.load_minus_flux_data(refl, norm_refl_data)

# second run
pt = mp.Vector3(0.5*sx-dpml-pad/4, 0)
sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ez, pt, 1e-6))
qw_refl_flux = mp.get_fluxes(refl)
flux_freqs = mp.get_flux_freqs(refl)

# normalize fluxes
normalized_flux = 

##########################################
# Merge data from the parallel processes #
##########################################

flux_freqs_merged = mp.merge_parallel_processes(flux_freqs)

with open()