import numpy as np
import meep as mp

def sym(arrx_in,arry_in,arrz_in):

    arrx = arrx_in.copy()
    arry = arry_in.copy()
    arrz = arrz_in.copy()
    
    arrx = np.concatenate((np.zeros((arrx.shape[0],1)),arrx,np.flip(arrx,axis=1)[:,1:]),axis=1)
    arrx = np.concatenate((np.flip(arrx,axis=0),arrx),axis=0)

    arry = np.concatenate((np.zeros((1,arry.shape[1])),np.flip(arry,axis=0)[:-1,:],arry),axis=0)
    arry = np.concatenate((arry,np.flip(arry,axis=1)),axis=1)

    arrz = np.concatenate((arrz,np.flip(arrz,axis=1)[:,1:]),axis=1)
    arrz = np.concatenate((np.flip(arrz,axis=0)[:-1,:],arrz),axis=0)
    arrz = np.concatenate((np.zeros((arrz.shape[0],1)),arrz),axis=1)
    arrz = np.concatenate((np.zeros((1,arrz.shape[1])),arrz),axis=0)

    return arrx,arry,arrz
    
def dat2pos(r,datax,datay,dataz,nx,ny,du):

    decplace=5

    x0=np.around(-nx*du/2.,decplace)
    y0=np.around(-ny*du/2.,decplace)
    rx=np.around(r.x,decplace)
    ry=np.around(r.y,decplace)
    jx=np.around( (rx-x0)/du, decplace)
    jy=np.around( (ry-y0)/du, decplace)
    ix = int(np.floor( jx ))
    iy = int(np.floor( jy ))
    dx = jx-float(ix)
    dy = jy-float(iy)

    if dx>0.001 and dy<0.001 and 0<=iy<ny and 0<=ix<nx:
        ic=0
    elif dx<0.001 and dy>0.001 and 0<=ix<nx and 0<=iy<ny:
        ic=1
    elif dx<0.001 and dy<0.001 and 0<=ix<nx and 0<=iy<ny:
        ic=2
    else:
        ic=-1

    if ic==0:
        ret = datax[ix,iy]
    elif ic==1:
        ret = datay[ix,iy]
    elif ic==2:
        ret = dataz[ix,iy]
    else:
        ret = 0.

    return ret
    
def fdtd(epsx,epsy,epsz,
         jx,jy,jz,
         fcen,df,courant,
         nx,ny,du,npml,
         monitor_comp=0,
         src_intg=True,
         xphase=0,yphase=0):

    dpml=npml*du
    Lx=nx*du
    Ly=ny*du

    resolution=1./du

    mtr_dt=10./fcen
    if monitor_comp==0:
        mtr_c=mp.Ex
    if monitor_comp==1:
        mtr_c=mp.Ey
    if monitor_comp==2:
        mtr_c=mp.Ez
    mtr_r=mp.Vector3(0,0,0)
    mtr_tol=1e-8
    
    def epsfun(r):
        return dat2pos(r,epsx,epsy,epsz,nx,ny,du)

    def jfun(r):
        return dat2pos(r,jx,jy,jz,nx,ny,du)

    cell = mp.Vector3(Lx,Ly,0)
    pml_layers = [mp.PML(dpml)]
    sources = [ mp.Source(mp.GaussianSource(fcen,fwidth=df,is_integrated=src_intg),
                          component=mp.Ex,
                          center=mp.Vector3(0,0,0),
                          size=cell,
                          amp_func=jfun),
                mp.Source(mp.GaussianSource(fcen,fwidth=df,is_integrated=src_intg),
                          component=mp.Ey,
                          center=mp.Vector3(0,0,0),
                          size=cell,
                          amp_func=jfun),
                mp.Source(mp.GaussianSource(fcen,fwidth=df,is_integrated=src_intg),
                          component=mp.Ez,
                          center=mp.Vector3(0,0,0),
                          size=cell,
                          amp_func=jfun) ]

    if xphase!=0 and yphase!=0:
        syms = [mp.Mirror(mp.X,xphase),mp.Mirror(mp.Y,yphase)]
    else:
        syms = []
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        epsilon_func=epsfun,
                        eps_averaging=False,
                        sources=sources,
                        resolution=resolution,
                        force_complex_fields=True,
                        symmetries=syms,
                        Courant=courant)

    sim.init_sim()

    dft_vol = mp.Volume(center=mp.Vector3(0,0,0), size=cell)
    dft_obj = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez], fcen, df,1, where=dft_vol, yee_grid=True)

    sim.run( until_after_sources=mp.stop_when_fields_decayed(mtr_dt, mtr_c, mtr_r, mtr_tol) )

    ex = sim.get_dft_array(dft_obj, mp.Ex, 0)
    ey = sim.get_dft_array(dft_obj, mp.Ey, 0)
    ez = sim.get_dft_array(dft_obj, mp.Ez, 0)

    dt = sim.fields.dt

    sim.reset_meep()
    del sim

    return ex, ey, ez, dt