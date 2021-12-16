import numpy as np
import solver2d as s2
import adjoint_2d
import meep as mp
import sys
np.set_printoptions(threshold=sys.maxsize)


def sumE2(eps,gdat,
          fcen,df,courant,
          nx,ny,du,npml):

    jx=np.zeros((nx,ny))
    jy=np.zeros((nx,ny))
    jz=np.zeros((nx,ny))
    jz[:,npml+1]=1./du

    p=np.zeros((nx,ny))
    p[npml:nx-npml,ny-npml-1]=1.0
    
    ex, ey, ez,dt = s2.fdtd(eps,eps,eps,
                          jx,jy,jz,
                          fcen,df,courant,
                          nx,ny,du,npml, 
                          monitor_comp=2,
                          src_intg=True)
    print(ez[:, ny-npml-1])
    ret = np.sum(p * np.abs(ez)**2)

    omega=2.*np.pi*fcen
    iw = (1.0 - np.exp(-1j*omega*dt)) * (1.0/dt)
    w2 = omega*omega
    t0 = 5.0/df
    expfac = np.exp(1j*omega*t0)
    adjz = 2.*p*np.conj(ez) * (df/(expfac*iw))
    
    if gdat.size>0:

        ux,uy,uz,dt = s2.fdtd(eps,eps,eps,
                              np.zeros((nx,ny)),
                              np.zeros((nx,ny)),
                              adjz,
                              fcen,df,courant,
                              nx,ny,du,npml,
                              monitor_comp=2,
                              src_intg=False)

        grad = np.real( w2 * (uz*ez) )

        gdat[:,:] = grad[:,:]

    return ret

np.set_printoptions(linewidth=100000)

nx,ny=200,200
du=0.025
npml=80

ndgn=25
dgn=np.zeros((nx,ny))
dgn[:,ny//2-ndgn:ny//2+ndgn]=1.0 #design region is a box with widths of +/-ndgn pixels around the center

fcen=0.94
df=0.1
courant=0.5

np.random.seed(1234)

ndat=100
dp=0.01
tmp=np.zeros((nx,ny))
gdat=np.zeros((nx,ny))
for i in range(ndat):
    #eps=np.random.uniform(low=0.0,high=1.5,size=(nx,ny))*dgn + 1.0
    eps = np.ones((nx, ny))
    obj = sumE2(eps,gdat,
                fcen,df,courant,
                nx,ny,du,npml)

    chkx = np.random.randint(low=nx//2-ndgn,high=nx//2+ndgn)
    chky = np.random.randint(low=ny//2-ndgn,high=ny//2+ndgn)

    tmp[:,:]=eps[:,:]
    tmp[chkx,chky] -= dp
    obj0 = sumE2(tmp,np.array([]),
                 fcen,df,courant,
                 nx,ny,du,npml)

    tmp[:,:]=eps[:,:]
    tmp[chkx,chky] += dp
    obj1 = sumE2(tmp,np.array([]),
                 fcen,df,courant,
                 nx,ny,du,npml)

    cendiff = (obj1-obj0)/(2.*dp)

    grad = gdat[chkx,chky] 
    print("check: {} {} {}".format(grad,cendiff,(grad-cendiff)/grad))
    
    