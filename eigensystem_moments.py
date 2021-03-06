# Copyright (C) 2021 Sherwood Richers & Sam Flynn
# This file is part of neutrino_linear_stability <https://github.com/srichers/neutrino_linear_stability>.
#
# neutrino_linear_stability is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# neutrino_linear_stability is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with neutrino_linear_stability.  If not, see <http://www.gnu.org/licenses/>.

#======================================================================#

# set the number of openmp threads to 1.
# Do not change this for multithreading
# instead, change the "nthreads" variable below
import os
os.environ[       "OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ[  "OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ[       "MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ[   "NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import time
import numpy as np
import h5py
import fileinput
import sys
import glob
import scipy.linalg
import copy
import multiprocessing as mp
np.set_printoptions(linewidth=200)

#===========#
# constants #
#===========#
h = 6.6260755e-27 # erg s
hbar = h/(2.*np.pi) # erg s
c = 2.99792458e10 # cm/s
MeV = 1.60218e-6 # erg
eV = MeV/1e6 # erg
GF_GeV2 = 1.1663787e-5 # GeV^-2
GF = GF_GeV2 / (1000*MeV)**2 * (hbar*c)**3 # erg cm^3
Mp = 1.6726219e-24 # g

#========#
# inputs #
#========#
flavor_trace_chi = True  # in calculation of P. Always true in Cij
flavor_trace_fhat = True # in calculation of P. Always true in Cij
dm2 = 2.4e-3 * eV**2 # erg
nthreads = 2
rho = 0 # g/ccm
Ye = 0.5
average_energy = 50*MeV # erg

#numb_k = 10
#nphi_at_equator = 16
#min_ktarget_multiplier = 1e-1
#max_ktarget_multiplier = 1e1
#Nee    = 4.89e32
#Neebar = 4.89e32
#Nxx    = 0
#Nxxbar = 0
#Fee    = np.array([0,0,  1./3]) * Nee
#Feebar = np.array([0,0, -1./3]) * Neebar
#Fxx    = np.array([0,0,  0   ])
#Fxxbar = np.array([0,0,  0   ])

# 90Degree case
#numb_k = 200
#nphi_at_equator = 64
#min_ktarget_multiplier = 1
#max_ktarget_multiplier = 3
#Nee    = 4.89e32
#Neebar = 4.89e32
#Nxx    = 0
#Nxxbar = 0
#Fee    = np.array([0, 0   , 1./3]) * Nee
#Feebar = np.array([0, 1/3., 0   ]) * Neebar
#Fxx    = np.array([0, 0   , 0   ])
#Fxxbar = np.array([0, 0   , 0   ])

# TwoThirds case
#numb_k = 200
#nphi_at_equator = 64
#min_ktarget_multiplier = 0.1
#max_ktarget_multiplier = 3
#Nee    = 4.89e32
#Neebar = 4.89e32 * 2./3.
#Nxx    = 0
#Nxxbar = 0
#Fee    = np.array([0,0,  0   ]) * Nee
#Feebar = np.array([0,0, -1./3]) * Neebar
#Fxx    = np.array([0,0,  0   ])
#Fxxbar = np.array([0,0,  0   ])

# NSM case
numb_k = 20
nphi_at_equator = 16
min_ktarget_multiplier = 0.01
max_ktarget_multiplier = 5
Nee    = 1.421954234999705e+33
Neebar = 1.9146237131657563e+33
Nxx    = 1.9645407875568215e+33
Nxxbar = Nxx
Fee    = np.array([0.0974572,    0.04217632, -0.13433261]) * Nee
Feebar = np.array([ 0.07237959,  0.03132354, -0.3446878 ]) * Neebar
Fxx    = np.array([-0.02165833,  0.07431613, -0.53545951]) * Nxx
Fxxbar = Fxx

#==========================#
# Construct N and F arrays #
#==========================#
N = np.array([[Nee   , Nxx   ],
              [Neebar, Nxxbar]]) # [nu/nubar, flavor]
N_FT = np.sum(N, axis=1)         # [nu/nubar]
print("\nN:")
print(N)

F = np.array([[Fee   , Fxx   ],
              [Feebar, Fxxbar]]) # [nu/nubar, flavor, i]
F_FT = np.sum(F, axis=1)         # [nu/nubar, i]
print("\nF:")
print(F)

#====================================#
# flux factors and flux unit vectors #
#====================================#
Fmag    = np.sqrt(np.sum(F**2   , axis=2)) # [nu/nubar, flavor]
Fmag_FT = np.sqrt(np.sum(F_FT**2, axis=1)) # [nu/nubar]

fluxfac    = Fmag    / N    # [nu/nubar, flavor]
fluxfac_FT = Fmag_FT / N_FT # [nu/nubar]
fluxfac[np.where(N==0)] = 0
fluxfac_FT[np.where(N_FT==0)] = 0

Fhat    = F    / Fmag[:,:,np.newaxis]  # [nu/nubar, flavor, i]
Fhat_FT = F_FT / Fmag_FT[:,np.newaxis] # [nu/nubar, i]
Fhat[np.where(Fhat!=Fhat)] = 0
Fhat_FT[np.where(Fhat_FT!=Fhat_FT)] = 0

#=========================#
# interpolation parameter #
#=========================#
def get_chi(ff):
    return 1./3. + 2./15. * ff**2 * (3. - ff + 3.*ff**2)
chi_FT = get_chi(fluxfac_FT)   # [nu/nubar]
if flavor_trace_chi:
    chi = chi_FT[:,np.newaxis]
else:
    chi = get_chi(fluxfac) # [nu/nubar, flavor]

#===============================#
# thick part of pressure tensor #
#===============================#
Ptilde_thick_FT = np.identity(3)[np.newaxis,:,:] / 3. # [nu/nubar, i, j]
Ptilde_thick = Ptilde_thick_FT[:,np.newaxis,:,:]      # [nu/nubar, flavor, i, j]

#==============================#
# thin part of pressure tensor #
#==============================#
Ptilde_thin_FT = Fhat_FT[:,:,np.newaxis] * Fhat_FT[:,np.newaxis,:] # [nu/nubar, i, j]
if flavor_trace_fhat:
    Ptilde_thin = Ptilde_thin_FT[:,np.newaxis,:,:]
else:    
    Ptilde_thin = Fhat[:,:,:,np.newaxis] * Fhat[:,:,np.newaxis,:]  # [nu/nubar, flavor, i, j]

#====================================#
# interpolate between thick and thin #
#====================================#
def get_Ptilde(Ptilde_thick, Ptilde_thin, chi):
    return (3.*chi-1.)/2.*Ptilde_thin + 3.*(1.-chi)/2.*Ptilde_thick
Ptilde_FT = get_Ptilde(Ptilde_thick_FT, Ptilde_thin_FT, chi_FT[:,np.newaxis,np.newaxis]) # [nu/nubar, i, j]
Ptilde    = get_Ptilde(Ptilde_thick   , Ptilde_thin   , chi[:,:,np.newaxis,np.newaxis] ) # [nu/nubar, flavor, i, j]

#==================================#
# construct actual pressure tensor #
#==================================#
P_FT = Ptilde_FT * N_FT[:,np.newaxis,np.newaxis] # [nu/nubar, i, j]
P    = Ptilde    * N[:,:,np.newaxis,np.newaxis] # [nu/nubar, flavor, i, j]
print("\nP:")
print(P)
print(np.shape(P))
print(np.shape(P_FT))


#=====================================#
# Set Cij to the flavor-traced Ptilde #
#=====================================#
C = Ptilde_FT # [nu/nubar, i, j]
print("C=")
print(C)

#----------
# get vmatter and phis (all in ergs)
#----------
mu_N = np.sqrt(2.) * GF * (N[:,0]     - N[:,1]    ) # [nu/nubar]
mu_F = np.sqrt(2.) * GF * (F[:,0,:]   - F[:,1,:]  ) # [nu/nubar, i]
mu_P = np.sqrt(2.) * GF * (P[:,0,:,:] - P[:,1,:,:]) # [nu/nubar, i, j]
Ve   = np.sqrt(2.) * GF * (rho * Ye / Mp)
phi0 = mu_N[0] - mu_N[1]
phi1 = mu_F[0] - mu_F[1] # [i]
phi1mag = np.sqrt(np.sum(phi1**2))

print("\nphi0:")
print(phi0)
print("\nphi1:")
print(phi1)
print("\nmu_N:")
print(mu_N)
print("\nmu_F:")
print(mu_F)
print("\nmu_P:")
print(mu_P)


# mu parts of the stability matrix [nu/nubar, i, j] where 0 is N and 1-3 are F
mu = np.zeros((2,4,4))
mu[:, 0  , 0  ] =  mu_N
mu[:, 0  , 1:4] = -mu_F
mu[:, 1:4, 0  ] =  mu_F
mu[:, 1:4, 1:4] = -mu_P
print("\nmu:")
print(mu)

# Stability matrix without k term
S_nok_RA = mp.RawArray('d',8*8)
S_nok = np.frombuffer(S_nok_RA).reshape((8,8))
#S_nok = np.zeros((8,8))
S_nok[0:4, 0:4] =  mu[0]
S_nok[0:4, 4:8] = -mu[0]
S_nok[4:8, 0:4] =  mu[1]
S_nok[4:8, 4:8] = -mu[1]
print("\nS_nok:")
print(S_nok)

# build the kprime grid
dkprime = phi1mag * (max_ktarget_multiplier - min_ktarget_multiplier)/numb_k / (hbar*c)
kprime_mag_grid = phi1mag * np.arange(min_ktarget_multiplier, max_ktarget_multiplier, dkprime)
#kprime_mag_grid = phi1mag * np.geomspace(min_ktarget_multiplier,max_ktarget_multiplier,num=numb_k,endpoint=True)
print()
print("Using",len(kprime_mag_grid),"kprime_mag points between",kprime_mag_grid[0]/(hbar*c),"and",kprime_mag_grid[-1]/(hbar*c),"cm^-1")
print("dkprime =",dkprime,"cm^-1") # 


# build uniform covering of unit sphere
dtheta = np.pi * np.sqrt(3) / nphi_at_equator
theta = 0
phi0 = 0
direction_grid = []
while theta < np.pi/2.:
    nphi = nphi_at_equator if theta==0 else int(nphi_at_equator * np.cos(theta) + 0.5)
    dphi = 2.*np.pi/nphi
    if nphi==1:
        theta = np.pi/2.
    for iphi in range(nphi):
        phi = phi0 + iphi * dphi
        x = np.cos(theta)*np.cos(phi)
        y = np.cos(theta)*np.sin(phi)
        z = np.sin(theta)
        direction_grid.append([x,y,z])
        if theta>0:
            direction_grid.append([-x, -y, -z])
    theta += dtheta
    phi0 += 0.5*dphi
direction_grid = np.array(direction_grid)
ndir = np.shape(direction_grid)[0]
print("Using",ndir,"k directions")

# construct full kprime grid
kprime_grid = [kprime_mag*direction for direction in direction_grid for kprime_mag in kprime_mag_grid]
kprime_grid.append([0,0,0])
kprime_grid = np.array(kprime_grid)

#=================================#
# do the real work for a single k #
#=================================#
# k:erg output:erg
def eigenvalues_single_k(kprime):
    # complete the stability matrix
    S_nok = np.frombuffer(S_nok_RA).reshape((8,8))
    S = copy.deepcopy(S_nok)
    S[0  , 1:4] -= kprime
    S[4  , 5:8] -= kprime
    S[1:4, 0  ] -= np.tensordot(kprime, C[0], axes=1)
    S[5:8, 4  ] -= np.tensordot(kprime, C[1], axes=1)

    # find the eigenvalues
    return np.linalg.eigvals(S)


#===========================================#
# do all eigenvalue calculations for a file #
#===========================================#
def compute_all_eigenvalues():
    #----------
    # loop over radii
    #----------
    eigenvalues = []

    #----------
    # loop over k points
    #----------
    start_time = time.time()
    #for kprime in kprime_grid:
    #    eigenvalues.append(eigenvalues_single_k(kprime))
    with mp.Pool(processes=nthreads) as pool:
        eigenvalues = pool.map(eigenvalues_single_k, kprime_grid)
    end_time = time.time()
    
    #----------
    # write info to screen
    #----------
    print()
    R = np.real(eigenvalues)
    I = np.imag(eigenvalues)
    print("time="+"{:.2e}".format(end_time-start_time)+"s.")
    print("Re(Omega_prime) within [",np.min(R),np.max(R),"]")
    print("Im(Omega_prime) within [",np.min(I),np.max(I),"]")
    Rabs = np.abs(R)
    Iabs = np.abs(I)
    print("|Re(Omega_prime)| within [",np.min(Rabs),np.max(Rabs),"]")
    print("|Im(Omega_prime)| within [",np.min(Iabs),np.max(Iabs),"]")
    print()
    Imax = np.max(I, axis=1)
    imax = np.argmax(Imax)
    kprime_max = kprime_grid[imax]
    print("kprime_max = ",kprime_max/(hbar*c),"cm^-1")
    print("|kprime_max| = ", np.linalg.norm(kprime_max)/(hbar*c),"cm^-1")
    print("1/lambda =", np.linalg.norm(kprime_max)/(hbar*c)/(2.*np.pi),"cm^-1")
    print()   
    
    #----------
    # set output filename
    #----------
    output_filename = "moment_eigenvalues.h5"
    print("Writing",output_filename)

    #----------
    # output data
    #----------
    fout = h5py.File(output_filename, "w")
    fout["Ve (erg)"] = Ve
    fout["phi0 (erg)"] = phi0
    fout["phi1 (erg)"] = phi1
    fout["N (1|ccm)"] = N
    fout["F (1|ccm)"] = F
    fout["P (1|ccm)"] = P
    fout["kprime_grid (erg)"] = kprime_grid
    fout["omegaprime (erg)"] = eigenvalues
    fout.close()

if __name__ == '__main__':
    compute_all_eigenvalues()
