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
dm2 = 2.4e-3 * eV**2 # erg
nthreads = 2


# Fiducial case
#numb_k = 1000
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
numb_k = 200
nphi_at_equator = 64
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

# number density [nu/nubar, flavor] (1/ccm)
N = np.array([[Nee   , Nxx   ],
              [Neebar, Nxxbar]])
print("\nN:")
print(N)
TrN = np.sum(N, axis=1)

# number flux density [nu/nubar, flavor, i] (1/ccm)
F = np.array([[Fee   , Fxx   ],
              [Feebar, Fxxbar]])
print("\nF:")
print(F)
TrF = np.sum(F, axis=1)

# closure
def get_ptilde(F_N):
    F2 = np.sum(F_N**2)
    
    # thick pressure tensor
    Pthick = np.identity(3) / 3.
        
    if F2==0:
        return Pthick
    else:
        fluxfac = np.sqrt(F2)
        chi = 1./3. + 2./15. * fluxfac**2 * (3. - fluxfac + 3.*fluxfac**2)
        
        # thin pressure tensor
        Pthin = np.array([[
            F_N[i]*F_N[j]/F2
            for j in range(3)] for i in range(3)])
        
        ptilde = (3.*chi-1.)/2.*Pthin + 3.*(1.-chi)/2.*Pthick
        
        return ptilde

# [nu/antinu]
ptilde = np.array([get_ptilde(TrF[0]/TrN[0]),
                   get_ptilde(TrF[1]/TrN[1])])
print("\nptilde:")
print(ptilde[0])
print("\nptildebar:")
print(ptilde[1])
    
# other inputs
rho = 0 # g/ccm
Ye = 0.5
average_energy = 50*MeV # erg

#----------
# get vmatter and phis (all in ergs)
#----------
mu_N = np.sqrt(2.) * GF * (N[:,0]     - N[:,1]    ) # [nu/nubar]
mu_F = np.sqrt(2.) * GF * (F[:,0,:]   - F[:,1,:]  ) # [nu/nubar, i]
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

#=============================#
# set up global shared memory #
#=============================#
# Stability matrix without k term
#S_nok_RA = mp.RawArray('d',8*8)
#S_nok = np.frombuffer(S_nok_RA).reshape((8,8))
S_nok = np.zeros((8,8))
# top left quadrant
S_nok[0  ,0  ] =   mu_N[1]
S_nok[0  ,1:4] = - mu_F[1]
S_nok[1:4,0  ] =   np.tensordot(mu_F[0], ptilde[0]+np.identity(3), axes=1) \
                 - np.tensordot(mu_F[1], ptilde[0]               , axes=1)
S_nok[1:4,1:4] = - mu_N[0] * (ptilde[0] + np.identity(3)) \
                 + mu_N[1] *              np.identity(3)
# bottom right quadrant
S_nok[4  ,4  ] = - mu_N[0]
S_nok[4  ,5:8] =   mu_F[0]
S_nok[5:8,4  ] = - np.tensordot(mu_F[1], ptilde[1]+np.identity(3), axes=1) \
                 + np.tensordot(mu_F[0], ptilde[1]               , axes=1)
S_nok[5:8,5:8] =   mu_N[1] * (ptilde[1] + np.identity(3)) \
                 - mu_N[0] *              np.identity(3)
# bottom left quadrant
S_nok[4  ,0  ] =  mu_N[1]
S_nok[4  ,1:4] = -mu_F[1]
S_nok[5:8,0  ] =  mu_F[1]
S_nok[5:8,1:4] = -mu_N[1] * ptilde[1]
# top right quadrant
S_nok[0  ,4  ] = -mu_N[0]
S_nok[0  ,5:8] =  mu_F[0]
S_nok[1:4,4  ] = -mu_F[0]
S_nok[1:4,5:8] =  mu_N[0] * ptilde[0]
print("\nS_nok:")
print(S_nok)

# build the k grid
#kmag_grid = phi1mag * np.geomspace(min_ktarget_multiplier,max_ktarget_multiplier,num=numb_k,endpoint=True)
dk = phi1mag * (max_ktarget_multiplier - min_ktarget_multiplier)/numb_k / (hbar*c)
kmag_grid = phi1mag * np.arange(min_ktarget_multiplier, max_ktarget_multiplier, dk)
print()
print("Using",len(kmag_grid),"kmag points between",kmag_grid[0]/(hbar*c),"and",kmag_grid[-1]/(hbar*c),"cm^-1")
print("dk =",dk,"cm^-1")

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

# construct full k grid
kgrid = [kmag*direction for direction in direction_grid for kmag in kmag_grid]
#kgrid = [[0,0,1e-16],] # should yield a growth rate of 6.931165501863837e10 for the Fiducial setup
kgrid.append([0,0,0])
kgrid = np.array(kgrid)

#=================================#
# do the real work for a single k #
#=================================#
# k:erg output:erg
def eigenvalues_single_k(k):
    # complete the stability matrix
    #S_nok = np.frombuffer(S_nok_RA).reshape((8,8))
    S = copy.deepcopy(S_nok)
    S[0  , 1:4] -= k
    S[4  , 5:8] -= k
    S[1:4, 0  ] -= np.tensordot(k, ptilde[0], axes=1)
    S[5:8, 4  ] -= np.tensordot(k, ptilde[1], axes=1)
    #print("\nS:")
    #print(S)

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
    for k in kgrid:
        evals_thisk = eigenvalues_single_k(k)
        eigenvalues.append(evals_thisk)
        #print(k, np.max(np.imag(evals_thisk))/hbar/1e10,"*1e10")
    #with mp.Pool(processes=nthreads) as pool:
    #    eigenvalues = pool.map(eigenvalues_single_k, kgrid)
    end_time = time.time()
    
    #----------
    # write info to screen
    #----------
    print()
    R = np.real(eigenvalues)/hbar/1e10
    I = np.imag(eigenvalues)/hbar/1e10
    print("time="+"{:.2e}".format(end_time-start_time)+"s.")
    print("R within [",np.min(R),np.max(R),"]e10 s^-1")
    print("I within [",np.min(I),np.max(I),"]e10 s^-1")
    Rabs = np.abs(R)
    Iabs = np.abs(I)
    print("|R| within [",np.min(Rabs),np.max(Rabs),"]e10 s^-1")
    print("|I| within [",np.min(Iabs),np.max(Iabs),"]e10 s^-1")
    print()
    Imax = np.max(I, axis=1)
    imax = np.argmax(Imax)
    kmax = kgrid[imax]
    print("kmax = ",kmax/(hbar*c),"cm^-1")
    print("|kmax| = ", np.linalg.norm(kmax)/(hbar*c),"cm^-1")
    print("1/lambda =", np.linalg.norm(kmax)/(hbar*c)/(2.*np.pi),"cm^-1")
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
    fout["ptilde"] = ptilde
    fout["kgrid (erg)"] = kgrid
    fout["eigenvalues (erg)"] = eigenvalues
    fout.close()

if __name__ == '__main__':
    compute_all_eigenvalues()
