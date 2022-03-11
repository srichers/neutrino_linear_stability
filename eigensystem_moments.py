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
import discrete_ordinates
import minerbo

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
numb_k = 100
nphi_at_equator = 16
min_ktarget_multiplier = 1e-3
max_ktarget_multiplier = 1e1

# number density [nu/nubar, flavor] (1/ccm)
Nee = 1.421954234999705e+33
Neebar = 1.9146237131657563e+33
Nxx = 1.9645407875568215e+33
Nxxbar = Nxx
N = np.array([[Nee   , Nxx   ],
              [Neebar, Nxxbar]])

# number flux density [nu/nubar, flavor, i] (1/ccm)
Fee    = np.array([0.0974572,    0.04217632, -0.13433261]) * Nee
Feebar = np.array([ 0.07237959,  0.03132354, -0.3446878 ]) * Neebar
Fxx    = np.array([-0.02165833,  0.07431613, -0.53545951]) * Nxx
Fxxbar = Fxx
F = np.array([[Fee   , Fxx   ],
              [Feebar, Fxxbar]])
# closure
def get_Pij(N, F):
    F2 = np.sum(F**2)
    fluxfac = np.sqrt(F2)/N
    chi = 1./3. + 2./15. * fluxfac**2 * (3. - fluxfac + 3.*fluxfac**2)
    
    # thick pressure tensor
    Pthick = np.identity(3) * N/3.

    # thin pressure tensor
    Pthin = np.array([[
        F[i]*F[j]/F2 * N 
        for j in range(3)] for i in range(3)])

    return (3.*chi-1.)/2. * Pthin + 3.*(1.-chi)/2. * Pthick

# number pressure [nu/nubar, flavor, i, j]
P = np.array([[ get_Pij(Nee,   Fee   ), get_Pij(Nxx   , Fxx   )],
              [ get_Pij(Neebar,Feebar), get_Pij(Nxxbar, Fxxbar)]])

# other inputs
rho = 0 # g/ccm
Ye = 0.5
average_energy = 50*MeV # erg

#----------
# get vmatter and phis (all in ergs)
#----------
mu_N = np.sqrt(2.) * (N[:,0]     - N[:,1]    ) # [nu/nubar]
mu_F = np.sqrt(2.) * (F[:,0,:]   - F[:,1,:]  ) # [nu/nubar, i]
mu_P = np.sqrt(2.) * (P[:,0,:,:] - P[:,0,:,:]) # [nu/nubar, i, j]
Ve   = np.sqrt(2.) * GF * (rho * Ye / Mp)
phi0 = mu_N[0] - mu_N[1]
phi1 = mu_F[0] - mu_F[1] # [i]

# mu parts of the stability matrix [nu/nubar, i, j] where 0 is N and 1-3 are F
mu = np.zeros((2,4,4))
mu[:, 0  , 0  ] =  mu_N
mu[:, 0  , 1:4] =  mu_F
mu[:, 1:4, 0  ] = -mu_F
mu[:, 1:4, 1:4] = -mu_P

# Stability matrix without k term
S_nok = np.zeros((8,8))
S_nok[0:4, 0:4] =  mu[0]
S_nok[0:4, 4:8] = -mu[0]
S_nok[4:8, 0:4] =  mu[1]
S_nok[4:8, 4:8] = -mu[1]

# build the k grid
kmag_grid = phi0 * np.geomspace(min_ktarget_multiplier,max_ktarget_multiplier,num=numb_k,endpoint=True)
print("Using",len(kmag_grid),"kmag points")

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
kgrid = np.array([kmag*direction for direction in direction_grid for kmag in kmag_grid])

#=============================#
# set up global shared memory #
#=============================#
S_nok_RA = mp.RawArray('d',8*8)
S_nok = np.frombuffer(S_nok_RA).reshape((8,8))

#=================================#
# do the real work for a single k #
#=================================#
# k:erg output:erg
def eigenvalues_single_k(k):
    # complete the stability matrix
    S_nok = np.frombuffer(S_nok_RA).reshape((8,8))
    S = copy.deepcopy(S_nok)
    S[0  , 1:4] -= k
    S[1:4, 0  ] -= k # should contain C
    S[4  , 5:8] -= k
    S[5:8, 4  ] -= k # should contain C

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
    #for k in kgrid:
    #    eigenvalues.append(eigenvalues_single_k(k))
    with mp.Pool(processes=nthreads) as pool:
        eigenvalues = pool.map(eigenvalues_single_k, kgrid)
    end_time = time.time()
    
    #----------
    # write info to screen
    #----------
    print()
    R = np.real(eigenvalues)
    I = np.imag(eigenvalues)
    print("time="+"{:.2e}".format(end_time-start_time)+"s.")
    print("R within [",np.min(R),np.max(R),"]")
    print("I within [",np.min(I),np.max(I),"]")
    Rabs = np.abs(R)
    Iabs = np.abs(I)
    print("|R| within [",np.min(Rabs),np.max(Rabs),"]")
    print("|I| within [",np.min(Iabs),np.max(Iabs),"]")
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
    fout["kgrid (erg)"] = kgrid
    fout["eigenvalues (erg)"] = eigenvalues
    fout.close()

if __name__ == '__main__':
    compute_all_eigenvalues()
