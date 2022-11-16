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
import numpy as np
import h5py
import fileinput
import sys
import glob
import scipy.linalg
import copy
import multiprocessing as mp
import scipy.optimize
from scipy.optimize import minimize, Bounds, basinhopping
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

# Define shared memory block outside of functions
S_nok_RA = mp.RawArray('d',8*8)
C_RA = mp.RawArray('d',2*3*3)

# fill in the shared arrays S_nok and C
# return Ntot (cm^-3) and phi1mag (unitless - already normalized by Ntot)
def construct_S_nok(N, F, flavor_trace_chi, flavor_trace_fhat, verbose=False):
    N = copy.deepcopy(N)
    F = copy.deepcopy(F)
    
    # normalize by the total number of neutrinos
    Ntot = np.sum(N)
    N /= Ntot
    F /= Ntot
    
    N_FT = np.sum(N, axis=1)         # [nu/nubar]
    F_FT = np.sum(F, axis=1)         # [nu/nubar, i]

    if(verbose):
        print("\nN:")
        print(N)
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
    if(verbose):
        print("\nP:")
        print(P)
        print("P shape:",np.shape(P))
        print("P_FT shape:",np.shape(P_FT))
    
    
    #=====================================#
    # Set Cij to the flavor-traced Ptilde #
    #=====================================#
    C = np.frombuffer(C_RA).reshape((2,3,3))
    C[:,:,:] = Ptilde_FT # [nu/nubar, i, j]
    if(verbose):
        print("C shape:", np.shape(C))
        print("C=")
        print(C)
    
    #----------
    # get vmatter and phis (all in ergs)
    #----------
    mu_N = np.sqrt(2.) * (N[:,0]     - N[:,1]    ) # [nu/nubar]
    mu_F = np.sqrt(2.) * (F[:,0,:]   - F[:,1,:]  ) # [nu/nubar, i]
    mu_P = np.sqrt(2.) * (P[:,0,:,:] - P[:,1,:,:]) # [nu/nubar, i, j]
    phi0 = mu_N[0] - mu_N[1]
    phi1 = mu_F[0] - mu_F[1] # [i]
    phi1mag_unitless = np.sqrt(np.sum(phi1**2))

    if(verbose):
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
    if(verbose):
        print("\nmu:")
        print(mu)
    
    # Stability matrix without k term
    S_nok = np.frombuffer(S_nok_RA).reshape((8,8))
    S_nok[0:4, 0:4] =  mu[0]
    S_nok[0:4, 4:8] = -mu[0]
    S_nok[4:8, 0:4] =  mu[1]
    S_nok[4:8, 4:8] = -mu[1]
    if(verbose):
        print("\nS_nok:")
        print(S_nok)

    return phi1mag_unitless

def construct_kprime_grid(phi1mag_unitless, max_ktarget_multiplier, numb_k, nphi_at_equator):
    # build the kprime grid
    dkprime = max_ktarget_multiplier/numb_k
    kprime_mag_grid = phi1mag_unitless * np.arange(dkprime, max_ktarget_multiplier, dkprime)
    print("Using",len(kprime_mag_grid),"kprime_mag points")
    
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

    return kprime_grid

#=================================#
# do the real work for a single k #
#=================================#
# k:erg output:erg
def eigenvalues_single_k(kprime):
    # complete the stability matrix
    C = np.frombuffer(C_RA).reshape((2,3,3))
    S_nok = np.frombuffer(S_nok_RA).reshape((8,8))
    S = copy.deepcopy(S_nok)
    S[0  , 1:4] -= kprime
    S[4  , 5:8] -= kprime
    S[1:4, 0  ] -= np.tensordot(kprime, C[0], axes=1)
    S[5:8, 4  ] -= np.tensordot(kprime, C[1], axes=1)

    # find the eigenvalues
    return np.linalg.eigvals(S)

# kprime,eigenvaluesprime dimensionless, all others are standard CGS units
# k:erg eigenvalues:erg
# assumes calculation is done in the fluid rest frame
def restore_units(ne, N, F, kprime, eigenvaluesprime):
    Ntot = np.sum(N)
    units = GF*Ntot # erg
    eigenvalues = eigenvaluesprime*GF*Ntot + np.sqrt(2)*GF * ( (N[0,0]-N[0,1]) - (N[1,0]-N[1,1]) + ne)
    k           = kprime          *GF*Ntot + np.sqrt(2)*GF * ( (F[0,0]-F[0,1]) - (F[1,0]-F[1,1]) )[np.newaxis,:]
    return k, eigenvalues

#===========================================#
# do all eigenvalue calculations for a file #
#===========================================#
def compute_all_eigenvalues(ne, N, F, flavor_trace_chi, flavor_trace_fhat, numb_k, nphi_at_equator, max_ktarget_multiplier, nthreads):
    # set up the shared data
    phi1mag_unitless = construct_S_nok(N, F, flavor_trace_chi, flavor_trace_fhat)
    kprime_grid = construct_kprime_grid(phi1mag_unitless, max_ktarget_multiplier, numb_k, nphi_at_equator) # dimensionless

    #----------
    # loop over k points
    #----------
    with mp.Pool(processes=nthreads) as pool:
        eigenvaluesprime = pool.map(eigenvalues_single_k, kprime_grid)
    eigenvaluesprime = np.array(eigenvaluesprime)

    # restore units
    k_grid, eigenvalues = restore_units(ne, N, F, kprime_grid, eigenvaluesprime)

    return k_grid, eigenvalues


#=====================================#
# optimize to find the max eigenvalue #
#=====================================#
def compute_max_eigenvalue(ne, N, F, flavor_trace_chi, flavor_trace_fhat, max_ktarget_multiplier):
    # set up the shared data
    phi1mag_unitless = construct_S_nok(N, F, flavor_trace_chi, flavor_trace_fhat)

    # function that optimizer will use
    def max_eigenval_single_k(kprime):
        evals = eigenvalues_single_k(kprime)
        result = -np.max(np.imag(evals))
        return result

    # find k that maximizes eigenvalues. All dimensionless
    kprime_bound = phi1mag_unitless * max_ktarget_multiplier
    kprime_bounds = (-kprime_bound, kprime_bound)
    bounds = (kprime_bounds, kprime_bounds, kprime_bounds)
    result = scipy.optimize.shgo(max_eigenval_single_k, bounds)
    kprime_max = np.array([result.x])
    eigenvaluesprime = np.array([eigenvalues_single_k(kprime_max[0])])

    # restore units
    kmax, eigenvalues = restore_units(ne, N, F, kprime_max, eigenvaluesprime)

    print(result.message)
    return kmax, eigenvalues

def print_info(kprime_grid, eigenvalues):
    k_mag_max = np.max(np.sqrt(np.sum(kprime_grid**2,axis=1)))
    print("Using",len(kprime_grid),"kprime points up to","{:e}".format(np.max(kprime_grid)/(hbar*c)),"cm^-1")
    R = np.real(eigenvalues)
    I = np.imag(eigenvalues)
    #print("Re(Omega_prime) within [","{:e}".format(np.min(R)/hbar),"{:e}".format(np.max(R)/hbar),"] s^-1")
    #print("Im(Omega_prime) within [","{:e}".format(np.min(I)/hbar),"{:e}".format(np.max(I)/hbar),"] s^-1")
    Rabs = np.abs(R)
    Iabs = np.abs(I)
    #print("|Re(Omega_prime)| within [","{:e}".format(np.min(Rabs)/hbar),"{:e}".format(np.max(Rabs)/hbar),"] s^-1")
    #print("|Im(Omega_prime)| within [","{:e}".format(np.min(Iabs)/hbar),"{:e}".format(np.max(Iabs)/hbar),"] s^-1")
    #print()
    Imax = np.max(I, axis=1)
    imax = np.argmax(Imax)
    kprime_max = kprime_grid[imax]
    print("kprime_max = ",kprime_max/(hbar*c),"cm^-1")
    print("|kprime_max| = ", "{:e}".format(np.linalg.norm(kprime_max)/(hbar*c)),"cm^-1")
    print("lambda =", "{:e}".format((2.*np.pi*hbar*c)/np.linalg.norm(kprime_max)),"cm")
    print("max_growthrate =","{:e}".format(np.max(I)/hbar),"s^-1")
    print()

    
if __name__ == '__main__':
    
    #--------
    # initial conditions
    #--------
    #Nee    = 4.89e32
    #Neebar = 4.89e32
    #Nxx    = 0
    #Nxxbar = 0
    #Fee    = np.array([0,0,  1./3]) * Nee
    #Feebar = np.array([0,0, -1./3]) * Neebar
    #Fxx    = np.array([0,0,  0   ])
    #Fxxbar = np.array([0,0,  0   ])
    
    # 90Degree case
    #Nee    = 4.89e32
    #Neebar = 4.89e32
    #Nxx    = 0
    #Nxxbar = 0
    #Fee    = np.array([0, 0   , 1./3]) * Nee
    #Feebar = np.array([0, 1/3., 0   ]) * Neebar
    #Fxx    = np.array([0, 0   , 0   ])
    #Fxxbar = np.array([0, 0   , 0   ])
    
    # TwoThirds case
    #Nee    = 4.89e32
    #Neebar = 4.89e32 * 2./3.
    #Nxx    = 0
    #Nxxbar = 0
    #Fee    = np.array([0,0,  0   ]) * Nee
    #Feebar = np.array([0,0, -1./3]) * Neebar
    #Fxx    = np.array([0,0,  0   ])
    #Fxxbar = np.array([0,0,  0   ])
    
    # NSM case
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
    flavor_trace_chi = True  # in calculation of P. Always true in Cij
    flavor_trace_fhat = True # in calculation of P. Always true in Cij
    max_ktarget_multiplier = 10
    rho = 0 # g/ccm
    Ye = 0.5
    nthreads = 8

    ne = rho*Ye/Mp
    N = np.array([[Nee   , Nxx   ],
                  [Neebar, Nxxbar]]) # [nu/nubar, flavor]
    F = np.array([[Fee   , Fxx   ],
                  [Feebar, Fxxbar]]) # [nu/nubar, flavor, i]


    # global optimizer
    print("### SHGO OPTIMIZER ###")
    kmax_opt, eigenvalues_opt = compute_max_eigenvalue(ne, N, F, flavor_trace_chi, flavor_trace_fhat, max_ktarget_multiplier)
    print_info(kmax_opt, eigenvalues_opt)

    # compute full grid
    print("### GRID SEARCH ###")
    numb_k = 200
    nphi_at_equator = 32
    k_grid, eigenvalues = compute_all_eigenvalues(ne, N, F, flavor_trace_chi, flavor_trace_fhat, numb_k, nphi_at_equator, max_ktarget_multiplier, nthreads)
    print_info(k_grid, eigenvalues)

    # output data from grid search
    output_filename = "moment_eigenvalues.h5"
    print("Writing",output_filename)
    fout = h5py.File(output_filename, "w")
    fout["k_grid (erg)"] = k_grid
    fout["omega (erg)"] = eigenvalues
    fout.close()

