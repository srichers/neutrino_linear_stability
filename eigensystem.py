# Copyright (C) 2021 Sherwood Richers & Sam Flynn
<<<<<<< HEAD
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

import time
import numpy as np
import h5py
import fileinput
import sys
import glob
import scipy.linalg
import copy
import multiprocessing as mp

h = 6.6260755e-27 # erg s
hbar = h/(2.*np.pi)
c = 2.99792458e10 # cm/s
MeV = 1.60218e-6 # erg
eV = MeV/1e6 # erg
GF_GeV2 = 1.1663787e-5 # GeV^-2
GF = GF_GeV2 / (1000*MeV)**2 * (hbar*c)**3
Mp = 1.6726219e-24

# inputs #
target_resolution = 20
input_filename = "lotsadata/original_data_DO/15Msun_400ms_DO.h5"
dm2 = 2.4e-3 * eV**2 # erg
k = 1.088e-18 # erg
ir_start = 254
ir_stop = 255
nthreads = 2

# set up global shared memory
S_nok_RA = mp.RawArray('d',(target_resolution*2)**2)
mu_tilde_RA= mp.RawArray('d', target_resolution*2)

def get_shared_numpy_arrays():
    # create numpy object from shared RawArray memory for easy reading
    S_nok = np.frombuffer(S_nok_RA).reshape((target_resolution*2,target_resolution*2))
    mu_tilde = np.frombuffer(mu_tilde_RA)
    return S_nok, mu_tilde

#===============#
# read the file #
#===============#
def read_data(filename):
    fin  = h5py.File(input_filename,"r")

    mugrid = np.array(fin["distribution_costheta_grid(lab)"])
    mumid = np.array(fin["distribution_costheta_mid(lab)"])
    dist = np.array(fin["distribution(erg|ccm,lab)"])
    rho = np.array(fin["rho(g|ccm,com)"])
    Ye = np.array(fin["Ye"])
    frequency = np.array(fin["distribution_frequency_mid(Hz,lab)"])
    
    fin.close()

    # sum over phi because we don't need it
    dist = np.sum(dist, axis=4)
    
    # heavy lepton distribution represents mu,tau,mubar,taubar
    dist[:,2,:,:] /= 4. 

    return mugrid, mumid, frequency, dist, rho, Ye
=======
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

import time
import numpy as np
import h5py
import fileinput
import sys
import glob
import scipy.linalg

h = 6.6260755e-27 # erg s
hbar = h/(2.*np.pi)
c = 2.99792458e10 # cm/s
MeV = 1.60218e-6 # erg
eV = MeV/1e6 # erg
GF_GeV2 = 1.1663787e-5 # GeV^-2
GF = GF_GeV2 / (1000*MeV)**2 * (hbar*c)**3
Mp = 1.6726219e-24

# inputs #
target_resolution = 20
input_filename = "lotsadata/original_data_DO/15Msun_400ms_DO.h5"
#SDF path "../FFO Data/FFO Run Me/15Msun_400ms_DO.h5"
dm2 = 2.4e-3 * eV**2 # erg
k = 1.088e-18 # erg
ir = 254

def stability_matrix_nok(mu_tilde,n_tilde,omega_tilde, Ve, phi0, phi1):
    matrix_size = len(mu_tilde)

    # self-interaction part
    S_nok = -np.sqrt(2) * GF * np.array([[
        (1.0 - mu_tilde[i]*mu_tilde[j])*n_tilde[j]
        for j in range(matrix_size)] for i in range(matrix_size)])
    print("after SI:",np.min(S_nok),np.max(S_nok))
    
    # vacuum and phis
    for i in range(matrix_size):
        S_nok[i,i] += omega_tilde[i] \
            + Ve \
            + phi0 \
            - mu_tilde[i]*phi1
    print("after vac/matter:",np.min(S_nok),np.max(S_nok))

    return S_nok

def stability_matrix(S_nok, mu_tilde, k):
    matrix_size = len(mu_tilde)
            
    # k term
    S = S_nok
    for i in range(matrix_size):
        S[i,i] += mu_tilde[i]*k
    print("after k:",np.min(S),np.max(S))

    return S
>>>>>>> changed the logspace to geomspace in build_k_grid. Grids now match perfectly without extra steps.

def build_k_grid(n_nu,n_nubar,numb_k,min_multiplier,max_multiplier):
    #multipliers set the range away from k_target that the grid should begin/end.
    #Target k is based on the 'total' self-interaction at this radius
    k_target = np.sqrt(2.) * GF * (n_nu-n_nubar)
    #make log spaced array for positive values of k_target
    k_grid_pos=np.geomspace(min_multiplier*k_target,max_multiplier*k_target,num=numb_k,endpoint=True)
    #repeat for negative values of k_target
    k_grid_neg=np.geomspace(-min_multiplier*k_target,-max_multiplier*k_target,num=numb_k,endpoint=True)
    #join into single array
    k_grid=np.concatenate((k_grid_pos,k_grid_neg),axis=0)
    return k_grid
   #SDF: Later, I would like to make this grid truly log spaces about the target, 
   # i.e. log spaced out in both directions away from k_target, as below.
   #k_upper=np.logspace(ktarget,ktarget*max_multiplier,num=numb_k,endpoint=True,base=np.exp(1))
   # k_lower=np.logspace(ktarget,ktarget*min_multiplier,num=numb_k,endpoint=True,base=np.exp(1))
   #k_grid=np.concatenate[(k_grid_lower,k_grid_upper),axis=0]

<<<<<<< HEAD

#=========================#
# refine the distribution #
#=========================#
def refine_distribution(old_mugrid, old_mumid, old_dist, target_resolution):
    # store moments for comparison
    old_nmu = len(old_mumid)
    old_edens = np.sum(old_dist               , axis=(1,2,3))
    old_flux  = np.sum(old_dist * old_mumid   , axis=(1,2,3))
    old_press = np.sum(old_dist * old_mumid**2, axis=(1,2,3))

    # initially set to old values in case not refining
    new_mugrid = old_mugrid
    new_mumid = old_mumid
    new_dist = old_dist
    new_nmu = old_nmu
    
    nref =  int(np.log2(target_resolution / old_nmu))
    for iref in range(nref):
        if(iref>0):
            old_dist = new_dist
            old_mugrid = new_mugrid
    
        # get the new mugrid
        old_nmu = len(old_mugrid)-1
        old_dmu = np.array([old_mugrid[i+1]-old_mugrid[i] for i in range(old_nmu)])
        new_mugrid = []
        for i in range(len(old_mugrid)-1):
            new_mugrid.append(old_mugrid[i])
            new_mugrid.append(0.5*(old_mugrid[i]+old_mugrid[i+1]))
        new_mugrid.append(old_mugrid[-1])
        new_nmu = len(new_mugrid)-1
        new_dmu = np.array([new_mugrid[i+1]-new_mugrid[i] for i in range(new_nmu)]) # dist between cell faces
        print("\tIncreasing from nmu=",old_nmu,"to nmu=",new_nmu)

        # get properties & derivatives of the old distribution function
        nr = old_dist.shape[0]
        ns = old_dist.shape[1]
        ne = old_dist.shape[2]
        
        new_dist = np.zeros((nr,ns,ne,new_nmu))
        for i in range(old_nmu):
            new_dist[:,:,:,2*i+0] = old_dist[:,:,:,i]/2.
            new_dist[:,:,:,2*i+1] = old_dist[:,:,:,i]/2.

    # new grid structure #
    new_mumid = np.array([0.5*(new_mugrid[i]+new_mugrid[i+1]) for i in range(new_nmu)])

    # calculate error in moments #
    new_edens = np.sum(new_dist               , axis=(1,2,3))
    new_flux  = np.sum(new_dist * new_mumid   , axis=(1,2,3))
    new_press = np.sum(new_dist * new_mumid**2, axis=(1,2,3))
    print("Max relative error in net energy density:", np.max(np.abs((new_edens-old_edens)/old_edens)))
    print("Max relative error in net flux:", np.max(np.abs((new_flux-old_flux)/old_edens)))
    print("Max relative error in net pressure:", np.max(np.abs((new_press-old_press)/old_edens)))

    return new_mugrid, new_mumid, new_dist


#=========================#
# construct tilde vectors #
#=========================#
def construct_tilde_vectors(mumid, dm2, average_energy, number_dist):
    S_nok, mu_tilde = get_shared_numpy_arrays()

    omega = np.ones(len(mumid))[np.newaxis,:] * (dm2 / (2.*average_energy))[:,np.newaxis]
    omega_tilde = np.concatenate((-omega, omega), axis=1)
    print("omega_tilde:",np.shape(omega_tilde))

    # set global mu_tilde
    np.copyto(mu_tilde, np.concatenate((mumid, mumid)))
    print("mu_tilde:",np.shape(mu_tilde))
    
    n_nu    = number_dist[:,0,:] - number_dist[:,2,:]
    n_nubar = number_dist[:,1,:] - number_dist[:,2,:]
    n_tilde = np.concatenate((n_nu, -n_nubar), axis=1)
    print("n_tilde:",np.shape(n_tilde))

    return omega_tilde, n_tilde

#===========================================#
# construct stability matrix without k term #
#===========================================#
def set_stability_matrix_nok(n_tilde,omega_tilde, Ve, phi0, phi1):
    S_nok, mu_tilde = get_shared_numpy_arrays()
    matrix_size = len(mu_tilde)

    for i in range(matrix_size):
        # self-interaction part
        S_nok[i,:] = -np.sqrt(2) * GF * np.array([
            (1.0 - mu_tilde[i]*mu_tilde[j])*n_tilde[j]
            for j in range(matrix_size)])
        
        # vacuum and phis
        S_nok[i,i] += omega_tilde[i] \
            + Ve \
            + phi0 \
            - mu_tilde[i]*phi1

    print("after vac/matter:",np.min(S_nok),np.max(S_nok))

#=================================#
# do the real work for a single k #
#=================================#
def eigenvalues_single_k(k):
    # complete the stability matrix
    S = stability_matrix(k)

    # find the eigenvalues
    return scipy.linalg.eigvals(S)

#=================================#
# construct full stability matrix #
#=================================#
def stability_matrix(k):
    S_nok, mu_tilde = get_shared_numpy_arrays()
    matrix_size = len(mu_tilde)
            
    # k term
    S = copy.deepcopy(S_nok)
    for i in range(matrix_size):
        S[i,i] += mu_tilde[i]*k
    print("after k:",np.min(S),np.max(S))

    return S

def single_file(input_filename):
    # read in data and calculate moments
    mugrid, mumid, frequency, dist, rho, Ye = read_data(input_filename)

    # refine distribution
    mugrid, mumid, dist = refine_distribution(mugrid, mumid, dist, target_resolution)
    
    #=======================#
    # integrate over energy #
    #=======================#
    neutrino_energy_mid = h * frequency
    number_dist = np.sum(dist / neutrino_energy_mid[np.newaxis,np.newaxis,:,np.newaxis],\
                         axis=(2))
    print("number_dist:",np.shape(number_dist))
    
    #==========================#
    # calculate average energy #
    #==========================#
    edens = np.sum(dist,axis=(1,2,3))
    ndens = np.sum(number_dist,axis=(1,2))
    average_energy = edens / ndens
    print("average_energy:",np.min(average_energy)/MeV, np.max(average_energy)/MeV)

    #======================#
    # get vmatter and phis #
    #======================#
    Ve = np.sqrt(2.) * GF * rho * Ye / Mp
    M0 = np.sqrt(2.) * GF * np.sum(number_dist, axis=2)
    M1 = np.sqrt(2.) * GF * np.sum(number_dist * mumid, axis=2)
    phi0 = (M0[:,0] - M0[:,2]) - (M0[:,1] - M0[:,2])
    phi1 = (M1[:,0] - M1[:,2]) - (M1[:,1] - M1[:,2])
    print("phi0:",np.shape(phi0))

    # get processors ready
    pool = mp.Pool(nthreads)

    # construct tilde vectors
    # use copyto to keep mu_tilde pointing at the shared buffer
    omega_tilde, n_tilde = construct_tilde_vectors(mumid, dm2, average_energy, number_dist)

    # loop over radii
    ir_list = range(ir_start, ir_stop+1)
    eigenvalues = []
    for ir in ir_list:
        start = time.time()

        # construct stability matrix
        set_stability_matrix_nok(n_tilde[ir], omega_tilde[ir], Ve[ir], phi0[ir], phi1[ir])

        # loop over k points. Temporary stupid k list.
        klist = np.arange(3)*k
        eigenvalues_thisr = pool.map(eigenvalues_single_k, klist)
        eigenvalues.append(eigenvalues_thisr)
        
        end = time.time()
        print("Time elapsed for ir =",ir,":",end-start, "sec.")
        print(eigenvalues[-1])

    # output data
    S_nok, mu_tilde = get_shared_numpy_arrays()
    output_filename = input_filename[:-3]+"_eigenvalues.h5"
    print("Writing",output_filename)
    fout = h5py.File(output_filename, "w")
    fout["Ve (erg)"] = Ve
    fout["phi0 (erg)"] = phi0
    fout["phi1 (erg)"] = phi1
    fout["omega_tilde (erg)"] = omega_tilde
    fout["mu_tilde"] = mu_tilde
    fout["n_tilde"] = n_tilde
    fout["ir (base 0)"] = ir_list
    fout["ir (base 1)"] = [ir+1 for ir in ir_list]
    fout["number_dist (1|ccm)"] = number_dist
    fout["eigenvalues (erg)"] = eigenvalues
    fout.close()

single_file(input_filename)
=======
def single_file(input_filename):
    # get grid data from old file
    output_filename = input_filename.strip(".h5") + "_"+str(target_resolution)+".h5"
    print("Creating", output_filename)

    fin  = h5py.File(input_filename,"r")

    #==================#
    # get the old data #
    #==================#
    old_mugrid = np.array(fin["distribution_costheta_grid(lab)"])
    old_mumid = np.array(fin["distribution_costheta_mid(lab)"])
    old_nmu = len(old_mugrid)-1
    old_dist = np.array(fin["distribution(erg|ccm,lab)"])
    old_edens = np.sum(old_dist,\
                       axis=(1,2,3,4))
    old_flux  = np.sum(old_dist * old_mumid[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]   ,\
                       axis=(1,2,3,4))
    old_press = np.sum(old_dist * old_mumid[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]**2,\
                       axis=(1,2,3,4))

    # initially set to old values in case not refining
    new_mugrid = old_mugrid
    new_mumid = old_mumid
    new_dist = old_dist
    new_nmu = old_nmu
    
    #=========================#
    # refine the distribution #
    #=========================#
    nref =  int(np.log2(target_resolution / old_nmu))
    for iref in range(nref):
        if(iref>0):
            old_dist = new_dist
            old_mugrid = new_mugrid
    
        # get the new mugrid
        old_nmu = len(old_mugrid)-1
        old_dmu = np.array([old_mugrid[i+1]-old_mugrid[i] for i in range(old_nmu)])
        new_mugrid = []
        for i in range(len(old_mugrid)-1):
            new_mugrid.append(old_mugrid[i])
            new_mugrid.append(0.5*(old_mugrid[i]+old_mugrid[i+1]))
        new_mugrid.append(old_mugrid[-1])
        new_nmu = len(new_mugrid)-1
        new_dmu = np.array([new_mugrid[i+1]-new_mugrid[i] for i in range(new_nmu)]) # dist between cell faces
        print("\tIncreasing from nmu=",old_nmu,"to nmu=",new_nmu)

        # get properties & derivatives of the old distribution function
        nr = old_dist.shape[0]
        ns = old_dist.shape[1]
        ne = old_dist.shape[2]
        nphi = old_dist.shape[4]
        
        new_dist = np.zeros((nr,ns,ne,new_nmu,nphi))
        for i in range(old_nmu):
            new_dist[:,:,:,2*i+0,:] = old_dist[:,:,:,i,:]/2.
            new_dist[:,:,:,2*i+1,:] = old_dist[:,:,:,i,:]/2.


    #====================#
    # new grid structure #
    #====================#
    new_mumid = np.array([0.5*(new_mugrid[i]+new_mugrid[i+1]) for i in range(new_nmu)])

    #===============================#
    # integrate over energy and phi #
    #===============================#
    neutrino_energy_mid = h * np.array(fin["distribution_frequency_mid(Hz,lab)"])
    number_dist = np.sum(new_dist / neutrino_energy_mid[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis],\
                         axis=(2,4))
    number_dist[:,2,:] /= 4. # heavy lepton distribution represents mu,tau,mubar,taubar
    print("number_dist:",np.shape(number_dist))
    
    #============================#
    # calculate error in moments #
    #============================#
    new_edens = np.sum(new_dist,axis=(1,2,3,4))
    new_flux  = np.sum(new_dist * new_mumid[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis] ,\
                       axis=(1,2,3,4))
    new_press = np.sum(new_dist * new_mumid[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]**2,\
                       axis=(1,2,3,4))
    print("Max relative error in net energy density:", np.max(np.abs((new_edens-old_edens)/old_edens)))
    print("Max relative error in net flux:", np.max(np.abs((new_flux-old_flux)/old_edens)))
    print("Max relative error in net pressure:", np.max(np.abs((new_press-old_press)/old_edens)))

    #==========================#
    # calculate average energy #
    #==========================#
    new_ndens = np.sum(number_dist,axis=(1,2))
    average_energy = new_edens / new_ndens
    print("average_energy:",np.min(average_energy)/MeV, np.max(average_energy)/MeV)

    #=========================#
    # construct tilde vectors #
    #=========================#
    omega = np.ones(new_nmu)[np.newaxis,:] * (dm2 / (2.*average_energy))[:,np.newaxis]
    omega_tilde = np.concatenate((-omega, omega), axis=1)
    print("omega_tilde:",np.shape(omega_tilde))

    mu_tilde = np.concatenate((new_mumid, new_mumid))
    print("mu_tilde:",np.shape(mu_tilde))
    
    n_nu    = number_dist[:,0,:] - number_dist[:,2,:]
    n_nubar = number_dist[:,1,:] - number_dist[:,2,:]
    n_tilde = np.concatenate((n_nu, -n_nubar), axis=1)
    print("n_tilde:",np.shape(n_tilde))
    
    
    #======================#
    # get vmatter and phis #
    #======================#
    Ve = np.sqrt(2.) * GF * np.array(fin["rho(g|ccm,com)"]) * np.array(fin["Ye"]) / Mp
    M0 = np.sqrt(2.) * GF * np.sum(number_dist, axis=2)
    M1 = np.sqrt(2.) * GF * np.sum(number_dist * new_mumid, axis=2)
    phi0 = (M0[:,0] - M0[:,2]) - (M0[:,1] - M0[:,2])
    phi1 = (M1[:,0] - M1[:,2]) - (M1[:,1] - M1[:,2])
    print("phi0:",np.shape(phi0))
    
    
    #============================#
    # construct stability matrix #
    #============================#
    S_nok = stability_matrix_nok(mu_tilde, n_tilde[ir], omega_tilde[ir], Ve[ir], phi0[ir], phi1[ir])
    S = stability_matrix(S_nok, mu_tilde, k)
    print("S:",np.shape(S))

    #=================#
    # get eigenvalues #
    #=================#
    start = time.time()
    evals = scipy.linalg.eigvals(S)
    end = time.time()
    print("Time elapsed:",end-start)
    print(evals)

single_file(input_filename)
>>>>>>> changed the logspace to geomspace in build_k_grid. Grids now match perfectly without extra steps.
