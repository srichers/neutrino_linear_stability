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
target_resolution = 20
input_filename = "lotsadata/original_data_DO/15Msun_400ms_DO.h5"
dm2 = 2.4e-3 * eV**2 # erg
ir_start = 254
ir_stop = 255
nthreads = 2
numb_k = 10
min_ktarget_multiplier = 1e-3
max_ktarget_multiplier = 1e1
distribution_interpolator = "Minerbo" # "DO" or "Minerbo"

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

#===========================================#
# build array of k values to be looped over #
#===========================================#
# input/output units of ergs
def build_k_grid(k_target,numb_k,min_ktarget_multiplier,max_ktarget_multiplier):
    #make log spaced array for positive values of k_target
    k_grid_pos = k_target * np.geomspace(min_ktarget_multiplier,max_ktarget_multiplier,num=numb_k,endpoint=True)

    #join into single array
    k_grid=np.concatenate((-np.flip(k_grid_pos),[0],k_grid_pos),axis=0)
    return k_grid


#=========================#
# construct tilde vectors #
#=========================#
def construct_tilde_vectors(mumid, dm2, average_energy, number_dist):
    S_nok, mu_tilde = get_shared_numpy_arrays()

    # omega_tilde (erg)
    omega = np.ones(len(mumid))[np.newaxis,:] * (dm2 / (2.*average_energy))[:,np.newaxis]
    omega_tilde = np.concatenate((-omega, omega), axis=1)

    # set global mu_tilde (dimensionless)
    np.copyto(mu_tilde, np.concatenate((mumid, mumid)))

    # n_tilde (cm^-3)
    n_nu    = number_dist[:,0,:] - number_dist[:,2,:]
    n_nubar = number_dist[:,1,:] - number_dist[:,2,:]
    n_tilde = np.concatenate((n_nu, -n_nubar), axis=1)

    return omega_tilde, n_tilde

#===========================================#
# construct stability matrix without k term #
#===========================================#
# n_tilde:cm^-3 omega_tilde:erg Ve:erg phi:erg output:erg
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

#=================================#
# do the real work for a single k #
#=================================#
# k:erg output:erg
def eigenvalues_single_k(k):
    # complete the stability matrix
    S = stability_matrix(k)

    # find the eigenvalues
    return np.linalg.eigvals(S)

#=================================#
# construct full stability matrix #
#=================================#
# k:erg output:erg
def stability_matrix(k):
    S_nok, mu_tilde = get_shared_numpy_arrays()
    matrix_size = len(mu_tilde)
            
    # k term
    S = copy.deepcopy(S_nok)
    for i in range(matrix_size):
        S[i,i] += mu_tilde[i]*k

    return S

#=============================#
# set up global shared memory #
#=============================#
S_nok_RA = mp.RawArray('d',(target_resolution*2)**2)
mu_tilde_RA= mp.RawArray('d', target_resolution*2)
def get_shared_numpy_arrays():
    # create numpy object from shared RawArray memory for easy reading
    S_nok = np.frombuffer(S_nok_RA).reshape((target_resolution*2,target_resolution*2))
    mu_tilde = np.frombuffer(mu_tilde_RA)
    return S_nok, mu_tilde

#===========================================#
# do all eigenvalue calculations for a file #
#===========================================#
def single_file(input_filename):
    #----------
    # read in data
    #----------
    mugrid, mumid, frequency, dist, rho, Ye = read_data(input_filename)

    #----------
    # refine distribution
    #----------
    mugrid, mumid, dist = discrete_ordinates.refine_distribution(mugrid, mumid, dist, target_resolution)
    if distribution_interpolator == "Minerbo":
        dist = minerbo.redistribute_distribution(mugrid, mumid, dist, target_resolution)
        
    #----------
    # integrate over energy
    #----------
    neutrino_energy_mid = h * frequency
    number_dist = np.sum(dist / neutrino_energy_mid[np.newaxis,np.newaxis,:,np.newaxis],\
                         axis=(2))
    print("number_dist shape:",np.shape(number_dist))
    
    #----------
    # calculate average energy
    #----------
    edens = np.sum(dist,axis=(1,2,3))
    ndens = np.sum(number_dist,axis=(1,2))
    average_energy = edens / ndens

    #----------
    # get vmatter and phis
    #----------
    Ve = np.sqrt(2.) * GF * rho * Ye / Mp
    M0 = np.sqrt(2.) * GF * np.sum(number_dist, axis=2)
    M1 = np.sqrt(2.) * GF * np.sum(number_dist * mumid, axis=2)
    phi0 = (M0[:,0] - M0[:,2]) - (M0[:,1] - M0[:,2])
    phi1 = (M1[:,0] - M1[:,2]) - (M1[:,1] - M1[:,2])

    #----------
    # construct tilde vectors
    #----------
    # use copyto to keep mu_tilde pointing at the shared buffer
    omega_tilde, n_tilde = construct_tilde_vectors(mumid, dm2, average_energy, number_dist)

    #----------
    # loop over radii
    #----------
    loopstart = time.time()
    ir_list = range(ir_start, ir_stop+1)
    eigenvalues = []
    kgrid_list = []
    for ir in ir_list:
        start = time.time()

        #----------
        # construct stability matrix
        #----------
        set_stability_matrix_nok(n_tilde[ir], omega_tilde[ir], Ve[ir], phi0[ir], phi1[ir])

        #----------
        # loop over k points
        #----------
        kgrid = build_k_grid(phi0[ir], numb_k, min_ktarget_multiplier, max_ktarget_multiplier)
        with mp.Pool(processes=nthreads) as pool:
            eigenvalues_thisr = pool.map(eigenvalues_single_k, kgrid)
    
        eigenvalues.append(eigenvalues_thisr)
        kgrid_list.append(kgrid)
        
        #----------
        # write info to screen
        #----------
        end = time.time()
        print(" ir="+str(ir),
              " time="+"{:.2e}".format(end-start)+"s.",
              " ktarget="+"{:.2e}".format(phi0[ir]),
              " Eavg="+"{:.1f}".format(average_energy[ir]/MeV)+"MeV",
              " minR="+"{:.2e}".format(np.min(np.real(eigenvalues_thisr))),
              " maxR="+"{:.2e}".format(np.max(np.real(eigenvalues_thisr))),
              " minI="+"{:.2e}".format(np.min(np.imag(eigenvalues_thisr))),
              " maxI="+"{:.2e}".format(np.max(np.imag(eigenvalues_thisr))))
    loopend=time.time()
    print("total time ="+str(loopstart-loopend)+" seconds")        

    #----------
    # get mu_tilde for output
    #----------
    S_nok, mu_tilde = get_shared_numpy_arrays()

    #----------
    # set output filename
    #----------
    output_filename = input_filename[:-3]
    output_filename += "_dm"+"{:.2e}".format(dm2/eV**2)
    output_filename += "_"+distribution_interpolator
    output_filename += str(target_resolution)
    output_filename += "_eigenvalues.h5"
    print("Writing",output_filename)

    #----------
    # output data
    #----------
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
    fout["kgrid (erg)"] = kgrid_list
    fout.close()

if __name__ == '__main__':
    single_file(input_filename)
