# INPUTS
path_to_nulib_scripts = "/home/srichers/software/NuLib/src/scripts"
SN_data_filename = "/mnt/data/SamFlynn/lotsadata/original_data_DO/27Msun_10ms_DO.h5"
nulib_table_filename = "NuLib_rho82_temp65_ye51_ng20_ns3_version1.0_20210722.h5"

# import libraries
import h5py
import numpy as np
import sys
sys.path.append(path_to_nulib_scripts)
import interpolate_table

# constants
h = 6.6260755e-27 # erg s
erg_to_MeV = 624151.

# get energy grid
# Output text in format usable in NuLib table generation script
SNdata = h5py.File(SN_data_filename,"r")
egrid = np.array(SNdata["distribution_frequency_grid(Hz,lab)"]) * h * erg_to_MeV
emid = np.array(SNdata["distribution_frequency_mid(Hz,lab)"]) * h * erg_to_MeV
#for i in range(len(emid)):
#    print("bin_bottom("+str(i+1)+") = "+str(egrid[i]))
#    print("energies("+str(i+1)+") = "+str(emid[i]))
#    print("bin_top("+str(i+1)+") = "+str(egrid[i+1]))
#    print("bin_widths("+str(i+1)+") = "+str(egrid[i+1]-egrid[i]))
rho = np.array(SNdata["rho(g|ccm,com)"])
r = np.array(SNdata["r(cm)"])
T_MeV = np.array(SNdata["T_gas(MeV,com)"])
Ye = np.array(SNdata["Ye"])
dist = np.array(SNdata["distribution(erg|ccm,lab)"])
SNdata.close()

# get energy density and number density (in arbitrary units - just used for averaging later)
# result has indices of [radius, species, energy]
energy_dens = np.sum(dist, axis=(3,4)) # sum over phi,theta because we don't need them
number_dens = energy_dens / emid[np.newaxis,np.newaxis,:]

# create opacity arrays [radius, species, energy]
absopac  = np.zeros( (len(rho), 3, len(emid)) )
scatopac = np.zeros( (len(rho), 3, len(emid)) )

# open NuLib table
table = h5py.File(nulib_table_filename,"r")

# calculate the opacities for every species and every energy bin
for s in range(3):
    for ig in range(len(emid)):
        print("  s="+str(s+1)+"/3  ig="+str(ig+1)+"/"+str(len(emid)))
        absopac[:,s,ig] = interpolate_table.interpolate_eas(ig, s, rho, T_MeV, Ye, table, "absorption_opacity")
        scatopac[:,s,ig] = interpolate_table.interpolate_eas(ig, s, rho, T_MeV, Ye, table, "scattering_opacity")

# close the NuLib table
table.close()

# compute number-weighted averages
absopac_averaged  = np.sum(absopac  * number_dens, axis=2) / np.sum(number_dens, axis=2)
scatopac_averaged = np.sum(scatopac * number_dens, axis=2) / np.sum(number_dens, axis=2)

# output to file
fout = h5py.File("averaged_opacities.h5","w")
fout["absopac(1|cm)"] = absopac_averaged
fout["scatopac(1|cm)"] = scatopac_averaged
fout.close()
