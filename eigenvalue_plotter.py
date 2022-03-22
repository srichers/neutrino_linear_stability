import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

hbar = 1.05457266e-27 # erg s
c = 2.99792458e10 # cm/s
eV = 1.60218e-12 # erg
GeV = 1e9 * eV
GF = 1.1663787e-5 / GeV**2 * (hbar*c)**3 # erg cm^3

def get_lat_lon_3vec(F):
    Fhat = F / np.sqrt(np.sum(F**2))
    latitude = np.arccos(Fhat[2]) - np.pi/2.
    longitude = np.arctan2(Fhat[1],Fhat[0])
    return latitude,longitude

# initial conditions
Nee    = 1.421954234999705e+33
Neebar = 1.9146237131657563e+33
Nxx    = 1.9645407875568215e+33
Fee    = np.array([0.0974572,    0.04217632, -0.13433261]) * Nee
Feebar = np.array([ 0.07237959,  0.03132354, -0.3446878 ]) * Neebar
Fxx    = np.array([-0.02165833,  0.07431613, -0.53545951]) * Nxx
lat_ee, lon_ee = get_lat_lon_3vec(Fee)
lat_aa, lon_aa = get_lat_lon_3vec(Feebar)
lat_xx, lon_xx = get_lat_lon_3vec(Fxx)
lat_eln, lon_eln = get_lat_lon_3vec(Fee-Feebar)


# grab data, ignoring k=0
data = h5py.File("moment_eigenvalues.h5","r")
kgrid = np.array(data["kgrid (erg)"])[:-1,:] / (hbar*c) # 1/cm
maxeigvalsI = np.max(np.imag(np.array(data["eigenvalues (erg)"])), axis=1)[:-1] / (hbar) # 1/s. maximized over the 8 eigenvalues for each k
data.close()

kmag = np.sqrt(np.sum(kgrid**2, axis=1))
khat = kgrid/kmag[:,np.newaxis]
latitude = np.arccos(khat[:,2])-np.pi/2.
longitude = np.arctan2(khat[:,1],khat[:,0])

imax = np.argmax(maxeigvalsI)
kmax = kgrid[imax]
kmaxmag = np.sqrt(np.sum(kmax**2))
kmaxhat = kmax / kmaxmag

locs = np.where(np.abs(kmag-kmaxmag)/kmag < 1e-2)
khat_same_mag = khat[locs]
maxeigvalsI_same_mag = maxeigvalsI[locs]
fig, ax = plt.subplots(1,1, figsize=(8,8),subplot_kw={'projection':"mollweide"})
sc0 = ax.scatter(longitude[locs], latitude[locs], c=maxeigvalsI_same_mag/1e10, cmap=mpl.cm.plasma, s=10)#, vmax=3)
cbar = fig.colorbar(sc0,ax=ax)
cbar.set_label(r"$\mathrm{max}(\mathrm{Im}(\Omega))\,(10^{10}\,s^{-1})$")
cbar.ax.minorticks_on()
cbar.ax.tick_params(axis='y', which='both', direction='in')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])

ax.scatter(longitude[imax],latitude[imax], c='green', s=200, marker='*')
ax.scatter(lon_ee, lat_ee, c='orange', s=200, marker='*')
ax.scatter(lon_aa, lat_aa, c='red', s=200, marker='*')
ax.scatter(lon_xx, lat_xx, c='gray', s=200, marker='*')
ax.scatter(lon_eln, lat_eln, c='pink', s=200, marker='*')

plt.savefig("eigenvalues_same_magnitude.pdf",bbox_inches='tight')


diff = np.sum(np.abs(khat-kmaxhat[np.newaxis,:]),axis=1)
locs = np.where(diff<1e-6)
kmag_same_direction = kmag[locs]
maxeigvalsI_same_direction = maxeigvalsI[locs]
plt.clf()
plt.plot(kmag_same_direction, maxeigvalsI_same_direction)
plt.savefig("eigenvalues_same_direction.pdf",bbox_inches='tight')

