import numpy as np
import scipy.optimize

# compute inverse Langevin function for closure
# Levermore 1984 Equation 20
def function(Z,fluxfac):
    return (1./np.tanh(Z) - 1./Z) - fluxfac
def dfunctiondZ(Z,fluxfac):
    return 1./Z**2 - 1./np.sinh(Z)**2
def get_Z(fluxfac):
    badlocs = np.where(np.abs(fluxfac)<1e-4)

    shape = np.shape(fluxfac)
    nr = shape[0]
    ns = shape[1]
    initial_guess = 1
    Z = np.zeros(shape)
    residual = 0
    for ir in range(nr):
        for s in range(ns):
            Z[ir,s] = scipy.optimize.fsolve(function, initial_guess, fprime=dfunctiondZ, args=(fluxfac[ir,s]))
            residual = max(residual, np.abs(function(Z[ir,s],fluxfac[ir,s])) )

    # when near isotropic, use taylor expansion
    # f \approx Z/3 - Z^3/45 + O(Z^5)
    Z[badlocs] = 3.*fluxfac[badlocs]

    return Z, residual

# angular factor for distribution
# Angular integral is 1
# above Levermore 1984 Equation 20
# units of 1/steradian
def angular_distribution(Z,mu):
    return 1./(4.*np.pi) * (Z/np.sinh(Z)) * np.exp(Z*mu)

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
    
    # get properties & derivatives of the old distribution function
    nr = old_dist.shape[0]
    ns = old_dist.shape[1]
    ne = old_dist.shape[2]

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
        
    # new grid structure #
    new_mumid = np.array([0.5*(new_mugrid[i]+new_mugrid[i+1]) for i in range(new_nmu)])
    new_dist = np.zeros((nr,ns,ne,new_nmu))

    # get solid angle for each bin
    dmu = np.array([new_mugrid[i+1]-new_mugrid[i] for i in range(new_nmu)])[np.newaxis,np.newaxis,:]
    dOmega = 2.*np.pi*dmu

    # compute moments of the distribution function
    max_residual = 0
    for ig in range(ne):
        dist_ig = old_dist[:,:,ig,:] # [0-r, 1-species, 2-mu] 1/ccm
        N = np.sum(dist_ig          , axis=(2)) # [0-r, 1-species] 1/ccm
        F = np.sum(dist_ig*old_mumid, axis=(2))
        fluxfac = F/N

        # get Z parameter for this group at all radii/species
        Z,residual = get_Z(fluxfac)
        max_residual = max(max_residual, residual)
        angular_dist = angular_distribution(Z[:,:,np.newaxis],new_mumid)
        new_dist[:,:,ig,:] = N[:,:,np.newaxis] * angular_dist * dOmega

    # calculate error in moments #
    new_edens = np.sum(new_dist               , axis=(1,2,3))
    new_flux  = np.sum(new_dist * new_mumid   , axis=(1,2,3))
    new_press = np.sum(new_dist * new_mumid**2, axis=(1,2,3))
    print("Minerbo max residual:", max_residual)
    print("Max relative error in net energy density:", np.max(np.abs((new_edens-old_edens)/old_edens)))
    print("Max relative error in net flux:", np.max(np.abs((new_flux-old_flux)/old_edens)))
    print("Max relative error in net pressure:", np.max(np.abs((new_press-old_press)/old_edens)))

    return new_mugrid, new_mumid, new_dist
