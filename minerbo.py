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
def redistribute_distribution(mugrid, mumid, old_dist, target_resolution):
    # store moments for comparison
    nmu = len(mumid)
    old_edens = np.sum(old_dist           , axis=(1,2,3))
    old_flux  = np.sum(old_dist * mumid   , axis=(1,2,3))
    old_press = np.sum(old_dist * mumid**2, axis=(1,2,3))
    fluxfac = old_flux / old_edens

    # get properties & derivatives of the old distribution function
    nr = old_dist.shape[0]
    ns = old_dist.shape[1]
    ne = old_dist.shape[2]

    # initialize the new distribution
    new_dist = np.zeros(np.shape(old_dist))
    
    # get solid angle for each bin
    dmu = np.array([mugrid[i+1]-mugrid[i] for i in range(nmu)])[np.newaxis,np.newaxis,:]
    dOmega = 2.*np.pi*dmu

    # compute moments of the distribution function
    max_residual = 0
    for ig in range(ne):
        dist_ig = old_dist[:,:,ig,:] # [0-r, 1-species, 2-mu] 1/ccm
        N = np.sum(dist_ig      , axis=(2)) # [0-r, 1-species] 1/ccm
        F = np.sum(dist_ig*mumid, axis=(2))
        fluxfac = F/N

        # get Z parameter for this group at all radii/species
        Z,residual = get_Z(fluxfac)
        max_residual = max(max_residual, residual)
        angular_dist = angular_distribution(Z[:,:,np.newaxis],mumid)
        new_dist[:,:,ig,:] = N[:,:,np.newaxis] * angular_dist * dOmega

    # calculate error in moments #
    new_edens = np.sum(new_dist           , axis=(1,2,3))
    new_flux  = np.sum(new_dist * mumid   , axis=(1,2,3))
    new_press = np.sum(new_dist * mumid**2, axis=(1,2,3))
    print("  MINERBO max residual:", max_residual)
    print("  MINERBO Max relative error in net energy density:", np.max(np.abs((new_edens-old_edens)/old_edens)))
    print("  MINERBO Max relative error in net flux:", np.max(np.abs((new_flux-old_flux)/old_edens)))
    print("  MINERBO Max relative error in net pressure:", np.max(np.abs((new_press-old_press)/old_edens)))

    return new_dist
