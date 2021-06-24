import numpy as np

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
    print("  DO Max relative error in net energy density:", np.max(np.abs((new_edens-old_edens)/old_edens)))
    print("  DO Max relative error in net flux:", np.max(np.abs((new_flux-old_flux)/old_edens)))
    print("  DO Max relative error in net pressure:", np.max(np.abs((new_press-old_press)/old_edens)))

    return new_mugrid, new_mumid, new_dist

