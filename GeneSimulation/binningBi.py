from binning import *

@jit(nopython=True)
def vbinsbi(x1,x2,b1,b2,r1,r2):
    # bivariate version of binner
    # get bin edges
    bins1 = np.linspace(r1[0],r1[1],b1+1)
    bins2 = np.linspace(r2[0],r2[1],b2+1)
    # get bin of each
    dig1 = np.searchsorted(bins1, x1, side='right')
    dig2 = np.searchsorted(bins2, x2, side='right')
    res = np.empty((x1.shape[0],b1,b2),dtype=np.int32)
    for i in range(b1):# for all bins for x1
        for j in range(b2):# for all bins for x2
            res[:,i,j] = np.sum((dig1==i+1) & (dig2==j+1),axis=-1)
    s = res.sum(-1).sum(-1)
    # normalise
    s[s==0] = 1
    return res/s.reshape(x1.shape[0],1,1)


if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    t = 100 # set number of bins
    comm.Barrier()
    if rank ==0:
        tic = time.perf_counter()
    with h5py.File("modelDataBi.hdf5","r") as f:
        xP = f["resultsP"]
        xM = f["resultsM"]
        s = xP.shape
        n= s[0]
        start,end = MPE_DECOMP1D(n,nprocs,rank) # get indicies
        it = np.append(np.arange(0,end-start,768,dtype=int),[int(end-start)]) # only load 768 rows at a time
        x = np.empty((end-start,t,t),dtype="f")
        for i in range(len(it)-1): # bin data
            
            x[it[i]:it[i+1]] = vbinsbi(xM[(it[i]+start):(it[i+1]+start)],xP[(it[i]+start):(it[i+1]+start)],t,t,(0,300),(0,300))
        
        if rank ==0:# in root store reaction rates and time
            y = f["parameters"][:,np.arange(6)!=4]
            times = f["avg_time"][:]
    # send send counts and disp to root
    sendcounts = np.array(comm.gather((end-start)*t*t, root=0))
    disp = np.array(comm.gather((start)*t*t, root=0))
    if rank == 0:
        r = np.empty((n,t,t),dtype="f")
   
    else:
        r = None
    # gather results in root
    comm.Gatherv(x,[r,sendcounts,disp,MPI.FLOAT],root=0)
    comm.Barrier()
    if rank ==0:
        toc =time.perf_counter() # stop timer
        # get scaling parameters
        scalesM = np.mean(y,axis=0)
        scalesS = np.std(y,axis=0)
    
        with h5py.File(f"modelDataBiBin_{t}.hdf5","w") as f:# save binned results
            f.create_dataset("results",data=r)
            f.create_dataset("scalesM",data=scalesM)
            f.create_dataset("scalesS",data=scalesS)

            f.create_dataset("parameters",data=scale(y))
            f.create_dataset("avg_time",data=times+(toc-tic)*nprocs/(s[0]))