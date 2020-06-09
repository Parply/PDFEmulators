import numpy as np 
import h5py
import time
from tqdm import tqdm
from sklearn.preprocessing import scale
import mpi4py.MPI as MPI
from GeneModelSimulation import MPE_DECOMP1D
from numba import jit

@jit(nopython=True)
def vbins(x,b,r):
    
    bins = np.linspace(r[0],r[1],b+1)# get bins
    dig = np.searchsorted(bins, x, side='right')# get bins
    res = np.empty((b,x.shape[0]),dtype=np.int32)
    for i in range(b):# count numbers in each bin
        res[i] = np.sum(dig==i+1,axis=-1)
    res = res.T
    s = res.sum(-1)
    # stop division by zero and normalise
    s[s==0] = 1
    return res/s.reshape(x.shape[0],1)



if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    t = 100 # number of time points
    comm.Barrier()
    if rank ==0:# start timer
        tic = time.perf_counter()
    with h5py.File("modelData.hdf5","r") as f:
        tx = f["results"]
        s = tx.shape
        n= s[0]
        start,end = MPE_DECOMP1D(n,nprocs,rank)# get index
        it = np.append(np.arange(0,end-start,1536,dtype=int),[int(end-start)])# only load 1536 rows at once due to mem constraints
        x = np.empty((end-start,t),dtype="f")
        for i in range(len(it)-1):
            
            x[it[i]:it[i+1]] = vbins(tx[(it[i]+start):(it[i+1]+start)],t,(0,300))# bin
        
        if rank ==0:# in root store reaction rates and time
            y = f["parameters"][:,np.arange(6)!=4]
            times = f["avg_time"][:]
    # gather send counts
    sendcounts = np.array(comm.gather((end-start)*t, root=0))
    disp = np.array(comm.gather((start)*t, root=0))
    if rank == 0:
        r = np.empty((n,t),dtype="f")
   
    else:
        r = None
    # use gatherv to put results in r
    comm.Gatherv(x,[r,sendcounts,disp,MPI.FLOAT],root=0)
    comm.Barrier()
    if rank ==0:
        
        toc =time.perf_counter() # stop timer
        # get scaling parameters
        scalesM = np.mean(y,axis=0) 
        scalesS = np.std(y,axis=0)
    
        with h5py.File(f"modelDataBin_{t}.hdf5","w") as f: # save binned data
            f.create_dataset("results",data=r)
            f.create_dataset("scalesM",data=scalesM)
            f.create_dataset("scalesS",data=scalesS)

            f.create_dataset("parameters",data=scale(y))
            f.create_dataset("avg_time",data=times+(toc-tic)*nprocs/(s[0]))