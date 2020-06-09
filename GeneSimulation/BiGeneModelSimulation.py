from GeneModelSimulation import *
class biGeneModel(geneModel):
    """
    Joint distribution of mrna and proteins
    """
    def __init__(self):
        super().__init__()# inherit
    def generateParams(self,n):
        # generate parameters
        self.n = n
        k01 = np.power(10,np.random.uniform(-2,1,(self.n,2)))
        d1 = np.random.uniform(0.01,1,(self.n))
        self.paramGrid = np.ones((self.n,6))
        self.paramGrid[:,0:2] = k01 
        self.paramGrid[:,2] = 50*(( self.paramGrid[:,0]+ self.paramGrid[:,1]))/self.paramGrid[:,0]
        self.paramGrid[:,3] = d1
        self.paramGrid[:,-2] = 50
        self.paramGrid[:,-1]= 100*self.paramGrid[:,3]/self.paramGrid[:,-2]
        
      
if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    model = biGeneModel()
    n = 10**5 # number of samples
    if rank==0:
        model.generateParams(n)# generate params on root
        temp =model.paramGrid
    else:
        temp=None
    
    
    
    start,end = MPE_DECOMP1D(n,nprocs,rank) # get indicies
    temp = comm.bcast(temp,root=0) #bcast params
    if rank != 0:
        model.paramGrid = temp[start:end] # set param grid
    del temp
    # create array to store proteins and mrna
    sampleP = np.empty((end-start,10**5),dtype="f")
    sampleM = np.empty((end-start,10**5),dtype="f")
    if rank ==0:# time
        tic = time.perf_counter()
    for i in range(end-start):
        
        model.setValues(i)# set rates
        res =  model.run(number_of_trajectories=1,solver=SSACSolver())# run gillespie
        # store samples
        sampleP[i]=res["p"][1:]
        sampleM[i]=res["m"][1:]
    # get indicies on root
    ends = np.array(comm.gather(int(end),root=0))
    starts = np.array(comm.gather(int(start),root=0))
    comm.Barrier()
    if rank == 0:
        toc = time.perf_counter()#stop timer
        temp1 = np.empty((n,10**5),dtype="f")
        temp1[start:end] = sampleP
        temp2 = np.empty((n,10**5),dtype="f")
        temp2[start:end] = sampleM
        # save data
        with h5py.File("modelDataBi.hdf5","w") as f:
            f.create_dataset("resultsP",data=temp1,dtype="f")
            f.create_dataset("resultsM",data=temp2,dtype="f")
            f.create_dataset("parameters",data=model.paramGrid)
            f.create_dataset("avg_time",data=np.array([(toc-tic)*nprocs/n]))
        del temp1
        del temp2
    for i in range(1,nprocs):# we send from each proc seperatly rather than using Gather as Gather has a limit on the number of elements 
        if rank == i:# send proteins from i to root
            comm.Send([sampleP,MPI.FLOAT],dest=0)
        elif rank == 0:
            temp = np.empty((ends[i]-starts[i],10**5),dtype="f")
            comm.Recv(temp,source=i)
            with h5py.File("modelDataBi.hdf5","a") as f:# save into file
                f["resultsP"][starts[i]:ends[i]] = temp
        if rank == i:# send mrna from i to root
            comm.Send([sampleM,MPI.FLOAT],dest=0)
        elif rank == 0:
            temp = np.empty((ends[i]-starts[i],10**5),dtype="f")
            comm.Recv(temp,source=i)
            with h5py.File("modelDataBi.hdf5","a") as f:# save into file
                f["resultsM"][starts[i]:ends[i]] = temp