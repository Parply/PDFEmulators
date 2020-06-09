import gillespy2 as gp
from gillespy2.solvers.cpp import SSACSolver
import numpy as np
import os
from sklearn.model_selection import ParameterGrid
import mpi4py.MPI as MPI
import h5py
import time


class geneModel(gp.Model):
    """
    Gene Model gillespie simulator
    """  
    def __init__(self):
        gp.Model.__init__(self,name='geneModel')# inherit

        self.paramGrid = None
        self.n=0
        
    def setValues(self,n):
        # set model parameters
        parameters = self.paramGrid[n]
        self.delete_all_parameters()
        self.delete_all_reactions()
        self.delete_all_species()

        # set reaction rates
        k0 = gp.Parameter(name='k0', expression=parameters[0])
        k1 = gp.Parameter(name='k1', expression=parameters[1])
        v0 = gp.Parameter(name='v0', expression=parameters[2])
        v1 = gp.Parameter(name='v1', expression=parameters[-1])#v1Value
        d0 = gp.Parameter(name='d0', expression=1.0)#1.0
        d1 = gp.Parameter(name='d1', expression=parameters[3])
        self.add_parameter([k0,k1,v0,v1,d0,d1])
        
        # define species
        ip = gp.Species(name='ip', initial_value=0)
        ap = gp.Species(name='ap', initial_value=1)
        m = gp.Species(name='m', initial_value=int(parameters[-2]))
        p = gp.Species(name='p', initial_value=100)
        self.add_species([ip,ap,m,p])
        
        # define reactions
        r_k0 = gp.Reaction(name="on", rate=k0, reactants={ip:1}, products={ap:1})
        r_k1 = gp.Reaction(name="off", rate=k1, reactants={ap:1}, products={ip:1})
        r_v0 = gp.Reaction(name="m_creation", rate=v0, reactants={ap:1}, products={m:1,ap:1})
        r_d0 = gp.Reaction(name="m_destruction", rate=d0, reactants={m:1}, products={})
        r_v1 = gp.Reaction(name="p_creation", rate=v1, reactants={m:1}, products={p:1,m:1})
        r_d1 = gp.Reaction(name="p_destruction", rate=d1, reactants={p:1}, products={})
        self.add_reaction([r_k0,r_k1,r_v0,r_d0,r_v1,r_d1])
        
        # get independent sample dists
        self.sample = int(np.maximum(np.ceil((1/parameters[np.arange(len(parameters))!=4])).max(),1))
        # get end time
        self.t = self.sample*10**5
        self.timespan(np.linspace(0,self.t,10**5+1))# set times to return results for
        
    def generateParams(self,n):
        # set parameters
        self.n = n
        k01 = np.power(10,np.random.uniform(-2,1,(self.n,2)))
        d1v0 = np.random.uniform(0.01,1,(self.n,2))
        self.paramGrid = np.ones((self.n,6))
        self.paramGrid[:,0:2] = k01 
        self.paramGrid[:,2:4] = d1v0
        self.paramGrid[:,-2] =  self.paramGrid[:,2]* self.paramGrid[:,0]/(( self.paramGrid[:,0]+ self.paramGrid[:,1]))
        self.paramGrid[:,-1]= 100*self.paramGrid[:,3]/self.paramGrid[:,-2]
        

    
def MPE_DECOMP1D(n,numprocs,myid):
    """
    Python function to get start end indicies for mpi proc

    """
    nlocal = n // numprocs
    s = myid * nlocal
    deficit = n % numprocs
    s = s + min(myid,deficit)
    if myid < deficit:
        nlocal = nlocal +1
    e = s + nlocal 
    if e > n or myid==numprocs-1:
        e = n 
    return s,e


        
if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    model = geneModel()
    n = 10**5 # number of samples
    if rank==0:# generate parameters on root
        model.generateParams(n)
        temp =model.paramGrid
    else:
        temp=None
    
    
    
    start,end = MPE_DECOMP1D(n,nprocs,rank) # get indicies
    temp = comm.bcast(temp,root=0) # bcast parameters to all processes 
    if rank != 0:
        model.paramGrid = temp[start:end] # set paramGrids of model
    del temp
    
    sample = np.empty((end-start,10**5),dtype="f") # create array to store results
    if rank ==0:# start timer
        tic = time.perf_counter()
    for i in range(end-start):
        
        model.setValues(i)# set rates
        sample[i]= model.run(number_of_trajectories=1,solver=SSACSolver())["p"][1:]# run gillespie and store protein series
    # gather indicies on root
    ends = np.array(comm.gather(int(end),root=0))
    starts = np.array(comm.gather(int(start),root=0))
    comm.Barrier()
    if rank == 0:
        toc = time.perf_counter()# stop timer
        temp = np.empty((n,10**5),dtype="f")
        temp[start:end] = sample
        with h5py.File("modelData.hdf5","w") as f:# create file to store results
            f.create_dataset("results",data=temp,dtype="f")
            f.create_dataset("parameters",data=model.paramGrid)
            f.create_dataset("avg_time",data=np.array([(toc-tic)*nprocs/n]))
        del temp
    for i in range(1,nprocs):# send samples from each process individually due to Gather limit on elements
        if rank == i:
            comm.Send([sample,MPI.FLOAT],dest=0)
        elif rank == 0:
            temp = np.empty((ends[i]-starts[i],10**5),dtype="f")
            comm.Recv(temp,source=i)
            with h5py.File("modelData.hdf5","a") as f:# save sample from proc
                f["results"][starts[i]:ends[i]] = temp
    

    
    
        