import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from scipy.stats import norm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import h5py
import time
#from copy import deepcopy
import pickle
import gc
from torch.distributions.normal import Normal
from cells import LSTMCell,PeepholeLSTMCell,GRUCell



def gaussianMix(x,points,dt):
    g1 = norm(loc=x[3],scale=x[0])
    g2 = norm(loc=x[4],scale=x[1])
    g3 = norm(loc=x[5],scale=x[2])
    return np.diff(x[6]*(g1.cdf(points))+x[7]*(g2.cdf(points))+x[8]*(g3.cdf(points)))/2

def h2_dist(x,y,ep=1e-12):
    return 2*((y.clamp(ep).sqrt()-x.clamp(ep).sqrt()).pow(2).sum(axis=tuple(np.linspace(1,x.dim()-1,x.dim()-1,dtype=np.int)))).mean()

def e_div(x,y,ep=1e-12):
    return (x*(((x.clamp(ep)/(y.clamp(ep)))).log()).pow(2)).sum(axis=tuple(np.linspace(1,x.dim()-1,x.dim()-1,dtype=np.int))).mean()

def e_div2(x,y,ep=1e-12):
    return (y*(((y.clamp(ep)/(x.clamp(ep)))).log()).pow(2)).sum(axis=tuple(np.linspace(1,x.dim()-1,x.dim()-1,dtype=np.int))).mean()

def gaussianLoss(x,y,sigma=0.5):
    dist = Normal(x,sigma)
    l = dist.log_prob(y)
    ind = l.ne(l)
    l[ind] = 0.0
    return -l.sum(axis=-1).mean()   

def polynomial(x,points):
    return  np.diff(np.polynomial.polynomial.polyval(points,x))
kld = nn.KLDivLoss(reduction="batchmean")
def MDDKLD(x,y):
    d1 = kld((x.clamp(1e-12)).log(),y)
    d2 = kld((y.clamp(1e-12)).log(),x)
    return torch.max(d1,d2)
def MDDEDIV(x,y):
    d1 = e_div(x,y)
    d2 = e_div2(x,y)
    return torch.max(d1,d2)
def LOGMSE(x,y,ep=1e-12):
    return (y.clamp(ep)/x.clamp(ep)).log().pow(2).mean()



def dataLoader(T=100,data="Gene"):
    """
    loads data from files into python and if it doesnt exist in the case 
    of the gaussian data it will generate it.
    """
    if data=="Gene":
        print(f"Using modelDataBin_{T}.hdf5")
        #load gene data
        with h5py.File(f"modelDataBin_{T}.hdf5","r") as f:
            xall = f["parameters"][:]
            yall = f["results"][:]
            scalesM =f["scalesM"][:]
            scalesS =f["scalesS"][:]
    elif data=="BiGene":
        print(f"Using modelDataBiBin_{T}.hdf5")
        #load bigene data
        with h5py.File(f"modelDataBiBin_{T}.hdf5","r") as f:
            xall = f["parameters"][:]
            yall = f["results"][:]
            scalesM =f["scalesM"][:]
            scalesS =f["scalesS"][:]
  
            
    elif data == "Gaussian":

        path = f"gaussian_{T}.hdf5"
        if os.path.exists(path):
            #load gaussian data
            print(f"Loading gaussian mixture from {path}")
            with h5py.File(path,"r") as f:
                xall=f["parameters"][:]
                scalesM =f["scalesM"][:]
                scalesS =f["scalesS"][:]
                yall = f["results"][:]
        else:

            print("Generating mixture of gaussians...")
            n = 10**5 # samples
            mu = np.random.uniform(-3,3,size=(n,3))
            var = np.random.uniform(0.1,5,size=(n,3))
            k = np.zeros((n,4))
            k[:,3] = 1
            k[:,1:3] = np.sort(np.random.uniform(0,1,size=(n,2)),axis=-1)
            k = np.diff(k)
            x = np.hstack((var,mu,k))

            points = np.linspace(-15,15,T+1)

            dt = 30/T

            # openmp parallelised calls function to get the bin probability 
            with Pool(os.cpu_count()-1) as p:
                y = p.map(partial(gaussianMix,points=points,dt=dt),x)
            y= np.array(y)
            # get scaling parameters
            scalesM = np.mean(x,axis=0)
            scalesS = np.std(x,axis=0)
            xall = scale(x) # scale input
            yall = y
            # save gaussian data
            with h5py.File(path,"w") as f:
                f.create_dataset("results",data=yall)
                f.create_dataset("scalesM",data=scalesM)
                f.create_dataset("scalesS",data=scalesS)
                f.create_dataset("parameters",data=xall)
    return xall,yall,scalesM,scalesS

class model(nn.Module):
    """
    LSTM Model class
    """
    def __init__(self,input_dim=9,
                 T=100,prev=32,lstm_size=512,device_index=0,
                 batch_size=1024,e_learning_rate=1e-5,bidir=True,
                 output_act=nn.Sigmoid(),output_act_dense=nn.Softmax(dim=1),dense_act=nn.ELU(),celltype=LSTMCell):

        super().__init__()
        # set model constants
        self.input_dim = input_dim
        self.T = T
        self.prev = prev
        self.batch_size = batch_size
        self.e_learning_rate = e_learning_rate
        self.enc_size = lstm_size
        self.output_act_type = output_act
        self.output_dense_act_type = output_act_dense
        self.dense_act_type = dense_act
        self.bidir = bidir
        self.celltype=celltype
        self.device_index = device_index
        
        self._build_net()
    def _build_net(self):
        # build layers
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{self.device_index}" if self.use_cuda else "cpu")
        self.index_tensor = torch.tensor([[[t]]*self.batch_size for t in range(self.T+1)]).to(self.device,non_blocking=self.use_cuda)
        self.input_1_1 = nn.Linear(self.input_dim,32).to(self.device,non_blocking=self.use_cuda)
        self.input_act = self.dense_act_type.to(self.device,non_blocking=self.use_cuda)
        self.input_1_2 = nn.Linear(32,64).to(self.device,non_blocking=self.use_cuda)
        self.input_1_3 = nn.Linear(64,128).to(self.device,non_blocking=self.use_cuda)
        self.input_1_4 = nn.Linear(128,256).to(self.device,non_blocking=self.use_cuda)
        self.input_1_5 = nn.Linear(256,128).to(self.device,non_blocking=self.use_cuda)
        self.input_1_x = nn.ModuleList([nn.Linear(128,64) for _ in range(self.T)]).to(self.device,non_blocking=self.use_cuda)


        self.output_layer=nn.Linear((self.T)*(self.bidir+1),self.T).to(self.device,non_blocking=self.use_cuda)
        self.output_act = self.output_act_type.to(self.device,non_blocking=self.use_cuda)
        self.loss = torch.zeros(1,dtype=torch.float32).to(self.device,non_blocking=self.use_cuda)
        # build lstm array
        if self.bidir:
            
            self.lstmArray("array1",self.input_dim,self.T,self.prev,self.enc_size,self.batch_size,self.dense_act_type,self.output_act_type )
            self.lstmArray("array2",self.input_dim,self.T,self.prev,self.enc_size,self.batch_size,self.dense_act_type,self.output_act_type )
        else:
            self.lstmArray("array1",self.input_dim,self.T,self.prev,self.enc_size,self.batch_size,self.dense_act_type,self.output_act_type )
    
    def lstmArray(self,array,input_dim=4,
                 T=100,prev=32,lstm_size=512,
                 batch_size=128,dense_act=nn.ReLU(),output_act=nn.Sigmoid(),celltype=LSTMCell):
        
        ys = [nn.Linear(self.enc_size,1) for _ in range(self.T)]
        setattr(self,array+"_ys",nn.ModuleList(ys).to(self.device,non_blocking=self.use_cuda))
        ys_sigmoid = [output_act for _ in range(self.T)]
        setattr(self,array+"_ys_sigmoid",nn.ModuleList(ys_sigmoid).to(self.device,non_blocking=self.use_cuda))
        y_prev_dense = [nn.Linear(self.prev,128) for _ in range(self.T)]
        setattr(self,array+"_y_prev_dense",nn.ModuleList(y_prev_dense).to(self.device,non_blocking=self.use_cuda))
        y_prev_dense_act = [dense_act for _ in range(self.T)]
        setattr(self,array+"_y_prev_dense_act",nn.ModuleList(y_prev_dense_act).to(self.device,non_blocking=self.use_cuda))
        #setattr(self,array+"_dropouts",nn.ModuleList([nn.Dropout(0.1) for _ in range(self.T+1)]).to(self.device,non_blocking=self.use_cuda))
        setattr(self,array+"_lstm_enc",torch.jit.script(self.celltype(192,self.enc_size)).to(self.device,non_blocking=self.use_cuda))   
    
    def _embedding_input(self,x):
        # forward function for initial dense layers
        x = self.input_1_1(x)
        x = self.input_act(x)
        x = self.input_1_2(x)
        x = self.input_act(x)
        x = self.input_1_3(x)
        x = self.input_act(x)
        x = self.input_1_4(x)
        x = self.input_act(x)
        x = self.input_1_5(x)
        x = self.input_act(x)
        return x
    def forward(self,x):

        # put through initial dense layers
        xe = self._embedding_input(x)
        # Feed through LSTM array
        ys = self._forwardLoop1(xe)
        if self.bidir: # if bidirectional feed through other lstm array and concatenate
            ys =  torch.cat((self._forwardLoop2(xe),ys),1)
        # feed through final dense layer and softmax
        x=ys
        x = self.output_layer(x)
        x = self.output_dense_act_type(x)
        return x
    
    def _forwardLoop1(self,xe):
        # forward LSTM array
        enc_state_0,enc_state_1 = self.initial_hidden() # initialise hidden layers
        # initialise list to store previous outputs       
        y_prev = [torch.zeros([self.batch_size,1],dtype=torch.float32).to(self.device,non_blocking=self.use_cuda)] * self.prev
        # initialise result tensor
        ys = torch.zeros([self.batch_size,self.T],dtype=torch.float32).to(self.device,non_blocking=self.use_cuda)

        for t in range(0,self.T):# go through time
            # feed previous results into dense layer 
            x = self.array1_y_prev_dense[t](torch.cat(tuple(y_prev),dim=1))
            x = self.array1_y_prev_dense_act[t](x)
            # feed output initial of dense layers into dense layer
            xt = self.input_1_x[t](xe)
            xt = self.input_act(xt)
            x = torch.cat((xt,x),1)# concatenate
            # feed concatenated results into lstm array
            enc_state_0,enc_state_1 = self.array1_lstm_enc(x,enc_state_0,enc_state_1)
            # put output into dense layer
            x = self.array1_ys[t](enc_state_0)
            x = self.array1_ys_sigmoid[t](x)
            ys.scatter_(1,self.index_tensor[t],x) # put results into results tensor
            # update previous tensor
            y_prev = y_prev[1:] +[x]
            
            
        return ys
    
    def _forwardLoop2(self,xe):
        enc_state_0,enc_state_1 = self.initial_hidden() # initialise hidden states of lstm
        # initialise list to store previous results
        y_prev = [torch.zeros([self.batch_size,1],dtype=torch.float32).to(self.device,non_blocking=self.use_cuda)] * self.prev
        # initialise tensor to store output
        ys = torch.zeros([self.batch_size,self.T],dtype=torch.float32).to(self.device,non_blocking=self.use_cuda)

        for t in range(0,self.T): # go through time index in for loop reverses order
            # feed previous states into dense layer
            x = self.array2_y_prev_dense[t](torch.cat(tuple(y_prev),dim=1))
            x = self.array2_y_prev_dense_act[t](x)
            # feed output of initial dense layers into dense layer
            xt = self.input_1_x[-t-1](xe)
            xt = self.input_act(xt)
            x = torch.cat((xt,x),1)#concetenate
            # feed into lstm array
            enc_state_0,enc_state_1 = self.array2_lstm_enc(x,enc_state_0,enc_state_1)
            # feed output into dense layer and apply softmax
            x = self.array2_ys[t](enc_state_0)
            x = self.array2_ys_sigmoid[t](x)
            # cast to results tensor
            ys.scatter_(1,self.index_tensor[t],x)
            # update previous results list
            y_prev = y_prev[1:] +[x]
        return ys
    
    @staticmethod
    def get_yprev23(y_prev,t,ys):
        return torch.cat((y_prev[:,1:],ys[:,t-1].unsqueeze(1)),1)
    
    def initial_hidden(self):
        # create initial states for lstm arrays
        enc_state_0 = torch.zeros([self.batch_size,self.enc_size],dtype=torch.float32).to(self.device,non_blocking=self.use_cuda)
        enc_state_1 = torch.zeros([self.batch_size,self.enc_size],dtype=torch.float32).to(self.device,non_blocking=self.use_cuda)
        return (enc_state_0,enc_state_1)
    @torch.jit.export
    def train_model(self,epochs=100,data="Gene",
                    optim="Adam",loss_fn="MSE",custom_data=None,
                    model_name=None):
        # create directory to store models
        self._directory_maker()
        # set constants
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optim = optim
        
        if custom_data is None:
            # load data 
            xall,yall=self.load_data(data=data)
            # create train and test split
            x_train, x_test,y_train,y_test = train_test_split(xall,yall,train_size=0.8)
        else:
            # if custom data is given use that
            x_train, x_test,y_train,y_test = custom_data["x_train"],custom_data["x_test"],custom_data["y_train"],custom_data["y_test"]
            self.scalesM,self.scalesS = custom_data["scalesM"],custom_data["scalesS"]
        print(f"Loading model onto {self.device}")
        print(f"Loading tensorboard writer")
        # create model name
        if model_name is None:
            oact = str(self.output_act_type).split("(",1)[0]
            dact = str(self.dense_act_type).split("(",1)[0]
            odact = str(self.output_dense_act_type).split("(",1)[0]
            celltype = self.celltype.__str__()
            t = time.strftime("%d-%m-%H-%M-%S",time.localtime())
            self.model_name = f"cell={celltype}_b={self.batch_size}_loss={self.loss_fn}_oact={oact}_dact={dact}_odact={odact}"
        else:
            self.model_name = model_name
        # start tensorboard log 
        writer = SummaryWriter(comment=self.model_name)
        # create optimiser
        self._init_optim()
        print("Loading dataset")
        # create data loaders for train and test data
        xtrainTensor=torch.from_numpy(x_train).type(torch.float32)
        ytrainTensor=torch.from_numpy(y_train).type(torch.float32)
        dataset = torch.utils.data.TensorDataset(xtrainTensor,ytrainTensor)
        loader = DataLoader(dataset, batch_size=self.batch_size,drop_last=True,shuffle=True,
                    pin_memory=self.use_cuda,num_workers=2)
        xtestTensor=torch.from_numpy(x_test).type(torch.float32)
        ytestTensor=torch.from_numpy(y_test).type(torch.float32)
        valDataset = torch.utils.data.TensorDataset(xtestTensor,ytestTensor)
        valLoader = DataLoader(valDataset, batch_size=self.batch_size,drop_last=True,shuffle=False,
                    pin_memory=self.use_cuda,num_workers=2)
        # create matrix to store loss
        self._init_loss()
        old_test_loss = np.inf
        train_batches = x_train.shape[0] // self.batch_size
        test_batches =x_test.shape[0] // self.batch_size
        del valDataset,dataset,xtestTensor,ytestTensor,x_train,x_test,y_train,y_test
        # start train loop
        print("Train Loop Starting...")
        for i in range(self.epochs):
            #nvtx.range_push("Epoch " + str(i+1))
            self.train() # put model in train mode
            print(f"STEP: {i+1}/{self.epochs}")
            #nvtx.range_push("Training")
            # train
            with tqdm(total=train_batches,ascii=True,desc="Training") as pbar:# loading bar
                for idx, (local_batch, local_labels) in enumerate(loader):
                    #nvtx.range_push("Batch "+str(idx))
                    
                    #nvtx.range_push("Copying to device")
                    # get batch and send to device
                    y=local_labels.to(self.device,non_blocking=self.use_cuda)
                    x_batch=local_batch.to(self.device,non_blocking=self.use_cuda)
                    #nvtx.range_pop()
                    #nvtx.range_push("Forward pass")
                    # forward pass
                    x=self(x_batch)
                    #nvtx.range_pop()
                    
                    #nvtx.range_push("Backward pass")
                    # update loss and backpropigate
                    self._update_loss_train(x,y,i)
                    
                    
                    #nvtx.range_pop()
                    #nvtx.range_pop()
                    
                    
                    
                    pbar.update(1)
                
            #nvtx.range_pop()
            # average
            self.train_loss[i] = self.train_loss[i]/train_batches
            if np.isnan(self.train_loss[i,0]):# if NaN stop training
                print("GOT NaN")
                break
            # update tensorbard and print
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/train'
                writer.add_scalar(name, float(self.train_loss[i,l]), i)
                print(f"{name}: {self.train_loss[i,l]}")
            
            

            #nvtx.range_push("Evaluation")
            # get val loss
            with torch.no_grad():
                self.eval() # set to eval
                with tqdm(total=test_batches,ascii=True,desc="Validating") as pbar:# loading bar
                    for local_batch, local_labels in valLoader:
                        # get batch
                        y = local_labels.to(self.device,non_blocking=self.use_cuda)
                        x_batch=local_batch.to(self.device,non_blocking=self.use_cuda)
                        # forward pass
                        pred = self(x_batch)
                        # update val loss
                        self._update_loss_val(pred,y,i)
                        pbar.update(1)
            #nvtx.range_pop()
            # average
            self.val_loss[i] = self.val_loss[i]/test_batches
            # update tensorboard and print results
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/test'
                writer.add_scalar(name, float(self.val_loss[i,l]), i)
                print(f"{name}: {self.val_loss[i,l]}")
            #nvtx.range_pop()
            # save modelif val loss is lower than previous best
            if self.val_loss[i,0] < old_test_loss:
                #nvtx.mark("Saving best model")
                torch.save({"state_dict":self.state_dict(),
                            "scalesM":self.scalesM,
                            "scalesS":self.scalesS,
                            "epoch":i+1,
                            "optim":self.optimiser,
                            "val_loss":self.val_loss,
                            "dense_act":str(self.dense_act_type),
                            "output_act":str(self.output_act_type),
                            "batch_size": self.batch_size,
                            "enc_size":self.enc_size,
                            "bidir":self.bidir,
                            "T": self.T,
                            "output_dense_act":self.output_dense_act_type,
                            "prev": self.prev,
                            "input_dim": self.input_dim,
                            "loss_fn":self.loss_fn}, f"./best/bestModel_{self.model_name}.pth")
                old_test_loss = self.val_loss[i,0]
        # save final model
        torch.save({"state_dict":self.state_dict(),
                        "scalesM":self.scalesM,
                        "scalesS":self.scalesS,
                        "epoch":self.epochs,
                        "optim":self.optimiser,
                        "val_loss":self.val_loss,
                        "dense_act":str(self.dense_act_type),
                        "output_act":str(self.output_act_type),
                        "batch_size": self.batch_size,
                        "enc_size":self.enc_size,
                        "bidir":self.bidir,
                        "T": self.T,
                        "output_dense_act":self.output_dense_act_type,
                        "prev": self.prev,
                        "input_dim": self.input_dim,
                        "loss_fn":self.loss_fn}, f"./final/finalModel_{self.model_name}.pth")
        return self.train_loss,self.val_loss
    def _directory_maker(self):
        # create directories for models
        try:
            print("Making directory for best models")
            os.mkdir("./best")
        except FileExistsError:
            print("Directory for best models exists")
        try:
            print("Making directory for final models")
            os.mkdir("./final")
        except FileExistsError:
            print("Directory for final models exists")

    def _init_optim(self):
        # create optimiser
        if self.optim == "Adam":
            self.optimiser = torch.optim.Adam(self.parameters(),
                                          lr=self.e_learning_rate,
                                          betas=(0.5,0.999))#,weight_decay=0.1
        elif self.optim == "SGD":
            self.optimiser = torch.optim.SGD(self.parameters(),
                                          lr=self.e_learning_rate)
                                          #momentum=0.1),weight_decay=0.1
    
    def _init_loss(self):
        # initialise loss functions
        self.mse_fn = nn.MSELoss()
        self._kld = nn.KLDivLoss(reduction="batchmean")
        self.div_fn = lambda x,y: self._kld(x.clamp(1e-20).log(),y)
        self.l1LossFN_t = nn.L1Loss(reduction="none")
        self.l1LossFN = nn.L1Loss()
        self.h2 = h2_dist
        self.ediv = e_div
        self.ediv2 = e_div2
        self.MDDKLD = MDDKLD
        self.MDDEDIV = MDDEDIV
        self.LOGMSE = LOGMSE
        #self.edivc = e_div_c
        #self.a_mse =lambda x,y : adj_mse(x,y)+ dist_loss(x,y)

        self.l1_max_fn = lambda x,y : self.l1LossFN_t(x,y).max(dim=-1)[0].mean()
        allLoss = np.array(["MSE","DIV","L1 MAX","L1","H2","EDIV","EDIV2","MDDKLD","MDDEDIV","LOGMSE"])
        lossFns = np.array([self.mse_fn,self.div_fn,self.l1_max_fn,self.l1LossFN,self.h2,self.ediv,self.ediv2,self.MDDKLD,self.MDDEDIV,self.LOGMSE])
        ind1 = np.where(allLoss==self.loss_fn)
        ind2 = np.where(allLoss!=self.loss_fn)
        s = allLoss.size
        self.loss_names = np.concatenate((allLoss[ind1],allLoss[ind2]))
        self.loss_fns = np.concatenate((lossFns[ind1],lossFns[ind2]))
        # create array to store loss
        self.train_loss = np.zeros((self.epochs,s))
        self.val_loss = np.zeros((self.epochs,s))
        
    def _update_loss_train(self,x,y,i):
        # get loss
        loss = self.loss_fns[0](x,y)
        self.optimiser.zero_grad()
        # backpropigate
        loss.backward()
        self.optimiser.step()
        # update loss
        self.train_loss[i,0] += float(loss.detach().cpu())
        self.train_loss[i,1:] += [float(s(x.detach(),y).cpu()) for s in self.loss_fns[1:]]
        
        
    def _update_loss_val(self,x,y,i):
        # update validation loss
        self.val_loss[i] += [float(s(x.detach(),y).cpu()) for s in self.loss_fns]
        
    def load_data(self,data="Gene"):
        #load data using data loader function
        xall,yall,self.scalesM,self.scalesS = dataLoader(T=self.T,data=data)
        return xall,yall
        
    @torch.jit.export      
    def load_model(self,PATH):
        # load model
        model = torch.load(PATH,map_location=self.device)
        self.scalesM = model["scalesM"]
        self.scalesS = model["scalesS"]
        self.output_act_type = eval("nn."+model["output_act"])
        self.dense_act_type = eval("nn."+model["dense_act"])
        self.batch_size = model["batch_size"]
        self.bidir = model["bidir"]
        self.enc_size = model["enc_size"]
        self.T = model["T"]
        self.prev = model["prev"]
        self.input_dim=model["input_dim"]
        #self._build_net()
        self.load_state_dict(model["state_dict"])

    



    @torch.jit.export
    def predict(self,x):
        # predict outputs
        self.eval()#eval mode
        #scale
        x = (np.atleast_2d(x) - self.scalesM)/self.scalesS
        p = int(np.ceil(x.shape[0] /self.batch_size))
        t = x.shape[0]
        r = self.batch_size - (t % self.batch_size)
        x = np.pad(x,((0,r),(0,0)))
        xs = np.array_split(x,p)
        res = [None] * p
        for i in range(p):# feed x in batch size chunks
            res[i] = self(torch.from_numpy(xs[i]).type(torch.float32).to(self.device,non_blocking=self.use_cuda)).detach().cpu().numpy()

        res[-1] = res[-1][:-r]
        return np.vstack(tuple(res))

if __name__ == "__main__":
    # learning_rate=1e-4
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cell_types = [LSTMCell,PeepholeLSTMCell,GRUCell]
    lstm_size=[512,1024]
    dense_act = [nn.ELU(),nn.ReLU(),nn.Tanhshrink(),nn.Tanh(),nn.Softplus()]
    output_act = [nn.Sigmoid(),nn.Softplus()]
    output_act_dense =[nn.Sigmoid(),nn.Softplus(),nn.Softmax()]
    loss = ["MSE","L1","H2","MDDKLD","MDDEDIV","LOGMSE"]
    optim = ["Adam","SGD"]
    prev = [1,4,8,16,32,64]
    bidir = [True,False]
    g = ParameterGrid({"optim":optim,"loss":loss})

    val_loss = [[None,i] for i in g] 
    train_loss = [[None,i] for i in g] 
    

    for v,i in enumerate(g):
        try:
            print(i)
            m = model(bidir=False,e_learning_rate=1e-4,batch_size=128).to(device)
            train_loss[v][0],val_loss[v][0] = m.train_model(data="Gaussian",optim=i["optim"],loss_fn=i["loss"],epochs=100)

        except Exception as inst:
            print("ERROR!!!!")
            print(type(inst))
            print(inst) 
    with open('train_loss.pickle', 'wb') as f:
        pickle.dump(train_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('val_loss.pickle', 'wb') as f:
        pickle.dump(val_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
  
        
    
