from MIXD import *

class ensembleMIX(MixD):
    """
    model to make ensemble of n mixture models
    """
    def __init__(self,n,T,bins,num_mix,mix_dist="Gaussian",batch_size=1024,p_act=(Exponential(),Exponential(),Exponential(),Exponential()),dense_act=nn.ELU(),learning_rate=1e-5):
        super(model,self).__init__()#inherit
        # set consts
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.n = n
        self.T = T
        self.bini = bins
        self.batch_size=batch_size
        self.e_learning_rate=learning_rate
        self.dense_act=dense_act
        self.num_mix = num_mix
        const = 3 if mix_dist != "bivariateGammaPoisson" else 5
        self.input_dim = const*self.n*self.num_mix
        # create layers
        self.embedding1 = nn.Linear(self.input_dim,self.input_dim*2).to(self.device,non_blocking=self.use_cuda)
        self.embedding1Act = dense_act
        self.embedding2 = nn.Linear(self.input_dim*2,self.input_dim*4).to(self.device,non_blocking=self.use_cuda)
        self.embedding2Act = dense_act
        self.embedding3 = nn.Linear(self.input_dim*4,self.input_dim*8).to(self.device,non_blocking=self.use_cuda)
        self.embedding3Act = dense_act
        self.embedding4 = nn.Linear(self.input_dim*8,self.input_dim*4).to(self.device,non_blocking=self.use_cuda)
        self.embedding4Act = dense_act
        self.embedding5 = nn.Linear(self.input_dim*4,self.input_dim).to(self.device,non_blocking=self.use_cuda)
        self.embedding5Act = dense_act
        self.p0_act = p_act[0]
        self.p1_act = p_act[1]
        self.bivariate = False
        
        
        self.p0 = nn.Linear(self.input_dim,self.num_mix).to(self.device,non_blocking=self.use_cuda)
        self.p1 = nn.Linear(self.input_dim,self.num_mix).to(self.device,non_blocking=self.use_cuda)

        self.ai = nn.Linear(self.input_dim,self.num_mix)
        self.aiAct = nn.Softmax(dim=1)
        self.mix = torch.arange(0,self.num_mix,dtype=torch.int).to(self.device,non_blocking=self.use_cuda)
        self.mix_dist = mix_dist
        # set mixture distribution and bins
        if mix_dist == "Gaussian":
            self.distPredict = self.normalPredict
            self.bins = bins.to(self.device,non_blocking=self.use_cuda)
        elif mix_dist == "GammaPoisson":
            self.distPredict = self.gammaPoissonPredict
            self.bins = (bins-1).to(self.device,non_blocking=self.use_cuda)
        elif mix_dist == "PoissonLogNormal":
            self.distPredict = self.poissonLogNormalPredict
            self.bins = (bins-1).to(self.device,non_blocking=self.use_cuda)
        elif mix_dist =="bivariateGammaPoisson":
            self.distPredict = self.biGammaPoissonPredict
            self.binsx = (bins[0]-1).to(self.device,non_blocking=self.use_cuda)
            self.binsy = (bins[1]-1).to(self.device,non_blocking=self.use_cuda)
            # create additional layers required by this distribution
            self.p2 = nn.Linear(self.input_dim,self.num_mix).to(self.device,non_blocking=self.use_cuda)
            self.p3 = nn.Linear(self.input_dim,self.num_mix).to(self.device,non_blocking=self.use_cuda)
            self.p2_act = p_act[2]
            self.p3_act = p_act[3]
            self.bivariate = True
        else:
            raise ValueError("Invalid mixture distribution")
        
        
    def forward(self,x):
        # forward method
        # initial dense layers
        x = self.embedding1Act(self.embedding1(x))
        x = self.embedding2Act(self.embedding2(x))
        x = self.embedding3Act(self.embedding3(x))
        x = self.embedding4Act(self.embedding4(x))
        x = self.embedding5Act(self.embedding5(x))
        # predict parameters
        if self.bivariate:
            p0 = self.p0_act(self.p0(x))
            p1 = self.p1_act(self.p1(x))
            p2 = self.p2_act(self.p2(x))
            p3 = self.p3_act(self.p3(x))
            p = (p0,p1,p2,p3)
        else:
            p0 = self.p0_act(self.p0(x))
            p1 = self.p1_act(self.p1(x))
            p = (p0,p1)
        # predict weights
        pr = self.aiAct(self.ai(x))
        
        return p,pr  
    @torch.jit.export
    def train_model(self,epochs=100,optim="Adam",data="gau",loss_fn="H2"):
        # create directory to store models
        self._directory_maker()
        # set consts
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.optim = optim
        # get data
        self.create_data(data)
        self.model_name = f"mix_{self.mix_dist}_{loss_fn}_{optim}_{self.num_mix}"
        # set tensorboard writer
        writer = SummaryWriter(comment=self.model_name)
        # initialise optimiser
        self._init_optim()
        print("Loading dataset")
        
        # set number of workers for data loaders
        if self.mix_dist=="bivariateGammaPoisson":
            num_workers = 0
        else:
            num_workers = 2


        # create data loaders
        xtrainTensor=torch.from_numpy(self.x_train).type(torch.float32)
        ytrainTensor=torch.from_numpy(self.y_train).type(torch.float32)
        dataset = torch.utils.data.TensorDataset(xtrainTensor,ytrainTensor)
        loader = DataLoader(dataset, batch_size=self.batch_size,drop_last=True,shuffle=True,
                    pin_memory=self.use_cuda,num_workers=num_workers)
        xtestTensor=torch.from_numpy(self.x_test).type(torch.float32)
        ytestTensor=torch.from_numpy(self.y_test).type(torch.float32)
        valDataset = torch.utils.data.TensorDataset(xtestTensor,ytestTensor)
        valLoader = DataLoader(valDataset, batch_size=self.batch_size,drop_last=True,shuffle=False,
                    pin_memory=self.use_cuda,num_workers=num_workers)
        # create array to store loss
        self._init_loss()
        old_test_loss = np.inf
        train_batches = self.x_train.shape[0] // self.batch_size
        test_batches =self.x_test.shape[0] // self.batch_size
        del valDataset,dataset,xtestTensor,ytestTensor,xtrainTensor,ytrainTensor
        # start train loop
        print("Train Loop Starting...")
        for i in range(self.epochs):
            #nvtx.range_push("Epoch " + str(i+1))
            self.train()# set train mode
            print(f"STEP: {i+1}/{self.epochs}")
            with tqdm(total=train_batches,ascii=True,desc="Training") as pbar:# loading bar
                for idx, (local_batch, local_labels) in enumerate(loader):
                    #nvtx.range_push("Batch "+str(idx))
                    
                    #nvtx.range_push("Copying to device")
                    # load batch
                    y=local_labels.to(self.device,non_blocking=self.use_cuda)
                    x_batch=local_batch.to(self.device,non_blocking=self.use_cuda)
                    #nvtx.range_pop()
                    #nvtx.range_push("Forward pass")
                    # forward pass
                    x=self(x_batch)
                    #nvtx.range_pop()
                    
                    #nvtx.range_push("Backward pass")
                    #evalate mixture and update loss and back prop
                    self._update_loss_train(self.distPredict(x),y,i)
                    
                    
                    #nvtx.range_pop()
                    #nvtx.range_pop()
                    
                    
                    
                    pbar.update(1)
                
            #nvtx.range_pop()
            # average
            self.train_loss[i] = self.train_loss[i]/train_batches
            if np.isnan(self.train_loss[i,0]):# if nan break
                print("GOT NaN")
                break
            # write tensorboard log and print loss
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/train'
                writer.add_scalar(name, float(self.train_loss[i,l]), i)
                print(f"{name}: {self.train_loss[i,l]}")
            # get val loss
            with torch.no_grad():
                self.eval()# set eval mode
                with tqdm(total=test_batches,ascii=True,desc="Validating") as pbar:# loading bar
                    for local_batch, local_labels in valLoader:
                        # load batch 
                        y = local_labels.to(self.device,non_blocking=self.use_cuda)
                        x_batch=local_batch.to(self.device,non_blocking=self.use_cuda)
                        # forward pass
                        pred = self(x_batch)
                        # evaluate mixture and get loss
                        self._update_loss_val(self.distPredict(pred),y,i)
                        pbar.update(1)
            #nvtx.range_pop()
            # average
            self.val_loss[i] = self.val_loss[i]/test_batches
            # update tensorbard and print loo
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/test'
                writer.add_scalar(name, float(self.val_loss[i,l]), i)
                print(f"{name}: {self.val_loss[i,l]}")
            #nvtx.range_pop()
            # if it has the lowest val loss print
            if self.val_loss[i,0] < old_test_loss:
                #nvtx.mark("Saving best model")
                torch.save({"state_dict":self.state_dict(),
                            "scalesM":self.scalesM,
                            "scalesS":self.scalesS,
                            "epoch":i+1,
                            "optim":self.optimiser,
                            "val_loss":self.val_loss,
                            "T": self.T,
                            "input_dim": self.input_dim}, f"./best/mdEnsBestModel_{self.model_name}.pth")
                old_test_loss = self.val_loss[i,0]
        # save final model
        torch.save({"state_dict":self.state_dict(),
                        "scalesM":self.scalesM,
                        "scalesS":self.scalesS,
                        "epoch":self.epochs,
                        "optim":self.optimiser,
                        "val_loss":self.val_loss,
                        "T": self.T,
                        "input_dim": self.input_dim}, f"./final/mdRnsFinalModel_{self.model_name}.pth")
        return self.train_loss,self.val_loss
    def _load_model(self,i):
        # load sub model and put in eval mode
        m = MixD(T=self.T,input_dim=self.inp,num_mix=self.num_mix,bins=self.bini,mix_dist=self.mix_dist,dense_act=nn.Tanh()).to(self.device)
        m.load_model(f"mdBestModel_ensemble_model_{self.data}_{i+1}.pth")
        m.eval()
        
        return m

    def _submodel_predict(self,x,n):
        # get prediction of n submodels in series
        out = [None] * n
        for i in range(n):
            out[i] = self._load_model(i).predict(x)
        return np.hstack(tuple(out))
    def create_data(self,data):
        # set const
        self.data = data
        if data == "gau":
            self.inp = 9
        else:
            self.inp = 5
        # load data
        with h5py.File(f"ensemble_data_{data}.hdf5","r") as f:
            x_train = f["x_train"][:]
            x_test = f["x_test"][:]
            y_train = f["y_train"][:]
            y_test = f["y_test"][:]
            
        self.x_train,self.x_test,self.y_train,self.y_test =self._submodel_predict(x_train,self.n),self._submodel_predict(x_test,self.n),y_train,y_test
        xall = np.vstack((self.x_train,self.x_test))
        self.scalesM = np.mean(xall,0)
        self.scalesS = np.std(xall,0)
        self.x_train,self.x_test = (self.x_train-self.scalesM)/self.scalesS,(self.x_test-self.scalesM)/self.scalesS
        
    