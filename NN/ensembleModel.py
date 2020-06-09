from pytorchModel import *

class ensembleLSTM(model):
    """
    ensemble of lstm models
    """
    def __init__(self,T,n,batch_size=1024,output_activation=nn.Softmax(dim=1),learning_rate=1e-5):
        super(model,self).__init__()# inherit
        # set consts
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.n = n
        self.T = T
        self.batch_size=batch_size
        self.e_learning_rate=learning_rate
        # create layer
        self.output_activation=output_activation
        self.linear = nn.Linear(self.n*(self.T),self.T).to(self.device,non_blocking=self.use_cuda)
        
    def forward(self,x):
        # forward method
        return self.output_activation(self.linear(x))
    @torch.jit.export
    def train_model(self,loss_fn,optim,epochs,data):
        self._directory_maker() # create directory to store models
        # set const
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optim = optim
        # create date
        self.create_data(data)
        # set model name
        self.model_name = f"ensemble_n={self.n}"
        # create tensorboard writer
        writer = SummaryWriter(comment=self.model_name)
        # create optimiser
        self._init_optim()
        print("Loading dataset")
        # create data loader
        xtrainTensor=torch.from_numpy(self.x_train).type(torch.float32)
        ytrainTensor=torch.from_numpy(self.y_train).type(torch.float32)
        dataset = torch.utils.data.TensorDataset(xtrainTensor,ytrainTensor)
        loader = DataLoader(dataset, batch_size=self.batch_size,drop_last=True,shuffle=True,
                    pin_memory=self.use_cuda,num_workers=2)
        xtestTensor=torch.from_numpy(self.x_test).type(torch.float32)
        ytestTensor=torch.from_numpy(self.y_test).type(torch.float32)
        valDataset = torch.utils.data.TensorDataset(xtestTensor,ytestTensor)
        valLoader = DataLoader(valDataset, batch_size=self.batch_size,drop_last=True,shuffle=False,
                    pin_memory=self.use_cuda,num_workers=2)
        # create loss array
        self._init_loss()
        old_test_loss = np.inf
        train_batches = self.x_train.shape[0] // self.batch_size
        test_batches =self.x_test.shape[0] // self.batch_size
        print("Train Loop Starting...")
        # start train loop
        for i in range(self.epochs):
            #nvtx.range_push("Epoch " + str(i+1))
            self.train()
            print(f"STEP: {i+1}/{self.epochs}")
            #nvtx.range_push("Training")
            with tqdm(total=train_batches,ascii=True,desc="Training") as pbar:#loading bar
                for idx, (local_batch, local_labels) in enumerate(loader):
                    #nvtx.range_push("Batch "+str(idx))
                    
                    #nvtx.range_push("Copying to device")
                    # get batches
                    y=local_labels.to(self.device,non_blocking=self.use_cuda)
                    x_batch=local_batch.to(self.device,non_blocking=self.use_cuda)
                    #nvtx.range_pop()
                    #nvtx.range_push("Forward pass")
                    # forward pass
                    x=self(x_batch)
                    #nvtx.range_pop()
                    
                    #nvtx.range_push("Backward pass")
                    # update loss and back prop
                    self._update_loss_train(x,y,i)
                    
                    
                    #nvtx.range_pop()
                    #nvtx.range_pop()
                    
                    
                    
                    pbar.update(1)
                
            #nvtx.range_pop()
            # average
            self.train_loss[i] = self.train_loss[i]/train_batches
            if np.isnan(self.train_loss[i,0]):# break if nan
                print("GOT NaN")
                break
            # write loss to tensorboard log and print loss
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/train'
                writer.add_scalar(name, float(self.train_loss[i,l]), i)
                print(f"{name}: {self.train_loss[i,l]}")
            
            

            #nvtx.range_push("Evaluation")
            # get val loss
            with torch.no_grad():
                self.eval() # set to eval mode
                with tqdm(total=test_batches,ascii=True,desc="Validating") as pbar:# loading bar
                    for local_batch, local_labels in valLoader:
                        #get batch
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
            # write tensorboard log print loss
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/test'
                writer.add_scalar(name, float(self.val_loss[i,l]), i)
                print(f"{name}: {self.val_loss[i,l]}")
            #nvtx.range_pop()
            # if model has the lowest val loss save 
            if self.val_loss[i,0] < old_test_loss:
                torch.save({"state_dict":self.state_dict(),
                            "n":self.n,
                            "epoch":i+1,
                            "optim":self.optimiser,
                            "val_loss":self.val_loss,
                            "batch_size": self.batch_size,
                            "T": self.T,
                            "loss_fn":self.loss_fn}, f"./best/bestModel_{self.model_name}.pth")
        # save final model
        torch.save({"state_dict":self.state_dict(),
                            "n":self.n,
                            "epoch":i+1,
                            "optim":self.optimiser,
                            "val_loss":self.val_loss,
                            "batch_size": self.batch_size,
                            "T": self.T,
                            "loss_fn":self.loss_fn}, f"./final/finalModel_{self.model_name}.pth")
        return self.train_loss,self.val_loss
    def _load_model(self,i):
        # load sub models and set to eval
        m = model(bidir=True,batch_size=512,input_dim=self.inp,T=100,dense_act=nn.ELU(),output_act=nn.Softplus(),output_act_dense=nn.Softmax()).to(self.device)
        m.load_model(f"bestModel_ensemble_model_lstm_{self.data}_{i+1}.pth")
        m.eval()
        
        return m

    def _submodel_predict(self,x):
        # load each model and get its prediction in series
        # done in series to save memory
        out = [None] * self.n
        for i in range(self.n):
            out[i] = self._load_model(i).predict(x)
        return np.hstack(tuple(out))
    def create_data(self,data):
        # set const
        self.data=data
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
        self.x_train,self.x_test,self.y_train,self.y_test =self._submodel_predict(x_train),self._submodel_predict(x_test),y_train,y_test
            
if __name__=="__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    m = ensemble(n=10,T=100).to(device)
    m(torch.randn((1024,10*101)).to(device))
    train_loss,val_loss=m.train_model("EDIV2","adam",500)
    with open('train_loss.pickle', 'wb') as f:
        pickle.dump(train_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('val_loss.pickle', 'wb') as f:
        pickle.dump(val_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
