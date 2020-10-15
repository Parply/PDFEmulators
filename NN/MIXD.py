from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.poisson import Poisson
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions.categorical import Categorical
from numbers import Number
from pytorchModel import *
import itertools


# define these wrappers so functions have a readable name when converted to a string

class Exponential():
    def __init__(self):
        return None
    def __call__(self,x):
        return x.exp().clamp(1e-12)
    def __str__(self):
        return "Exponential"

class Clamped():
    def __init__(self,ep=1e-12):
        self.ep = ep
        return None
    def __call__(self,x):
        return x.clamp(self.ep)
    def __str__(self):
        return "Clamped"



class gammaPoisson(ExponentialFamily):
    """
    create gamma-poisson distribution class
    """


    arg_constraints = {'alpha': constraints.positive,
                       'beta': constraints.positive}
    support = constraints.nonnegative_integer
    def __init__(self,alpha,beta,validate_args=None):
        # set parameters
        self.alpha,self.beta = broadcast_all(alpha,beta)
        if isinstance(alpha, Number):
            batch_shape = torch.Size()
            self.device = torch.device("cpu")
            self.batch_size = torch.tensor([1])
        else:
            batch_shape = self.alpha.size()
            self.device = self.alpha.device
            self.batch_size = batch_shape[:-1]
        
        super(gammaPoisson, self).__init__(batch_shape, validate_args=validate_args)
    def cdf(self,x):
        if self._validate_args:
            self._validate_sample(x)
        x, = broadcast_all(x)
        n = x.floor().long() # cast to integers
        # create tensor of all numbers up to max
        r = torch.linspace(0,n.max(),n.max()+1).to(self.device)
        res = torch.zeros((*self.batch_size,n.max()+2)).to(self.device)
        t1 = (r+self.alpha).lgamma()
        t2 = -self.alpha.lgamma()-(r+1.0).lgamma()
        t3 = self.alpha*self.beta.log() - (r+self.alpha)*(1.0+self.beta).log()
        # evaluate cdf and take cumsum
        res[...,1:] = (t1+t2+t3).exp().cumsum(dim=-1).clamp(0.0,1.0)
        # return relevant values
        ind = n +1
        ind[n<0] = 0
        return res[...,ind]
    @property
    def mean(self):
        return self.alpha/self.beta

    @property
    def variance(self):
        return (self.alpha * (1.0 + self.beta)+1.0)/self.beta.pow(2)
    
class bivariateGammaPoisson(ExponentialFamily):
    """
    bivariate multinomial-dirichlet gamma-poisson distribution class
    """
    def __init__(self,alpha,beta,theta0,theta1,validate_args=None):
        # initialise parameters of distribution
        self.alpha,self.beta,self.theta0,self.theta1 = broadcast_all(alpha,beta,theta0,theta1)
        self.alpha,self.beta,self.theta0,self.theta1 = self.alpha.reshape(self.alpha.shape[0],1,1),self.beta.reshape(self.alpha.shape[0],1,1),self.theta0.reshape(self.alpha.shape[0],1,1),self.theta1.reshape(self.alpha.shape[0],1,1)        
        self.thetaT = self.theta0+self.theta1
        if isinstance(alpha, Number):
            batch_shape = torch.Size()
            self.device = torch.device("cpu")
            self.batch_size = torch.tensor([1])
        else:
            batch_shape = self.alpha.size()
            self.device = self.alpha.device
            self.batch_size = batch_shape[:2]
        super(bivariateGammaPoisson, self).__init__(batch_shape, validate_args=validate_args)
    def cdf(self,x,y):
        # returns cdf
        if self._validate_args:
            self._validate_sample(x)
            self._validate_sample(y)
        x,y = broadcast_all(x,y)
        # cast to integers
        nx = x.floor().long()
        ny = y.floor().long()
        
        # create grid of all points up to max
        rx = torch.linspace(0,nx.max(),nx.max()+1).to(self.device)
        ry = torch.linspace(0,ny.max(),ny.max()+1).to(self.device)
        rx,ry=torch.meshgrid([rx,ry])
        
        # calculate log pdf
        rx = rx.repeat(self.batch_size[0],1).reshape(self.batch_size[0],*rx.shape)
        ry = ry.repeat(self.batch_size[0],1).reshape(self.batch_size[0],*ry.shape)
        t1 = self.alpha*(self.beta.log()-(self.beta+1).log() ) + (self.alpha+rx+ry).lgamma()
        t2 = - self.alpha.lgamma() - (rx+ry+1).lgamma() - (rx+ry)*(self.beta+1).log()
        t3 = (rx+ry+1).lgamma() - (rx+1).lgamma()-(ry+1).lgamma() + self.thetaT.lgamma() + (rx+self.theta0).lgamma()+(ry+self.theta1).lgamma()
        t4 = -(self.thetaT+rx+ry).lgamma() - self.theta0.lgamma() - self.theta1.lgamma()
        res = torch.zeros(self.batch_size[0],nx.max()+2,ny.max()+2).to(self.device)
        # using cumsum get cdf
        res[:,1:,1:] = (t1+t2+t3+t4).exp().cumsum(-1).cumsum(-2)
        # get relevant indicies
        ind1 = nx +1
        ind1[nx<0] = 0
        ind2 = ny +1
        ind2[ny<0] = 0
        return res[:,ind1,ind2].clamp(0.0,1.0)
    def _cdf(self,x,y):
        # get cdf across bins
        if self._validate_args:
            self._validate_sample(x)
            self._validate_sample(y)
        # cast to integers
        x,y = broadcast_all(x,y)
        nx = x.floor().long()
        ny = y.floor().long()
        # get bin sizes
        sx = (x[1:] - x[:-1]).long()
        sy = (y[1:] - y[:-1]).long()

        # get indicies to sum over
        indx = torch.zeros(sx.sum()+1, dtype=torch.long).to(self.device)
        indx[torch.cumsum(sx, dim=0)] = torch.tensor(1).to(self.device)
        indx = torch.cumsum(indx, dim=0)[:-1]

        indy = torch.zeros(sy.sum()+1, dtype=torch.long).to(self.device)
        indy[torch.cumsum(sy, dim=0)] = torch.tensor(1).to(self.device)
        indy = torch.cumsum(indy, dim=0)[:-1]
        #create grid of points up to max
        rx = torch.linspace(0,nx.max(),nx.max()+1).to(self.device)
        ry = torch.linspace(0,ny.max(),ny.max()+1).to(self.device)
        rx,ry=torch.meshgrid([rx,ry])
        # calculate log pdf
        rx = rx.repeat(self.batch_size[0],1).reshape(self.batch_size[0],*rx.shape)
        ry = ry.repeat(self.batch_size[0],1).reshape(self.batch_size[0],*ry.shape)
        t1 = self.alpha*(self.beta.log()-(self.beta+1).log() ) + (self.alpha+rx+ry).lgamma()
        t2 = - self.alpha.lgamma() - (rx+ry+1).lgamma() - (rx+ry)*(self.beta+1).log()
        t3 = (rx+ry+1).lgamma() - (rx+1).lgamma()-(ry+1).lgamma() + self.thetaT.lgamma() + (rx+self.theta0).lgamma()+(ry+self.theta1).lgamma()
        t4 = -(self.thetaT+rx+ry).lgamma() - self.theta0.lgamma() - self.theta1.lgamma()
        temp = torch.zeros(self.batch_size[0],len(x)-1,ny.max()+1).to(self.device)
        # sum pdf over x for bins
        temp.index_add_(1,indx,(t1+t2+t3+t4).exp())
        res = torch.zeros(self.batch_size[0],len(x)-1,len(y)-1).to(self.device)
        # sum pdf over y for bins
        res.index_add_(2,indy,temp)#.cumsum(-1).cumsum(-2)
        
        
        return res.clamp(0.0,1.0)
class poissonLogNormal(ExponentialFamily):
    """
    poisson log-normal distribution class
    """
    def __init__(self,mu,sigma,quad_points=8,validate_args=None):
        # set parameters
        self.mu,self.sigma = broadcast_all(mu,sigma)
        self.quad_points = quad_points
        if isinstance(mu, Number):
            batch_shape = torch.Size()
            self.device = torch.device("cpu")
            self.batch_size = torch.tensor([1])
        else:
            batch_shape = self.mu.size()
            self.device = self.mu.device
            self.batch_size = batch_shape[0]
        # get quadrature points
        grid = self.quadrature()
        # create poissons
        self.poisson = Poisson(grid.exp())
        # calculate logits
        self.logits =-np.log(self.quad_points)
        
        
        super(poissonLogNormal, self).__init__(batch_shape, validate_args=validate_args)
    def quadrature(self):
        # calculate quad points
        edges = torch.linspace(0.,1.,steps=self.quad_points+3).to(self.device)[1:-1] # dont use 0,1 as this may give inf
        logNormal = LogNormal(self.mu,self.sigma)
        # use icdf to get quad points
        quantiles =logNormal.icdf(edges)
        # take midpoints
        grid = (quantiles[:,1:]+quantiles[:,:-1])/2
        return grid
    def log_prob(self,x):
        # return log prob
        if self._validate_args:
            self._validate_sample(x)
        x, = broadcast_all(x)
        return torch.logsumexp(self.logits+self.poisson.log_prob(x[:,None,None]),dim=-1)
    def cdf(self,x):
        if self._validate_args:
            self._validate_sample(x)
        x, = broadcast_all(x)
        # cast to integers
        n = x.floor().long()
        # create vector of all points up to maximum
        r = torch.linspace(0,n.max(),n.max()+1).to(self.device)
        # calculate cdf from log prob
        res = self.log_prob(r).exp().t().cumsum(-1)
        res[res.ne(res)]=0.0 # replace nans with 0
        return res[:,n].clamp(0.0,1.0) 
    
class bivariatePoissonLogNormal(ExponentialFamily):  
    """
    bivariate poisson log-normal distribution class 
    (not working)
    """  
    def __init__(self,mu,sigma,rho,quad_points=10,validate_args=None):
        self.mu,self.sigma = broadcast_all(mu,sigma)
        self.rho,= broadcast_all(rho)
        self.quad_points= quad_points
        if isinstance(mu, Number):
            batch_shape = torch.Size()
            self.device = torch.device("cpu")
            self.batch_size = torch.tensor([1])
        else:
            batch_shape = self.mu.size()
            self.device = self.mu.device
            self.batch_size = batch_shape[0]
        c = self.sigma.sqrt().prod(dim=-1)[:,None]*self.rho
        
        self.covmatrix = torch.cat((self.sigma[:,0,None],c,c,self.sigma[:,1,None]),dim=-1).view(self.batch_size,2,2)
        self.sigmaM = self.covmatrix.cholesky()
        self.quadrature()
        
        self.poissonx = Poisson(self.lam1)
        self.poissony = Poisson(self.lam2)
        
        
        super(bivariatePoissonLogNormal, self).__init__(batch_shape, validate_args=validate_args)
    def quadrature(self):
        p,w = np.polynomial.hermite.hermgauss(self.quad_points)
        #p,w = torch.from_numpy(p).type(torch.float32).to(self.device),torch.from_numpy(w).type(torch.float32).to(self.device)
        xn = torch.tensor([i for i in itertools.product(p,repeat=2)]).type(torch.float32).to(self.device)
        self.wn =torch.tensor([i for i in itertools.product(w,repeat=2)]).type(torch.float32).to(self.device).prod(-1).log()
        r = (torch.matmul(self.sigmaM,xn.t()).transpose(1,2)*np.sqrt(2) + self.mu.unsqueeze(1).repeat(1,100,1)).exp() 
        lr = torch.matmul(self.sigmaM,(1./r).transpose(1,2)).transpose(1,2)
        self.lam1,self.lam2 = r[...,0], r[...,1]
        self.llam1,self.llam2 = lr[...,0], lr[...,1]
        self.const = self.sigmaM.det()/(np.pi)
        
        return 
    def prob(self,x,y):
        if self._validate_args:
            self._validate_sample(x)
            self._validate_sample(y)
        x,y = broadcast_all(x,y)
        
        return (self.const*(self.wn+(self.llam1*self.llam2).log()+self.poissonx.log_prob(x.unsqueeze(-1).unsqueeze(-1))+self.poissony.log_prob(y.unsqueeze(-1).unsqueeze(-1))-(self.lam1*self.lam2).log()).exp().sum(-1)).reshape(self.batch_size,*x.shape)
    def cdf(self,x,y):
        if self._validate_args:
            self._validate_sample(x)
        x,y = broadcast_all(x,y)
        nx = x.floor().long()
        ny = y.floor().long()
        rx = torch.linspace(0,nx.max(),nx.max()+1).to(self.device)
        ry = torch.linspace(0,ny.max(),ny.max()+1).to(self.device)
        rx,ry=torch.meshgrid([rx,ry])
        rx,ry=rx.flatten(),ry.flatten()
        
        res = torch.zeros((self.batch_size,nx.max()+2,ny.max()+2)).to(self.device)
        res[:,1:,1:] = self.prob(rx,ry).view(self.batch_size,nx.max()+1,ny.max()+1).cumsum(-1).cumsum(-2)
        return res 


class parameter_head(nn.Module):
    def __init__(self,out_act,base_size,out_size):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda" if self.use_cuda else "cpu")

        self.l1 = nn.Linear(base_size,32).to(self.device,non_blocking=self.use_cuda)

        self.l2 = nn.Linear(32,out_size).to(self.device,non_blocking=self.use_cuda)

        self.act = nn.ELU()
        self.outact = out_act
    def forward(self,x):
        x=self.act(self.l1(x))
        return self.outact(self.l2(x))



class MixD(model):
    """
    Mixture Model
    """

    def __init__(self,T,num_mix,input_dim,bins,dense_act=nn.ELU(),mix_dist="Gaussian",p_act=(Exponential(),Exponential(),Exponential(),Exponential()),learning_rate=1e-4,batch_size=1024):#1024
        super(model,self).__init__() # inherit
        # set constants
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda" if self.use_cuda else "cpu")
        self.batch_size = batch_size
        self.T = T
        self.input_dim = input_dim
        self.p0_act = p_act[0]
        self.p1_act = p_act[1]
        self.mix_dist = mix_dist
        self.bivariate = False
        self.num_mix = num_mix
        
        # create layers
        self.p0 = parameter_head(self.p0_act,64,self.num_mix).to(self.device,non_blocking=self.use_cuda)
        self.p1 = parameter_head(self.p1_act,64,self.num_mix).to(self.device,non_blocking=self.use_cuda)

        self.e_learning_rate = learning_rate
        self.embedding1 = nn.Linear(self.input_dim,64).to(self.device,non_blocking=self.use_cuda)
        self.embedding1Act = dense_act
        self.embedding2 = nn.Linear(64,128).to(self.device,non_blocking=self.use_cuda)
        self.embedding2Act = dense_act
        self.embedding3 = nn.Linear(128,256).to(self.device,non_blocking=self.use_cuda)
        self.embedding3Act = dense_act
        self.embedding4 = nn.Linear(256,128).to(self.device,non_blocking=self.use_cuda)
        self.embedding4Act = dense_act
        self.embedding5 = nn.Linear(128,64).to(self.device,non_blocking=self.use_cuda)
        self.embedding5Act = dense_act
        
        self.was_p = 2
        
        self.aiAct = nn.Softmax(dim=1)
        self.ai = parameter_head(self.aiAct,64,self.num_mix).to(self.device,non_blocking=self.use_cuda)
        
        self.mix = torch.arange(0,self.num_mix,dtype=torch.int).to(self.device,non_blocking=self.use_cuda)
        self.mix_dist = mix_dist
        # set which function to use for getting predictions and set bins
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
            # create additional layers required for this distribution
            self.p2_act = p_act[2]
            self.p3_act = p_act[3]
            self.p2 = parameter_head(self.aiAct,64,self.num_mix).to(self.device,non_blocking=self.use_cuda)
            self.p3 = parameter_head(self.aiAct,64,self.num_mix).to(self.device,non_blocking=self.use_cuda)
            
            self.bivariate = True
        else:
            raise ValueError("Invalid mixture distribution")
        if mix_dist != "bivariateGammaPoisson":
            self.mbins = (bins[1:] - bins[:-1]).to(self.device,non_blocking=self.use_cuda)

    def forward(self, x):
        # forward function
        # initial dense layers
        x = self.embedding1Act(self.embedding1(x))
        x = self.embedding2Act(self.embedding2(x))
        x = self.embedding3Act(self.embedding3(x))
        x = self.embedding4Act(self.embedding4(x))
        x = self.embedding5Act(self.embedding5(x))
        # get predictions of parameters
        if self.bivariate:
            p0 = self.p0(x)
            p1 = self.p1(x)
            p2 = self.p2(x)
            p3 = self.p3(x)
            p = (p0,p1,p2,p3)
        else:
            p0 = self.p0(x)
            p1 = self.p1(x)
            p = (p0,p1)
        # get weights
        pr = self.ai(x)
        
        return p,pr    
    def normalPredict(self,x):
        
        (m,s),p = x
        y = 0.0
        for i in self.mix: # for each dist in mixture evaluate cdf at bins
            dist = Normal(m[:,i,None],s[:,i,None])
            t = dist.cdf(self.bins)
            y += (t[:,1:]-t[:,:-1]).clamp(0)*p[:,i,None]
        return y
    def poissonLogNormalPredict(self,x):
        
        (m,s),p = x
        y = 0.0
        for i in self.mix:# for each dist in mixture evaluate cdf at bins
            dist = poissonLogNormal(m[:,i,None],s[:,i,None])
            t = dist.cdf(self.bins)
            y += (t[:,1:]-t[:,:-1]).clamp(0)*p[:,i,None]
        return y
    def gammaPoissonPredict(self,x):
        
        (alpha,beta),p = x
        y = 0.0
        for i in self.mix:# for each dist in mixture evaluate cdf at bins
            dist = gammaPoisson(alpha[:,i,None],beta[:,i,None])
            t = dist.cdf(self.bins)
            y += (t[:,1:]-t[:,:-1]).clamp(0)*p[:,i,None]
        return y
    def biGammaPoissonPredict(self,x):
        
        (alpha,beta,theta0,theta1),p = x
        y = 0.0
        for i in self.mix:# for each dist in mixture evaluate cdf at bins
            dist = bivariateGammaPoisson(alpha[:,i,None],beta[:,i,None],theta0[:,i,None],theta1[:,i,None])
            y += dist._cdf(self.binsx,self.binsy).clamp(0)*p[:,i,None,None]
        return y
    
    @torch.jit.export
    def train_model(self,epochs=100,optim="Adam",data="Gene",loss_fn="H2",custom_data=None,model_name=None):
        self._directory_maker() # create directory
        # set consts
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.optim = optim
        
        if custom_data is None:
            # load data and split into train and test data
            xall,yall=self.load_data(data=data)
            x_train, x_test,y_train,y_test = train_test_split(xall,yall,train_size=0.8)
        else:
            # if custom data is given use that
            x_train, x_test,y_train,y_test = custom_data["x_train"],custom_data["x_test"],custom_data["y_train"],custom_data["y_test"]
            self.scalesM,self.scalesS = custom_data["scalesM"],custom_data["scalesS"]
        # set name of model
        if model_name == None:
            self.model_name = f"mix_{self.mix_dist}_{loss_fn}_{optim}_{self.num_mix}"
        else:
            self.model_name=model_name
        # create tensorboard writer
        writer = SummaryWriter(comment=self.model_name)
        # create optimiser
        self._init_optim()
        print("Loading dataset")
        # cast from numpy to tensor
        xtrainTensor=torch.from_numpy(x_train).type(torch.float32)
        ytrainTensor=torch.from_numpy(y_train).type(torch.float32)

        if self.mix_dist=="bivariateGammaPoisson":
            # use no additional workers for loading in the bivariate case
            # to stop the pytorch data loader loading too many batches and causing
            # a memory overflow
            num_workers = 0
        else:
            # for the univariate distribution use extra loaders as we dont
            # get a memory overflow
            num_workers = 2


        # create datat loaders
        dataset = torch.utils.data.TensorDataset(xtrainTensor,ytrainTensor)
        loader = DataLoader(dataset, batch_size=self.batch_size,drop_last=True,shuffle=True,
                    pin_memory=self.use_cuda,num_workers=num_workers)#self.use_cuda
        xtestTensor=torch.from_numpy(x_test).type(torch.float32)
        ytestTensor=torch.from_numpy(y_test).type(torch.float32)
        valDataset = torch.utils.data.TensorDataset(xtestTensor,ytestTensor)
        valLoader = DataLoader(valDataset, batch_size=self.batch_size,drop_last=True,shuffle=False,
                    pin_memory=self.use_cuda,num_workers=num_workers)#self.use_cuda
        old_test_loss = np.inf
        train_batches = x_train.shape[0] // self.batch_size
        test_batches =x_test.shape[0] // self.batch_size
        # intialise loss arrays
        self._init_loss()
        del valDataset,dataset,xtestTensor,ytestTensor,x_train,x_test,y_train,y_test
        # start train loop
        print("Train Loop Starting...")
        for i in range(self.epochs):
            #nvtx.range_push("Epoch " + str(i+1))
            self.train() # put in train mode
            print(f"STEP: {i+1}/{self.epochs}")
            with tqdm(total=train_batches,ascii=True,desc="Training") as pbar:#loading bar
                for idx, (local_batch, local_labels) in enumerate(loader):
                    #nvtx.range_push("Batch "+str(idx))
                    
                    #nvtx.range_push("Copying to device")
                    # get batchees
                    y=local_labels.to(self.device,non_blocking=self.use_cuda)
                    x_batch=local_batch.to(self.device,non_blocking=self.use_cuda)
                    #nvtx.range_pop()
                    #nvtx.range_push("Forward pass")
                    # forward pass
                    x=self(x_batch)
                    #nvtx.range_pop()
                    
                    #nvtx.range_push("Backward pass")
                    # get output of mixture for bins and update loss and back prop
                    self._update_loss_train(self.distPredict(x),y,i)
                    
                    
                    #nvtx.range_pop()
                    #nvtx.range_pop()
                    
                    
                    
                    pbar.update(1)
                
            #nvtx.range_pop()
            # average
            self.train_loss[i] = self.train_loss[i]/train_batches
            if np.isnan(self.train_loss[i,0]):# break if we get nan
                print("GOT NaN")
                break
            # update tensorboard log and print loss
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/train'
                writer.add_scalar(name, float(self.train_loss[i,l]), i)
                print(f"{name}: {self.train_loss[i,l]}")
            # get val loss
            with torch.no_grad(): # dont keep gradients
                self.eval()# set to eval
                with tqdm(total=test_batches,ascii=True,desc="Validating") as pbar:
                    for local_batch, local_labels in valLoader:
                        # get batch
                        y = local_labels.to(self.device,non_blocking=self.use_cuda)
                        x_batch=local_batch.to(self.device,non_blocking=self.use_cuda)
                        # forward pass
                        pred = self(x_batch)
                        # update val loss
                        self._update_loss_val(self.distPredict(pred),y,i)
                        pbar.update(1)
            #nvtx.range_pop()
            # average
            self.val_loss[i] = self.val_loss[i]/test_batches
            # update tensorboard print val loss
            for l,s in enumerate(self.loss_names):
                name=f'Loss {s}/test'
                writer.add_scalar(name, float(self.val_loss[i,l]), i)
                print(f"{name}: {self.val_loss[i,l]}")
            #nvtx.range_pop()
            # if achieves lowest val loss save
            if self.val_loss[i,0] < old_test_loss:
                #nvtx.mark("Saving best model")
                torch.save({"state_dict":self.state_dict(),
                            "scalesM":self.scalesM,
                            "scalesS":self.scalesS,
                            "epoch":i+1,
                            "optim":self.optimiser,
                            "val_loss":self.val_loss,
                            "T": self.T,
                            "input_dim": self.input_dim}, f"./best/mdBestModel_{self.model_name}.pth")
                old_test_loss = self.val_loss[i,0]
        # save final model
        torch.save({"state_dict":self.state_dict(),
                        "scalesM":self.scalesM,
                        "scalesS":self.scalesS,
                        "epoch":self.epochs,
                        "optim":self.optimiser,
                        "val_loss":self.val_loss,
                        "T": self.T,
                        "input_dim": self.input_dim}, f"./final/mdFinalModel_{self.model_name}.pth")
        return self.train_loss,self.val_loss
    @torch.jit.export
    def predict(self,x,raw=True):
        # get predictions
        self.eval() # set to eval 
        # scale x
        x = (np.atleast_2d(x) - self.scalesM)/self.scalesS
        p = int(np.ceil(x.shape[0] /self.batch_size))
        t = x.shape[0]
        r = self.batch_size - (t % self.batch_size)
        x = np.pad(x,((0,r),(0,0)))
        xs = np.vsplit(x,p)
        res = [None] * p
        if raw: # return parameters
            for i in range(p):# feed x in batch size chunks
                # forward
                a,p = self(torch.from_numpy(xs[i]).type(torch.float32).to(self.device,non_blocking=self.use_cuda))
                # detach and send to numpy
                p = p.detach().cpu().numpy()
                a = torch.cat(a,dim=-1).detach().cpu().numpy()
                res[i] = np.hstack((p,a))

        else:# return bin probs
            for i in range(p):# feed x in batch size chunks
                # forward
                x = self(torch.from_numpy(xs[i]).type(torch.float32).to(self.device,non_blocking=self.use_cuda))
                # get dist predictions
                x = self.distPredict(x)
                
                res[i] = x.detach().cpu().numpy()
        res[-1] = res[-1][:-r]
        
        return np.vstack(tuple(res))
    @torch.jit.export
    def load_model(self,PATH):
        # load model
        model = torch.load(PATH,map_location=self.device)
        self.scalesM = model["scalesM"]
        self.scalesS = model["scalesS"]

        self.load_state_dict(model["state_dict"])
            
            
            
            
        

if __name__=="__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_mix = [2**i for i in range(7)]
    loss = ["SINK","L1","H2","MDDEDIV","MSE"]
    optim = ["Adam","SGD"]
    gau = torch.linspace(-15,15,100+1)
    real = torch.linspace(0,300,100+1)
    g = ParameterGrid({"loss":loss})

    val_loss = [[None,i] for i in g] 
    train_loss = [[None,i] for i in g] 
    for v,i in enumerate(g):
        try:
            print(i)
            m = MixD(T=100,p_act=(nn.Identity(),Exponential()),num_mix=3,input_dim=9,bins=gau,mix_dist="Gaussian").to(device)
            m.was_p = 2
            train_loss[v][0],val_loss[v][0] = m.train_model(epochs=5000,data="Gaussian",loss_fn=i["loss"],optim="Adam",model_name=i["loss"])

        except Exception as inst:
            print("ERROR!!!!")
            print(type(inst))
            print(inst) 
    with open('train_loss.pickle', 'wb') as f:
        pickle.dump(train_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('val_loss.pickle', 'wb') as f:
        pickle.dump(val_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
