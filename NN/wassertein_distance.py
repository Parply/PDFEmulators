import torch 
import torch.nn as nn

def pwasserstein(x,y,x_weights=None,y_weights=None,p=1):
    B,S = x.shape
    x_sorter,y_sorter=torch.argsort(x,dim=-1),torch.argsort(y,dim=-1)
    all_values = torch.cat((x,y),dim=-1)
    all_values,ind = torch.sort(all_values,dim=-1)
    deltas = all_values[:,1:] - all_values[:,:-1]
    x_cdf_ind = torch.searchsorted(x.gather(1,x_sorter),all_values[:,:-1],right=True)
    y_cdf_ind = torch.searchsorted(y.gather(1,y_sorter),all_values[:,:-1],right=True)
    
    if x_weights is None:
        x_cdf = torch.true_divide(x_cdf_ind, S)
    else:
        s1,s2=x_weights.shape
        x_sorted_cumweights = torch.zeros((s1,s2+1))
        x_sorted_cumweights[:,1:] = x_weights.gather(1,x_sorter).cumsum(dim=-1)
        x_cdf = x_sorted_cumweights.gather(1,x_cdf_ind)/x_sorted_cumweights[:,-1,None]
    

    if y_weights is None:
        y_cdf = torch.true_divide(y_cdf_ind, S)
    else:
        s1,s2=y_weights.shape
        y_sorted_cumweights = torch.zeros((s1,s2+1))
        y_sorted_cumweights[:,1:] = y_weights.gather(1,y_sorter).cumsum(dim=-1)
        y_cdf = y_sorted_cumweights.gather(1,y_cdf_ind)/y_sorted_cumweights[:,-1,None]

    return ((x_cdf-y_cdf).abs().pow(p)*deltas).sum(dim=-1).pow(1/p)



class SinkhornDistance(nn.Module):
    def __init__(self,eps,max_iter,threshold,reduction='None',p=2):
        super(SinkhornDistance,self).__init__()
        self.eps=eps
        self.max_iter=max_iter
        self.reduction=reduction
        self.p=p
        self.thershold = threshold

    def forward(self,x,y,x_support,y_support):
        device = x.device
        data_type = x.dtype
        if x.dim() == 1:
            batch_size=1
        else:
            batch_size = x.shape[0]
        if x_support.dim() ==1:
            s1 = x_support.shape[0]
            x_support=x_support.repeat(batch_size).reshape(batch_size,s1).type(data_type).to(device)
        if y_support.dim() ==1:
            s1 = y_support.shape[0]
            y_support=y_support.repeat(batch_size).reshape(batch_size,s1).type(data_type).to(device)

        x = torch.cat((x_support.unsqueeze(-1),x.unsqueeze(-1)),dim=-1).to(device)

        y = torch.cat((y_support.unsqueeze(-1),y.unsqueeze(-1)),dim=-1).to(device)


        


        cost = self._cost_matrix(x,y,self.p)
        x_points,y_points = x.shape[-2],y.shape[-2]
        batch_size = x.shape[0]

        mu = torch.empty(batch_size,x_points,dtype=data_type).fill_(1./x_points).to(device)

        nu = torch.empty(batch_size,y_points,dtype=data_type).fill_(1./y_points).to(device)


        u,v=torch.zeros_like(mu,dtype=data_type).to(device),torch.zeros_like(nu,dtype=data_type).to(device)

        it = 0
        err = self.thershold+1
        while (it<self.max_iter) and (err<self.thershold):
            u1=u
            u = self.eps * (mu.clamp(1e-12).log()-self.M(cost,u,v).logsumexp(dim=-1)) + u
            v = self.eps * (nu.clamp(1e-12).log()-self.M(cost,u,v).transpose(-2,-1).logsumexp(dim=-1)) + v

            err = (u-u1).abs().sum(-1).mean()

            it +=1

        p1 = self.M(cost,u,v).exp()
        cost = (p1*cost).sum(dim=(-2,-1))
        
        if self.reduction == 'mean':
            cost=cost.mean()
        elif self.reduction == 'sum':
            cost=cost.sum()

        return cost

    @staticmethod
    def _cost_matrix(x,y,p=2):
        return (x.unsqueeze(-2)-y.unsqueeze(-3)).abs().pow(p).sum(-1)
    
    def M(self,cost,u,v):
        return (u.unsqueeze(-1)+v.unsqueeze(-2)-cost)/self.eps









