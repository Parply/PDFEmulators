import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# numpy version of loss functions
h2 = lambda x,y: 2*(np.sqrt(y)-np.sqrt(x))**2
e_div = lambda x,y: (x)*(np.log(x)-np.log(y))**2
mse = lambda x,y: (y-x)**2
l1 = lambda x,y: np.abs(y-x)
kld = lambda x,y: (x)*(np.log(x)-np.log(y))
mkld = lambda x,y: np.maximum(kld(x,y),kld(y,x))
mediv = lambda x,y: np.maximum(e_div(x,y),e_div(y,x))

    








def plotter3d(x,y,funcs,labels):
    """
    Makes 3d plots of the loss functions
    """
    if type(funcs)!=list: funcs=[funcs]
    if type(labels)!=list: labels=[labels]
    assert len(funcs)==len(labels), "len(funcs)!=len(labels)"
    z = [np.minimum(i(x,y),5) for i in funcs]# dont plot values larger than 5
    # sub plot size
    plots = int(np.ceil(np.sqrt(len(funcs))))
    fig = plt.figure(figsize=(15,15))
    bottom, top = 0.03, 0.97
    left, right = 0.03, 0.93
    fig.tight_layout()
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.05, wspace=0.05)

    for i,k in enumerate(z): # plot each loss
        ax = fig.add_subplot(plots, plots, i+1, projection='3d')
        # plot surface
        surf = ax.plot_surface(x,y,k,cmap=cm.viridis,linewidth=0,antialiased=True,vmin=0,vmax=5)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Loss")
        ax.set_zlim(0,5)
        
        ax.set_title(labels[i])
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.view_init(elev=10, azim=45) # rotate
    # add an axis for colour bar
    cax=fig.add_axes([0.95, bottom, 0.03, top-bottom])
    # plot colour bar
    fig.colorbar(surf,cax=cax)
    
    
    return fig

if __name__ =="__main__":
    x = np.maximum(np.linspace(0,1,1001),1e-30)
    y = np.maximum(np.linspace(0,1,1001),1e-30)

    x,y=np.meshgrid(x,y) # points to plot for
    funcs = [h2,mse,e_div,l1,kld,mkld,mediv] # functions to plot
    # names of functions
    titles = ["Squared Hellinger Distance","Mean Squared Error","Exponential Divergence","L1","Kullback-Leibler Divergence","Maximum Kullback-Leibler Divergence","Maximum Exponential Divergence"]
    # plot
    fig = plotter3d(x,y,funcs,titles)
    # save
    fig.savefig("allLoss.png")
    
    