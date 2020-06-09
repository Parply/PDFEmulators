from pytorchModel import *

if __name__=="__main__":
    n=5
    xall,yall,scalesM,scalesS = dataLoader(T=100,data="BiGene")# load data
    x_train, x_test,y_train,y_test=train_test_split(xall,yall,train_size=0.8) # split into train and test
    with h5py.File("ensemble_data_bigene.hdf5","w") as f:
        f.create_dataset("x_train",data=x_train)
        f.create_dataset("y_train",data=y_train)
        f.create_dataset("x_test",data=x_test)
        f.create_dataset("y_test",data=y_test)
        f.create_dataset("scalesS",data=scalesS)
        f.create_dataset("scalesM",data=scalesM)
    t = x_train.shape[0]
    for i in range(n):# for n sub models take samples of the train data and save
        ind = np.random.randint(t,size=t)
        x = x_train[ind]
        y = y_train[ind]
        m = np.mean(x,axis=0)
        s = np.std(x,axis=0)
        with h5py.File("ensemble_data_bigene.hdf5","a") as f:
            f.create_dataset(f"x_train_{i+1}",data=scale(x))
            f.create_dataset(f"y_train_{i+1}",data=y)
            f.create_dataset(f"scalesM_{i+1}",data=m)
            f.create_dataset(f"scalesS_{i+1}",data=s)