import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
font = {'style' : 'normal',
#        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

def sample_class_0(N, noise):
    centre0 = np.array([-1, 0])
    centre1 = np.array([1, 0])
    bias = 0.5
    covmtx = noise*np.identity(2)
    class_ids = np.random.choice([0,1],N, p=[bias,1-bias])
    num_class_1 = np.sum(class_ids)
    num_class_0 = N-num_class_1
    data = np.empty((N,2))
    data[class_ids==1,:] = np.random.multivariate_normal(
        centre1, covmtx, num_class_1)
    data[class_ids==0,:] = np.random.multivariate_normal(
        centre0, covmtx, num_class_0)
    return data

def sample_class_1(N, noise):
    thetas = np.pi * np.random.beta(2,2,N) +np.pi/2.
    print("thetas.shape = %r" % (thetas.shape,) )
    print("np.random.normal(5) = %r" % (np.random.normal(5),) )
    rs = np.random.normal(loc=1,scale=noise, size=N)
    print("rs.shape = %r" % (rs.shape,) )
    are_left = thetas <= np.pi
    are_right =  thetas > np.pi
    
    xys = np.empty((N,2))
    xys[are_left,0] = -rs[are_left] *np.cos(thetas[are_left])-1
    xys[are_right,0] = rs[are_right] *np.cos(thetas[are_right])+1
    xys[:,1] = rs *np.sin(thetas)
    return xys

def sample_yin_yang(N, bias=0.5, noise=0.3):
    class_ids = np.random.choice([0,1],N, p=[bias,1-bias])
    num_class_1 = np.sum(class_ids)
    num_class_0 = N-num_class_1
    data = np.empty((N,2))
    data[class_ids==1,:] = sample_class_1(num_class_1, noise)
    data[class_ids==0,:] = sample_class_0(num_class_0, noise)
    return data, class_ids



