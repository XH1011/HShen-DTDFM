import matplotlib.pyplot as plt
import tensorflow as tf
from Utils.network_utils import GaussianDiffusion
from pylab import xticks
import numpy as np
def Add_NoiseTo_image(x_eval,noise_eval,t):
    gdf = GaussianDiffusion(beta_start=1e-4, beta_end=0.02,timesteps=1000+1)

    # gdf = GaussianDiffusion(beta_start=1e-6, beta_end=2e-4, timesteps=1000+1)
    noise_image = gdf.q_sample(x_eval, t, noise_eval)
    noise_image=(tf.reshape(noise_image, shape=[1024, ])).numpy()
    return noise_image

def show_Added_NoiseImage(x,noise):
    fig = plt.figure(figsize=(10,10))
    for index, t in enumerate([0, 5, 100, 999]): #这四个数的长度决定于变量timesteps的大小
        if index<1:
            noisy_im=(tf.reshape(x, shape=[1024, ])).numpy()#没有添加噪声
            print('List index:',index,'Without any noises')
        else:
            print('List index:',index,'The added noise is:', t)
            noisy_im=Add_NoiseTo_image(x,noise,t)#依次添加噪声
        plt.subplot(4, 1, index + 1)
        plt.plot(noisy_im.T, 'b')
        plt.xlim((0, 1024))
        xticks(np.linspace(0, 1024, 5, endpoint=True))
        plt.ylabel('Amplitude')
    plt.xlabel('Data points')




