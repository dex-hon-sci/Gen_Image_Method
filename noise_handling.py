#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:59:22 2023

@author: dexter
"""

#experiment stage

# Zeropoint --> flatting -> stack -->spark/ trail line/ abnormaly detection -> deconvolution -> noise removal -> Post processing enhancement
# flatting

# deconvolution

#resize image to resolve edge pprobken



def deconvolve(img, psf, num_iter = 200):
    deconv_img = restoration.richardson_lucy(img, psf, num_iter=num_iter)

    return deconv_img

# input image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
import random
from skimage import color, data, restoration

 
def gaussian_filter(kernel_size, sigma=1, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

    return gauss

def sharpen_filter(centre,surround = -1):
    
    sharpen_filter=np.array([[surround,surround,surround],
                             [surround,centre,surround],
                             [surround,surround,surround]])
    
    return sharpen_filter

def gen_noise_poisson(lam, dim):
    
    rng = np.random.default_rng()

    noise_frame = (rng.poisson(lam=lam, size=dim) - 10) / 255.

    return noise_frame

rng = np.random.default_rng()


astro_og = color.rgb2gray(data.immunohistochemistry())
#astro_og = data.microaneurysms()

#psf = np.random.multivariate_normal(mean, cov, (3,3))
psf = gaussian_filter(10,sigma=5,muu=0)

astro = conv2(astro_og, psf, 'same')
# Add Noise to Image

noise_frame = (rng.poisson(lam=25, size=astro.shape) - 10) / 255.

astro_noisy = astro.copy()
astro_noisy += noise_frame #(rng.poisson(lam=25, size=astro.shape) - 10) / 255.

astro_noisy1 =  astro.copy() + (rng.poisson(lam=25, size=astro.shape) - 10) / 255.
astro_noisy2 =  astro.copy() + (rng.poisson(lam=25, size=astro.shape) - 10) / 255.
astro_noisy3 =  astro.copy() + (rng.poisson(lam=25, size=astro.shape) - 10) / 255.
astro_noisy4 =  astro.copy() + (rng.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro, psf, num_iter=500)

#deconvolved_RL_denoise = deconvolved_RL - noise_frame


#fig = plt.subplot()

#plt.imshow(psf)
#plt.show()

#fig = plt.subplot()

#plt.imshow(rng.poisson(lam=25, size=astro.shape))
#plt.show()


def plot_three_frame(original, noisy, restore):
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
           a.axis('off')
           
    ax[0].imshow(original, vmin=original.min(), vmax=original.max())
    ax[0].set_title('Original Data (RAW+PSF)')

    ax[1].imshow(noisy, vmin=noisy.min(), vmax=noisy.max())
    ax[1].set_title('Noisy data\n(RAW+PSF+Noise)')

    ax[2].imshow(restore, vmin=restore.min(), vmax=restore.max())# vmin=astro_noisy.min(), vmax=astro_noisy.max())
    ax[2].set_title('Restoration using\nRichardson-Lucy')
    plt.show()

    return fig 

#plot_three_frame(astro,astro_noisy, deconvolved_RL_denoise)


#fig = plt.subplot()
#plt.imshow(astro_og-deconvolved_RL)
#plt.show()

#plot_three_frame(astro_og, deconvolved_RL, astro_og-deconvolved_RL)


from scipy.fft import ifftn
import matplotlib.cm as cm

lily = color.rgb2gray(data.hubble_deep_field())
#lily = data.hubble_deep_field()

#noise_frequency = ifftn(lily)

#fig = plt.subplot()
#plt.imshow(np.real(noise_frequency), norm='linear', #cmap=cm.Reds,
#           vmin=np.real(noise_frequency).min(), 
#           vmax=np.real(noise_frequency).max())
#plt.show()

import cv2 

# create a sharpening kernel
sharpen_filter = sharpen_filter(9,surround = -1)
# applying kernels to the input image to get the sharpened image
sharp_image=cv2.filter2D(lily,-1,sharpen_filter)

#fig = plt.subplot()
#plt.imshow(lily, vmin=astro_og.min(), vmax=astro_og.max())
#plt.show()

lily = conv2(lily, psf, 'same')
lily_deconv = deconvolve(lily, psf)


plot_three_frame(astro,astro_noisy, deconvolved_RL)

plot_three_frame(lily, psf, lily_deconv)