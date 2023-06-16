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
 
def deconvolve():
    
    return

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

rng = np.random.default_rng()


astro_og = color.rgb2gray(data.astronaut())

#psf = np.ones((5, 5)) / 25
mean = [5,5]
cov = [[1, 0], [0, 100]]
#psf = np.random.multivariate_normal(mean, cov, (3,3))
psf = gaussian_filter(10,sigma=7.0,muu=5.0)

print(psf)

astro = conv2(astro_og, psf, 'same')
# Add Noise to Image

noise_frame = (rng.poisson(lam=25, size=astro.shape) - 10) / 255.

astro_noisy = astro.copy()
astro_noisy += noise_frame #(rng.poisson(lam=25, size=astro.shape) - 10) / 255.

# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)

deconvolved_RL_denoise = deconvolved_RL - noise_frame


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
           
    ax[0].imshow(original,vmin=original.min(), vmax=original.max())
    ax[0].set_title('Original Data (RAW+PSF)')

    ax[1].imshow(noisy, vmin=original.min(), vmax=original.max())
    ax[1].set_title('Noisy data\n(RAW+PSF+Noise)')

    ax[2].imshow(restore, vmin=original.min(), vmax=original.max() )# vmin=astro_noisy.min(), vmax=astro_noisy.max())
    ax[2].set_title('Restoration using\nRichardson-Lucy')

    return fig 

plot_three_frame(astro,astro_noisy, deconvolved_RL_denoise)

plot_three_frame(astro,astro_noisy, deconvolved_RL)

#fig = plt.subplot()
#plt.imshow(astro_og-deconvolved_RL)
#plt.show()
plot_three_frame(astro_og, deconvolved_RL, astro_og-deconvolved_RL)


from scipy.fft import ifftn

import cv2 

# create a sharpening kernel
sharpen_filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])
# applying kernels to the input image to get the sharpened image
sharp_image=cv2.filter2D(astro_og,-1,sharpen_filter)

fig = plt.subplot()
plt.imshow(sharp_image, vmin=astro_og.min(), vmax=astro_og.max())
plt.show()

plot_three_frame(astro_og, sharp_image, astro_noisy)