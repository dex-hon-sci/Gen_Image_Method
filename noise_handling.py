#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:59:22 2023

@author: dexter

This script contains useful functions for noise handling.
"""

#experiment stage

# Zeropoint --> flatting -> stack -->spark/ trail line/ abnormaly detection -> deconvolution -> noise removal -> Post processing enhancement
# flatting

#resize image to resolve edge pprobken
from matplotlib.patches import Ellipse

import cv2 
import matplotlib.cm as cm
from numpy.fft import fftn, ifftn, fftshift 
import sep # source extraction package
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
import random
from skimage import color, data, restoration

#%%
# Generate noise
def gaussian_filter(kernel_size, sigma=1, muu=0):
    """
    Construction of a 2D gaussian squared grid.

    Parameters
    ----------
    kernel_size : float
        The length of the grid.
    sigma : float, optional
        The standard deviation of the Gaussian distribution. The default is 1.
    muu : float, optional
        The mean of the Gaussian. The default is 0.

    Returns
    -------
    gauss : 2D ndarray
        The frame with the Gaussian distribution (centred at zero).

    """
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1.0 /(2.0 * np.pi * sigma**2)

    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

    return gauss

def sharpen_filter(centre,surround = -1):
    """
    Construct a 3x3 sharpening kernel 

    Parameters
    ----------
    centre : float
        The centre value of the kernel (The degree of enhancement).
    surround : float, optional
        The centre value of the kernel (The degree of devaluation).
        The default is -1.

    Returns
    -------
    sharpen_filter : 2D ndarray
        The sharpening kernel.

    """
    sharpen_filter=np.array([[surround,surround,surround],
                             [surround,centre,surround],
                             [surround,surround,surround]])
    
    return sharpen_filter

def gen_noise_poisson(lam, dim, norm = 255):
    """
    Generate a 2D frame of poisson noise.

    Parameters
    ----------
    lam : float
        Lambda in the Poisson distribution, the expected value.
    dim : tuple
        The dimension of the frame.
        e.g. (5,5): 5x5 grid.
    norm : float
        Normalisation value. Default is 255.

    Returns
    -------
    noise_frame : 2D ndarray
        The noise frame.

    """
    
    rng = np.random.default_rng()
    
    # Possion noise normalised by norm=255 pixel value
    noise_frame = rng.poisson(lam=lam, size=dim) / 255.

    return noise_frame

def add_spark(img, spark_num=3, connect_pix= 30, val = 200):
    """
    Add random sparks in the image

    Parameters
    ----------
    img : 2D ndarray
        The original image.
    spark_num : int, optional
        Total number of sparks. The default is 3.
    connect_pix : int, optional
        The number of pixel the sparks occupies. The default is 30.
    val : float, optional
        The value to be replaced in the image. The default is 200.

    Returns
    -------
    img : 2D ndarray
        The new image.

    """
    
    # Initialise a blank frame
    origin = np.zeros((1000,1000))
    
    for i in range(spark_num):
        # Choose a random coordinate
        x0,y0 = np.random.choice(a=img.shape[0],size=1), np.random.choice(a=img.shape[0],size=1)
        
        # Define random wallk move set
        move_set = [-1,0,1]

        print('x0,y0', x0, y0)
        x,y = x0[0],y0[0]
        for j in range(connect_pix):
            # generate coordinate
            x, y = x + np.random.choice(move_set), y + np.random.choice(move_set)
#            print('x,y',x,y,type(x),type(y))
            #if x < img.shape[0] or y < img.shape[0] or x > 0 or y > 0:    
                # add bright trail
            origin[x][y] = val

            
    #img += origin
    origin2 = origin[0:img.shape[0], 0:img.shape[0]]
    print('origin2',origin2.shape)
    return origin2
    

#%%
def source_detect(img):
    """
    Source detection

    Parameters
    ----------
    img : 2D ndarray
        Original image.

    Returns
    -------
    bright_obj : list
        A list of bright objects.

    """
    # source detection and segmentation
    bright_obj = sep.extract(img, 1.5) #, err=bkg.globalrms)

    return bright_obj

def bg_estimate(img):
    """
    Background estimation

    Parameters
    ----------
    img : 2D ndarray
        Original image.

    Returns
    -------
    bkg : 2D ndarray
        A frame of the background estimate.

    """
    bkg = sep.Background(img, bw=64, bh=64, fw=3, fh=3)
    bkg = bkg.back()
    return bkg

def deconvolve(img, psf, num_iter = 200):
    """
    A function to deconvolve a @D image based on a given psf.
    At the moment, this function rely on the imperfect richardson-Lucy method.
    Parameters
    ----------
    img : 2D ndarray
        Original image.
    psf : 2D ndarray
        PSF image.
    num_iter : int, optional
        Number of iteration. The default is 200.

    Returns
    -------
    deconv_img : 2D ndarray
        Deconvolved image.

    """
    # under construction
    
    # frame expansion
    #blank = np.zeros((800,800))
    #blank[200:img.shape[0]+200,200:img.shape[0]+200] = img
    
    deconv_img0 = restoration.richardson_lucy(img, psf, num_iter=num_iter)
    
    #resize again
   # deconv_img = deconv_img0[200:img.shape[0],200:img.shape[0]]
    
    #otf = fftn(fftshift(psf))
    #otf_ = np.conjugate(otf)    
    #estimate = img#np.ones(image.shape)/image.sum()

    #for i in range(num_iter):
    #    #print(i)
    
    #    reblurred = ifftn(conv2(fftn(estimate),otf))
    #    ratio = img / (reblurred + 1e-30)
    #    estimate = estimate * (ifftn(fftn(ratio) * otf_)).astype(float)

    #return estimate
   # deconv_img = None
    #print("deconv_img", deconv_img.shape)
   
    return deconv_img0

def denoise():
    #under construction
    return None

#%% Plotting
def plot_three_frame(original, noisy, restore,
                     frame1_tit="Original Data (RAW+PSF)",
                     frame2_tit='Noisy data\n(RAW+PSF+Noise)',
                     frame3_tit='Restoration using\nRichardson-Lucy'):
    """
    Plot three frame

    Parameters
    ----------
    original : 2D ndarray
        original image.
    noisy : 2D ndarray
        Noisy Frame.
    restore : 2D ndarray
        restored image.

    Returns
    -------
    fig : plot
        Plot.

    """
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
           a.axis('off')
           
    ax[0].imshow(original, vmin=original.min(), vmax=original.max())
    ax[0].set_title(frame1_tit)

    ax[1].imshow(noisy, vmin=noisy.min(), vmax=noisy.max())
    ax[1].set_title(frame2_tit)

    ax[2].imshow(restore, vmin=original.min(), vmax=original.max())# vmin=astro_noisy.min(), vmax=astro_noisy.max())
    ax[2].set_title(frame3_tit)
    plt.show()

    return fig 


def draw_ellipses(img, objects,ref):
    """
    To plot ellipses around bright objects

    Parameters
    ----------
    img : 2D ndarray
        Original image.
    objects : 2D ndarray
        Bright object list produced by source_detect.
    ref : 2D ndarray
        Reference image for adjusting contrast level.
    
    Returns
    -------
    None.

    """
#   plot background-subtracted image
    fig, ax = plt.subplots()
    mean, std = np.mean(img), np.std(img)
    im = ax.imshow(img, interpolation='nearest', cmap='gray',
               vmin=ref.min(), vmax=ref.max())

    # plot an ellipse for each object
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6*objects['a'][i],
                    height=6*objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    return fig

#%%
rng = np.random.default_rng()
astro_og = color.rgb2gray(data.immunohistochemistry())

#psf = np.random.multivariate_normal(mean, cov, (3,3))
psf = gaussian_filter(10,sigma=5,muu=0)

astro = conv2(astro_og, psf, 'same')
# Add Noise to Image

astro_noisy = astro.copy()
astro_noisy += gen_noise_poisson(255, astro.shape, norm = 255) #(rng.poisson(lam=25, size=astro.shape) - 10) / 255.


# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro, psf, num_iter=500)
deconvolved_RL2 = deconvolve(astro, psf, num_iter=500)

#deconvolved_RL_denoise = deconvolved_RL - noise_frame



#plot_three_frame(astro,astro_noisy, deconvolved_RL_denoise)


#fig = plt.subplot()
#plt.imshow(astro_og-deconvolved_RL)
#plt.show()

#plot_three_frame(astro_og, deconvolved_RL, astro_og-deconvolved_RL)


lily = color.rgb2gray(data.hubble_deep_field())
#lily = data.hubble_deep_field()

#noise_frequency = ifftn(lily)

#fig = plt.subplot()
#plt.imshow(np.real(noise_frequency), norm='linear', #cmap=cm.Reds,
#           vmin=np.real(noise_frequency).min(), 
#           vmax=np.real(noise_frequency).max())
#plt.show()


# create a sharpening kernel
sharpen_filter = sharpen_filter(9,surround = -1)
# applying kernels to the input image to get the sharpened image
sharp_image=cv2.filter2D(lily,-1,sharpen_filter)

#fig = plt.subplot()
#plt.imshow(lily, vmin=astro_og.min(), vmax=astro_og.max())
#plt.show()

#lily = conv2(lily, psf, 'same')
#lily_deconv = deconvolve(lily, psf)


#plot_three_frame(astro,astro_noisy, deconvolved_RL)
plot_three_frame(astro,astro_noisy, deconvolved_RL2)

#plot_three_frame(lily, psf, lily_deconv)


bkg = bg_estimate(astro)


# spark detection
#sparky = add_spark(astro, spark_num =20, connect_pix= 400)
#plot_three_frame(astro,sparky, astro+sparky,
#                 'Original Data', 'Sparks', "Composed Image")

#bright_spot = source_detect(astro+sparky)
#draw_ellipses(astro+sparky, bright_spot,astro)