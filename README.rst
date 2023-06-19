*Gen_Image_method*
================
A toy project to create a generic image processing package.

The project consist of two main parts:
	1) Image enhancement via Machine learning, and
	2) Error detection and correction methods in image processing.

The aim of this project is to create a set of useful functions and network for more in-depth project related to Astronomical image processing or Atomic force microscpy in the future.

At the moment, few functions on source detection and deconvolution are built.

Quick Examples
==============
Adding bright sparks
--------------------
.. code:: python

	import noise_handling as N
	from skimage import color, data, restoration
	from scipy.signal import convolve2d as conv2

	# Load image
	Image = color.rgb2gray(data.immunohistochemistry())
	
	# Add sparks to the frame
	sparky = N.add_spark(Image, spark_num =20, connect_pix= 400)
	
	# Plot the image+ sparks
	N.plot_three_frame(Image,sparky, astro+sparky,'Original Data', 'Sparks', "Composed Image")

.. image:: https://github.com/dex-hon-sci/Gen_Image_Method/blob/master/images/spark_experiment.png

Source detection
----------------
.. code:: python

	# Detecting bright sources
	bright_spot = N.source_detect(astro+sparky)
	
	# Plot the detection
	N.draw_ellipses(astro+sparky, bright_spot,astro)

.. image:: https://github.com/dex-hon-sci/Gen_Image_Method/blob/master/images/spark_detection.png

Creating a noisy convolved image and Deconvolve it
--------------------------------------------------
.. code:: python

	# Generate a Gaussian point-spread-function
	psf = N.gaussian_filter(10,sigma=5,muu=0)

	# 2D convolution
	Image = conv2(Image, psf, 'same')
	
	# Add Noise to Image
	Image_noisy = Image.copy()
	Image_noisy += N.gen_noise_poisson(255, Image.shape, norm = 255) 
	
	# Deconvolution
	deconvolved_Image = N.deconvolve(Image_noisy, psf, num_iter=500)
	
	# Plot 
	N.plot_three_frame(Image,Image_noisy, deconvolved_Image)
	
	# There are much room to be improved on the deconvolution method

.. image:: https://github.com/dex-hon-sci/Gen_Image_Method/blob/master/images/convolve_experiment.png
