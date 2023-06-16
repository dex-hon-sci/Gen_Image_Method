#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:17:56 2023

@author: dexter
"""

#Train RS3 image enhancement

import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# resize image
def resize():
    
    return None
# fill image 
#generate noise frame

def gen_noise(frame):
    # Generate random possionian noise
    rng = np.random.default_rng()

    noise_frame = (rng.poisson(lam=25, size=frame.shape) - 10) / 255.

    return noise_frame 