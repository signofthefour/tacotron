from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import sys
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 10000

MAX_TEXT_LENGTH = 140

def build_dataset(sess, inputs, names):
    place_holders = []
    for inp in inputs:
        place_holders.append(tf.placeholer(inp.dtype, inp.shape))

    with tf.device('/cpu:0'):
        dataset = 