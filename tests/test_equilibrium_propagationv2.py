import pytest

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

from dynamics.equilibrium_propagation import *
from dynamics.equilibrium_propagationv2 import *

def test_forcing():
    x = tf.random_normal([5, 8])
    inputs = tf.random_normal([5, 4])
    input_idx = tf.range(4)

    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = forcing_fn(x, inputs, input_idx)

    grad = tape.gradient(loss, x)

    assert grad is not None
