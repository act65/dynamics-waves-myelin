import unittest
import tensorflow as tf
import numpy as np

from code.equilibrium_propagation import *

class TestNetwork(unittest.TestCase):
    def test_step_shape(self):
        state = tf.random_normal([1, 9])
        net = Network(3, 3, 3)
        state_t = net.step(state)
        self.assertEqual(state.shape, state_t.shape)

    def test_step_grad(self):
        """check that we can get a gradient for the parameters through the step fn"""
        state = tf.random_normal([1, 9])
        net = Network(3, 3, 3)
        state_t = net.step(state)

        loss = tf.reduce_mean(state_t**2)
        param_grads = tf.gradients(loss, net.weights)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            G = sess.run(param_grads)

        bools = np.equal(G, np.zeros_like(G))
        self.assertTrue(not bools.all())
        self.assertTrue(not np.isnan(G).any())

    def test_forward_grad(self):
        """check that we can get a gradient for the parameters through the forward"""
        state = tf.random_normal([1, 9])
        net = Network(3, 3, 3)
        state_t = net.forward(state)

        loss = tf.reduce_mean(state_t**2)
        param_grads = tf.gradients(loss, net.weights)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            G = sess.run(param_grads)

        bools = np.equal(G, np.zeros_like(G))
        self.assertTrue(not bools.all())
        self.assertTrue(not np.isnan(G).any())

    def test_forward_grad_w_forcing(self):
        """check that we can get a gradient for the parameters through the forward
        with clamped values"""
        state = tf.random_normal([1, 9])
        net = Network(3, 3, 3)

        x = tf.random_normal([1, 3])

        state_t = net.forward(state, x, net.input_idx)

        loss = tf.reduce_mean(state_t**2)
        param_grads = tf.gradients(loss, net.weights)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            G = sess.run(param_grads)

        bools = np.equal(G, np.zeros_like(G))
        self.assertTrue(not bools.all())
        self.assertTrue(not np.isnan(G).any())

class TestUtils(unittest.TestCase):
    def test_energy(self):
        n = 1000
        x = tf.random_normal([1, n])
        W = tf.random_normal([n, n])
        b = tf.random_normal([1, n])
        energy = energy_fn(x, W, b)

        grads = tf.gradients(energy, x)[0]
        grads = tf.clip_by_norm(grads, 1.0)

        with tf.Session() as sess:
            E = sess.run(energy)
            G = sess.run(grads)

        bools = np.equal(G, np.zeros_like(G))
        self.assertTrue(not bools.all())
        self.assertTrue(not np.isnan(G).any())


if __name__ == '__main__':
    unittest.main()