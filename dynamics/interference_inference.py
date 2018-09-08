import tensorflow as tf
import numpy as np

from absl import flags

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "epochs", default=50, help="Number of training steps to run.")
flags.DEFINE_string(
    "activation",
    default="selu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_string(
    "data_dir",
    default="/tmp/mnist",
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default="/tmp/myelin/",
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=500, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")
flags.DEFINE_integer(
    "n_steps", default=10, help="Number of forward steps to take.")
flags.DEFINE_integer(
    "n_hidden", default=32, help="Number of hidden units.")
flags.DEFINE_float(
    "beta", default=1.0, help="Beta.")


class Network():
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

        self.weights = tf.Variable(tf.random_normal([n_nodes, n_nodes]))

    def step(self, state):
        

        return state
