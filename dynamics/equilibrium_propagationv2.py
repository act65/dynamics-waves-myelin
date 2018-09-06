import tensorflow as tf
import tangent
import numpy as np

from dynamics import equilibrium_propagation as eq

def loss_fn(x, W, b, vals=None, idx=None):
    # loss = 1e-8*eq.energy_fn(x, W, b)

    # if vals is not None and idx is not None:
        # print(loss)
    print(vals)
    loss = eq.forcing_fn(x, vals, idx)
        # print(eq.forcing_fn(x, vals, idx))
        # print(loss)

    return loss

class Network():
    def __init__(self):
        self.state_opt = tf.train.AdamOptimizer(0.01)
        self.param_opt = tf.train.AdamOptimizer(0.0001)

    def step(self, state, W, b, vals, idx):

        # QUESTION why is `benergy` needed?
        with tf.GradientTape() as tape:
            loss = loss_fn(state, W, b, vals=vals, idx=idx)
        grads = tape.gradient(loss, [state, W, b])
        self.state_opt.apply_gradients(zip(grads[:1], [state]))
        # self.param_opt.apply_gradients(zip(grads[1:], [W, b]))

        return state, W, b

def main():
    batch_size = 32
    n_steps = 8

    n_inputs = 28*28
    n_hidden = 500
    n_outputs = 10

    n_nodes = n_inputs + n_hidden + n_outputs
    net = Network()

    x = tf.contrib.eager.Variable(tf.random_normal([batch_size, n_nodes]))
    W = tf.contrib.eager.Variable(tf.random_normal([n_nodes, n_nodes]))
    b = tf.contrib.eager.Variable(tf.random_normal([1, n_nodes]))

    input_idx = tf.range(n_inputs)
    output_idx = tf.range(n_outputs)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    global_step=tf.train.get_or_create_global_step()

    writer = tf.contrib.summary.create_file_writer('/tmp/test_forward/0')
    writer.set_as_default()

    with tf.contrib.summary.record_summaries_every_n_global_steps(n_steps*2):

        for i in range(100):
            batch_ims, batch_labels = mnist.train.next_batch(batch_size)

            for j in range(n_steps):
                x, W, b = net.step(x, W, b, batch_ims, input_idx)

            pred = tf.gather(x, input_idx, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), batch_labels), tf.float32))
            tf.contrib.summary.scalar('acc', acc)
            tf.contrib.summary.image('clamped_xs', tf.reshape(pred, [batch_size, 28, 28, 1]))

            for k in range(n_steps):
                x, W, b = net.step(x, W, b, tf.one_hot(batch_labels, 10, 1.0, 0.0), output_idx)

            L = loss_fn(x, W, b)

            tf.contrib.summary.scalar('loss', L)
            tf.contrib.summary.image('adjacency', tf.reshape(W, [1, n_nodes, n_nodes, 1]))

            global_step.assign_add(1)

            print('\r Step: {} Loss: {:.3f}'.format(i, L), end='', flush=True)

if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
