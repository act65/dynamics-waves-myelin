import tensorflow as tf
import tangent
import numpy as np

from dynamics import equilibrium_propagation as eq

def loss_fn(x, W, b, vals=None, idx=None):
    loss = 1e-4*eq.energy_fn(x, W, b)

    if vals is not None and idx is not None:
        loss += eq.forcing_fn(x, vals, idx)

    return loss

class Network():
    def __init__(self, n_nodes):
        self.state_opt = tf.train.AdamOptimizer(0.01)
        self.param_opt = tf.train.AdamOptimizer(0.0001)

        self.n_nodes = n_nodes
        self.beta = 10

        self.state = None
        self.weights = tf.contrib.eager.Variable(tf.random_normal([self.n_nodes, self.n_nodes]))
        self.biases = tf.contrib.eager.Variable(tf.random_normal([1, self.n_nodes]))
        # self.weights = tf.random_normal([self.n_nodes, self.n_nodes])
        # self.biases = tf.random_normal([1, self.n_nodes])

    def _init_state(self, batch_size):
        # self.state = tf.contrib.eager.Variable(tf.random_normal([batch_size, self.n_nodes]))
        self.state = tf.zeros([batch_size, self.n_nodes])

    def step(self, vals=None, idx=None, train=True):
        if self.state is None:
            self._init_state(tf.shape(vals)[0])

        with tf.GradientTape() as tape:
            tape.watch([self.state, self.weights, self.biases])
            loss = loss_fn(self.state, self.weights, self.biases, vals=vals, idx=idx)
        grads = tape.gradient(loss, [self.state, self.weights, self.biases])

        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads ]
        # self.state_opt.apply_gradients(zip(grads[:1], [self.state]))

        if train:
            self.param_opt.apply_gradients(zip(grads[1:], [self.weights, self.biases]),
            global_step=tf.train.get_or_create_global_step())
            # self.weights -= 1.0*grads[1]
            # self.biases -= 1.0*grads[2]

        self.state -= 200.0*grads[0]


        return self.state + 0.0001*tf.random_normal(tf.shape(self.state))

def main():
    batch_size = 32
    n_steps = 15

    n_inputs = 28*28
    n_hidden = 500
    n_outputs = 10

    n_nodes = n_inputs + n_hidden + n_outputs
    net = Network(n_nodes)

    input_idx = tf.range(n_inputs)
    output_idx = tf.range(n_outputs)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    global_step=tf.train.get_or_create_global_step()

    writer = tf.contrib.summary.create_file_writer('/tmp/online/0')
    writer.set_as_default()

    with tf.contrib.summary.record_summaries_every_n_global_steps(n_steps):

        for i in range(1000):
            batch_ims, batch_labels = mnist.train.next_batch(batch_size)

            for j in range(n_steps):
                x = net.step(batch_ims, input_idx, train=False)

            clamped_img = tf.gather(x, input_idx, axis=1)
            pred = tf.gather(x, output_idx, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), tf.constant(batch_labels, tf.int64)), tf.float32))
            tf.contrib.summary.scalar('acc', acc)
            tf.contrib.summary.image('clamped_xs', tf.reshape(clamped_img, [batch_size, 28, 28, 1]))

            for k in range(n_steps):
                net._init_state(tf.shape(x)[0])
                all_inputs = tf.concat([batch_ims, tf.one_hot(batch_labels, 10, 1.0, 0.0)], axis=1)
                all_idx = tf.concat([input_idx, output_idx], axis=0)
                x = net.step(all_inputs, all_idx)

            energy = eq.energy_fn(net.state, net.weights, net.biases)
            forcing = eq.forcing_fn(net.state, all_inputs, all_idx)
            tf.contrib.summary.scalar('loss', 1e-4*energy + forcing)
            tf.contrib.summary.scalar('energy', energy)
            tf.contrib.summary.scalar('forcing', forcing)

            tf.contrib.summary.image('adjacency', tf.reshape(net.weights, [1, n_nodes, n_nodes, 1]))

            print('\r Step: {} Acc: {:.3f}'.format(i, acc), end='', flush=True)

if __name__ == '__main__':
    tf.enable_eager_execution()
    main()
