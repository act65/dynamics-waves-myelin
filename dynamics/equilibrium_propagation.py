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
    default="/tmp/equib/",
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

FLAGS = flags.FLAGS

def energy_fn(x, W, b):
    """
    Somehow related to Hopfield nets?
    Define what we mean by 'energy'

    - neighbor energy
    i get the part that measures something like the label propagation.
    distance between two strongly connected nodes should be small

    What alternatives are there? Want to explore these!

    """
    with tf.name_scope('energy_fn'):

        h = tf.nn.sigmoid(x)

        energy = -0.5*tf.reduce_sum(tf.matmul(h, tf.matmul(W, h, transpose_b=True)), axis=1)
        energy += 0.5*tf.reduce_sum(tf.square(h*b), axis=1)
        energy += 0.5*tf.reduce_sum(x**2, axis=1)

        return tf.reduce_mean(energy)

def forcing_fn(state, vals, idx):
    """
    How can I get grads w.r.t the parameters!?
    dLdparam = mse(state, target)
    """
    with tf.name_scope('forcing_fn'):
        print('i', idx)
        print('s',tf.gather(state, idx, axis=1))
        return tf.losses.mean_squared_error(tf.gather(state, idx, axis=1), vals)

def energy_fnv2(x, W, b):
    h = tf.nn.relu(x)

    # use the graph laplacian to measure differences between neighbors
    # how to calculate the degree!?
    # D = tf.diag(tf.abs(tf.reduce_sum(W, axis=1))**-0.5)
    # L = tf.eye(tf.shape(W)[0]) - tf.matmul(D, tf.matmul(W, D))
    L = tf.diag(tf.reduce_sum(W, axis=1)) - W

    neighbor_energy = 0.5*tf.reduce_sum(tf.square(tf.matmul(L, h, transpose_b=True)), axis=0)
    acivation_energy = 0.005*tf.reduce_sum(tf.square(x), axis=1)
    biased_energy = -tf.reduce_sum(h*b, axis=1)


    return tf.reduce_mean(
    neighbor_energy
    + acivation_energy
    + biased_energy
    )

def get_sym_adj(n_nodes):
    """
    Why does the adjacency matrix need to be symmetric?
    Else we cant prove that the back prop is equivalent?
    """
    mat = tf.random_normal(shape=[n_nodes, n_nodes], dtype=tf.float32)
    mat = tf.Variable(mat, name='weights')
    sym = (mat + tf.transpose(mat))/2
    adj = sym - tf.eye(n_nodes)*sym
    return adj, mat

class Network():
    """
    https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop
    Rather than having two phases, want the nodes to have some temporal state.
    If some input values were recently 'clamped' then they should
    correlate with output values that are 'clamped' not long afterward.

    So there exists a delay between the clamping of the inputs and the outputs.
    What happens if;
    - delay is large
    - delay is variable
    - ?

    Might not even need to do anything smart? Bc SGD will want to find the shortest path from
    old state (clamped at inputs), to new state (clamped at labels).
    Optimise the parameters the minimize the distance travelled by the state!?

    Pros:
    - Can easily add more nodes or new inputs

    Cons:
    - must simulate for n steps rather than 1 shot prediction
    - ?
    """
    def __init__(self, n_inputs, n_hidden, n_outputs, name=''):
        self.n_nodes = n_inputs + n_hidden + n_outputs
        self.beta = 1.0

        self.input_idx = tf.range(n_inputs)
        self.output_idx = tf.range(n_inputs+n_hidden, self.n_nodes)

        with tf.variable_scope('network'):
            # TODO sparse matrix would be nicer/faster!?
            self.weights, self.weights_var = get_sym_adj(self.n_nodes)
            self.biases = tf.Variable(tf.random_normal(shape=[1, self.n_nodes], dtype=tf.float32), name='biases')
        self.variables = [self.weights_var, self.biases]

    def step(self, state, vals=None, idx=None):
        """
        Args:
            state (tf.tensor): the current state of the network
                shape = [batch_size, n_nodes], dtype = tf.float32
            vals (tf.tensor): the values to clamp certain nodes
                shape = [batch_size, N], dtype = tf.float32
            idx (tf.tensor): the indices of the tensors to clamp
                shape = [1], dtype = tf.int64

        Returns:
            new_state (tf.tensor): the new state of the network
                shape = [batch_size, n_nodes], dtype = tf.float32
        """
        with tf.name_scope('step'):
            # Always trying to find a state with lower enegy
            loss = energy_fn(state, self.weights, self.biases)
            if vals is not None and idx is not None:
                loss += self.beta*forcing_loss(state, vals, idx)

            grad = tf.gradients(loss, state)[0]
            # grad = tf.clip_by_norm(grad, 1.0)

        with tf.name_scope('gd'):
            # TODO want smarter optimisation here. AMSGrad!?
            new_state = state - 0.5*grad
            return new_state + 0.001*tf.random_normal(tf.shape(new_state))  # add some noise into the dynamics

    def forward(self, state, vals=None, idx=None, n_steps=10):
        """
        Use while loop to take advantage of tf's compiler optimisations!?
        but the problem is we now have a finite window of data we can view.
        """
        # TODO forward AD
        def step(i, state):
            # a wrapper for self.step(...)
            return i + 1, self.step(state, vals, idx)

        with tf.name_scope('forward'):
            while_condition = lambda i, m : tf.less(i, n_steps)   # TODO change to state - old_state!? or low loss
            i = tf.constant(0)
            i_, new_state = tf.while_loop(while_condition, step, loop_vars=[i, state], back_prop=False)

            return new_state

def model_fn(features, labels, mode, params, config):
    x = features['x']
    net = Network(28*28, params['n_hidden'], 10)

    tf.summary.image('adjacency', tf.reshape(net.weights, [1, net.n_nodes, net.n_nodes, 1]))
    tf.summary.image('bias', tf.reshape(tf.stack([net.biases for _ in range(30)]), [1, 30, net.n_nodes, 1]))

    init_state = tf.zeros([tf.shape(x)[0], net.n_nodes])

    ###########################################################################
    ### supervised learning ###
    # clamp inputs
    state_f = net.forward(init_state, x, net.input_idx, n_steps=params['n_steps'])

    im = tf.gather(state_f, net.input_idx, axis=1)
    pred = tf.gather(state_f, net.output_idx, axis=1)
    tf.summary.histogram('state_f', state_f)
    tf.summary.image('clamped_xs', tf.reshape(im, [tf.shape(x)[0], 28, 28, 1]))

    # clamp inputs and outputs
    all_inputs = tf.concat([x, tf.one_hot(labels, 10, 1.0, 0.0, dtype=tf.float32)], axis=1)
    all_idx = tf.concat([net.input_idx, net.output_idx], axis=0)
    state_b = net.forward(state_f, all_inputs, all_idx, n_steps=params['n_steps'])
    tf.summary.histogram('state_b', state_b)

    loss = net.energy_loss(state_b) # - net.energy_loss(state_f)

    # WANT minimise the distance to be travelled/the changes to be made. lazy. and the energy!?
    # loss = tf.losses.mean_squared_error(state_f, state_b)  # should stop grad through f?

    # just a test to see if forward pass is working.
    # loss = tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=labels)


    ###########################################################################
    ### unsupervised learning ###
    # noised_input = x + 0.1*tf.random_normal(tf.shape(x))
    # tf.summary.image('noised_input', tf.reshape(noised_input, [tf.shape(x)[0], 28, 28, 1]))
    #
    # # clamp inputs
    # state_f = net.forward(init_state, noised_input, net.input_idx, n_steps=params['n_steps'])
    # im = tf.gather(state_f, net.input_idx, axis=1)
    # tf.summary.histogram('state_f', state_f)
    # tf.summary.image('clamped_xs', tf.reshape(im, [tf.shape(x)[0], 28, 28, 1]))
    #
    # # run forward for a few more time steps (without clamping)
    # state_b = net.forward(state_f, n_steps=params['n_steps'])
    # recon = tf.gather(state_b, net.input_idx, axis=1)
    # tf.summary.image('recon', tf.reshape(recon, [tf.shape(x)[0], 28, 28, 1]))
    #
    # loss = tf.losses.mean_squared_error(x, im)  # reconstruction error

    ###########################################################################

    # training
    opt = tf.train.AdamOptimizer(params['learning_rate'])
    gnvs = opt.compute_gradients(loss, var_list=net.variables)
    for g, v in gnvs:
        tf.summary.scalar(v.name, tf.norm(g))
    gnvs = [(tf.clip_by_norm(g, 100.0), v) for g, v in gnvs]
    train_op = opt.apply_gradients(gnvs, global_step=tf.train.get_or_create_global_step())

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
      "accuracy": tf.metrics.accuracy(labels, tf.argmax(pred, axis=1))
      # "mean_loss": tf.metrics.mean(loss)
      }
    )

def main(_):
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])

    if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
        tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images[:5000, ...]  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)[:5000, ...]
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data = train_data
    eval_labels = train_labels

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": train_data},
          y=train_labels,
          batch_size=FLAGS.batch_size,
          num_epochs=1,
          shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=False)


    estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
    )

    for _ in range(FLAGS.epochs):
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)

if __name__ == "__main__":
    tf.app.run()
