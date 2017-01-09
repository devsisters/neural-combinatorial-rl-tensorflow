import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.util import nest

LSTMCell = rnn.LSTMCell
MultiRNNCell = rnn.MultiRNNCell
LSTMStateTuple = rnn.LSTMStateTuple

def trainable_initial_state(batch_size, state_size, initializer=None):
  flat_state_size = nest.flatten(state_size)

  if not initializer:
    flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
  else:
    flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

  names = ["init_state_{}".format(i) for i in xrange(len(flat_state_size))]

  tiled_states = []

  for name, size, init in zip(names, flat_state_size, flat_initializer):
    shape_with_batch_dim = [1] + [size]
    initial_state_variable = tf.get_variable(
        name, shape=shape_with_batch_dim, initializer=init())

    tile_dims = [batch_size] + [1]
    tiled_states.append(
        tf.tile(initial_state_variable, tile_dims, name=(name + "_tiled")))

  return nest.pack_sequence_as(structure=state_size,
                               flat_sequence=tiled_states)
