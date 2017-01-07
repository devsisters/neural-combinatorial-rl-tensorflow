import tensorflow as tf
from tensorflow.python.util import nest

LSTMCell = tf.contrib.rnn.LSTMCell
MultiRNNCell = tf.contrib.rnn.MultiRNNCell

def get_tiled_state(cell, inputs, initializer, name="tiled_state"):
  state_size = cell.state_size

	if nest.is_sequence(state_size):
    state_size_flat = nest.flatten(state_size)

		init_state_flat = [
      tf.get_variable("{}_{}".format(name, idx), cell.state_size)
        for idx, s in enumerate(state_size_flat)]
    init_state = nest.pack_sequence_as(structure=state_size,
                  flat_sequence=init_state_flat)
  else:
    init_state_size = _state_size_with_prefix(state_size)
    init_state = initializer(init_state_size, batch_size, dtype, None)


  batch_size = tf.shape(inputs)[0]
  initial_state = tf.get_variable(
      "initial_state", [1, hidden_dim])
  tiled_initial_state = tf.tile(initial_state, [batch_size, 1])
  return initial_state, tiled_initial_state
