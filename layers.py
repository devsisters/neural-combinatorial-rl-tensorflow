import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest

linear = layers.linear
LSTMCell = rnn.LSTMCell
MultiRNNCell = rnn.MultiRNNCell
LSTMStateTuple = rnn.LSTMStateTuple
dynamic_rnn_decoder = seq2seq.dynamic_rnn_decoder
simple_decoder_fn_train = seq2seq.simple_decoder_fn_train

def decoder_rnn(cell, transformed_dec_inputs,
                enc_outputs, enc_final_states, dec_init_state,
                seq_length, hidden_dim, num_glimpse, is_train):
  with tf.variable_scope("decoder_rnn") as scope:
    batch_size = tf.shape(transformed_dec_inputs)[0]
    max_dec_length = transformed_dec_inputs.get_shape()[1]

    first_decoder_input = trainable_initial_state(
        batch_size, hidden_dim, name="first_decoder_input")

    def attention(ref, query, with_softmax=True, scope="attention"):
      with tf.variable_scope(scope):
        W_ref = tf.get_variable("W_ref", [1, hidden_dim, hidden_dim])
        W_q = tf.get_variable("W_q", [hidden_dim, hidden_dim])
        v = tf.get_variable("v", [hidden_dim])

        encoded_ref = tf.nn.conv1d(ref, W_ref, 1, "VALID")
        encoded_query = tf.matmul(tf.reshape(query, [-1, hidden_dim]), W_q)
        encoded_query = tf.reshape(encoded_query, tf.shape(query))
        scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])

        if with_softmax:
          return tf.nn.softmax(scores)
        else:
          return scores

    def glimpse(ref, query, scope="glimpse"):
      p = attention(ref, query, scope=scope)
      alignments = tf.expand_dims(p, 2)
      return tf.reduce_sum(alignments * ref, [1])

    def output_fn(ref, query, num_glimpse):
      for idx in range(num_glimpse):
        query = glimpse(ref, query, "glimpse_{}".format(idx))
      return attention(ref, query, with_softmax=False)

    def decoder_fn_inference(
        time, cell_state, cell_input, cell_output, context_state):
      pass

    if is_train:
      decoder_fn = simple_decoder_fn_train(enc_final_states)
    else:
      decoder_fn = decoder_fn_inference

    outputs, states = dynamic_rnn_decoder(
        cell, decoder_fn, inputs=transformed_dec_inputs,
        sequence_length=seq_length, scope=scope)

    if is_train:
      output_logits = []
      unstacked_outputs = tf.unstack(outputs, max_dec_length, axis=1)
      for output in unstacked_outputs:
        output_logit = output_fn(enc_outputs, output, num_glimpse)
        scope.reuse_variables()
        output_logits.append(output_logit)
      outputs = tf.stack(output_logits, 1)

    return outputs, states

def trainable_initial_state(batch_size, state_size,
                            initializer=None, name="initial_state"):
  flat_state_size = nest.flatten(state_size)

  if not initializer:
    flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
  else:
    flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

  names = ["{}_{}".format(name, i) for i in xrange(len(flat_state_size))]

  tiled_states = []

  for name, size, init in zip(names, flat_state_size, flat_initializer):
    shape_with_batch_dim = [1, size]
    initial_state_variable = tf.get_variable(
        name, shape=shape_with_batch_dim, initializer=init())

    tiled_state = tf.tile(initial_state_variable,
                          [batch_size, 1], name=(name + "_tiled"))
    tiled_states.append(tiled_state)

  return nest.pack_sequence_as(structure=state_size,
                               flat_sequence=tiled_states)

def index_matrix_to_pairs(index_matrix):
  replicated_first_indices = tf.tile(
      tf.expand_dims(tf.range(tf.shape(index_matrix)[0]), dim=1), 
      [1, tf.shape(index_matrix)[1]])
  return tf.pack([replicated_first_indices, index_matrix], axis=2)

#with tf.name_scope("decoder_fn_inference",
#    [time, cell_state, cell_input, cell_output, context_state]):
#  for idx in range(num_glimpse):
#    query = glimpse(enc_outputs, query)
#  output = attention(enc_outputs, query)
#
#  output_fn = lambda x: linear(x, num_decoder_symbols, scope=varscope)
#
#  with ops.name_scope(name, "simple_decoder_fn_inference",
#                      [time, cell_state, cell_input, cell_output,
#                      context_state]):
#    if cell_input is not None:
#      raise ValueError("Expected cell_input to be None, but saw: %s" %
#                      cell_input)
#    if cell_output is None:
#      # invariant that this is time == 0
#      next_input_id = tf.ones([batch_size,], dtype=dtype) * (
#          start_of_sequence_id)
#      done = tf.zeros([batch_size,], dtype=dtypes.bool)
#      cell_state = encoder_state
#      cell_output = tf.zeros([num_decoder_symbols],
#                                    dtype=dtypes.float32)
#    else:
#      cell_output = output_fn(cell_output)
#      next_input_id = tf.cast(
#          tf.argmax(cell_output, 1), dtype=dtype)
#      done = tf.equal(next_input_id, end_of_sequence_id)
#      
#    next_input = tf.gather(embeddings, next_input_id)
#    # if time > maxlen, return all true vector
#    done = control_flow_ops.cond(tf.greater(time, maximum_length),
#        lambda: tf.ones([batch_size,], dtype=dtypes.bool),
#        lambda: done)
#    return (done, cell_state, next_input, cell_output, context_state)
#
#  # next_input = tf.concat_v2([cell_input, output], 1)
#  next_input = output
#
#  return (None, cell_state, next_input, cell_output, context_state)
