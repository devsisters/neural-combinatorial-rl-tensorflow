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

def decoder_rnn(cell, inputs,
                enc_outputs, enc_final_states,
                seq_length, hidden_dim, num_glimpse,
                max_dec_length, batch_size, is_train, end_of_sequence_id=0):
  with tf.variable_scope("decoder_rnn") as scope:
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

    maximum_length = tf.convert_to_tensor(max_dec_length, tf.int32)
    def decoder_fn_inference(
        time, cell_state, cell_input, cell_output, context_state):
      if context_state is None:
        context_state = tf.TensorArray(tf.float32, size=maximum_length)

      if cell_output is None:
        # invariant tha this is time == 0
        cell_state = enc_final_states
        cell_input = first_decoder_input
        done = tf.zeros([batch_size,], dtype=tf.bool)
      else:
        output_logit = output_fn(enc_outputs, cell_output, num_glimpse)
        sampled_idx = tf.multinomial(output_logit, 1)

        context_state.write(time, output_logit)
        done = tf.squeeze(tf.equal(sampled_idx, end_of_sequence_id), -1)

      done = tf.cond(tf.greater(time, maximum_length),
          lambda: tf.ones([batch_size,], dtype=tf.bool),
          lambda: done)
      return (done, cell_state, cell_input, cell_output, context_state)

    if is_train:
      decoder_fn = simple_decoder_fn_train(enc_final_states)
      inputs = tf.concat_v2([inputs, tf.expand_dims(first_decoder_input, 1)], 1)
    else:
      decoder_fn = decoder_fn_inference
      inputs = tf.expand_dims(first_decoder_input, 1)

    outputs, final_state, final_context_state = \
        dynamic_rnn_decoder(cell, decoder_fn, inputs=inputs,
                            sequence_length=seq_length, scope=scope)

    if is_train:
      output_logits = []
      unstacked_outputs = tf.unstack(outputs, max_dec_length, axis=1)
      for output in unstacked_outputs:
        output_logit = output_fn(enc_outputs, output, num_glimpse)
        scope.reuse_variables()
        output_logits.append(output_logit)
      outputs = tf.stack(output_logits, 1)

    return outputs, final_state, final_context_state

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
