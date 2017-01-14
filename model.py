import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *
from utils import show_all_variables

class Model(object):
  def __init__(self, config, data_loader, is_critic=False):
    self.data_loader = data_loader

    self.task = config.task
    self.debug = config.debug
    self.config = config

    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers

    self.max_enc_length = config.max_enc_length
    self.max_dec_length = config.max_dec_length
    self.num_glimpse = config.num_glimpse

    self.init_min_val = config.init_min_val
    self.init_max_val = config.init_max_val
    self.initializer = \
        tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

    self.use_terminal_symbol = config.use_terminal_symbol

    self.lr_start = config.lr_start
    self.lr_decay_step = config.lr_decay_step
    self.lr_decay_rate = config.lr_decay_rate
    self.max_grad_norm = config.max_grad_norm

    self.layer_dict = {}

    #self._build_input_ops()
    self._build_model()
    if is_critic:
      self._build_critic_model()

    #self._build_optim()
    #self._build_summary()

    show_all_variables()

  def _build_summary(self):
    tf.summary.scalar("learning_rate", self.lr)

  def _build_critic_model(self):
    pass

  def _build_input_ops(self):
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)

  def _build_model(self):
    self.global_step = tf.Variable(0, trainable=False)

    input_weight = tf.get_variable(
        "input_weight", [1, self.input_dim, self.hidden_dim],
        initializer=self.initializer)

    with tf.variable_scope("encoder"):
      self.enc_seq_length = tf.placeholder(
          tf.int32, [None], name="enc_seq_length")
      self.enc_inputs = tf.placeholder(
          tf.float32, [None, self.max_enc_length, self.input_dim],
          name="enc_inputs")
      self.transformed_enc_inputs = tf.nn.conv1d(
          self.enc_inputs, input_weight, 1, "VALID")

    batch_size = tf.shape(self.enc_inputs)[0]
    with tf.variable_scope("encoder"):
      self.enc_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.enc_cell] * self.num_layers
        self.enc_cell = MultiRNNCell(cells)
      self.enc_init_state = trainable_initial_state(
          batch_size, self.enc_cell.state_size)

      # self.encoder_outputs : [None, max_time, output_size]
      self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
          self.enc_cell, self.transformed_enc_inputs,
          self.enc_seq_length, self.enc_init_state)

      if self.use_terminal_symbol:
        tiled_zeros = tf.tile(tf.zeros(
          [1, self.hidden_dim]), [batch_size, 1], name="tiled_zeros")
        expanded_tiled_zeros = tf.expand_dims(tiled_zeros, axis=1)
        self.enc_outputs = tf.concat_v2([expanded_tiled_zeros, self.enc_outputs], axis=1)

    with tf.variable_scope("dencoder"):
      #self.first_decoder_input = \
      #    trainable_initial_state(batch_size, self.hidden_dim, name="first_decoder_input")

      #self.dec_inputs = tf.placeholder(tf.float32,
      #    [None, self.max_dec_length, self.input_dim], name="dec_inputs")
      #transformed_dec_inputs = \
      #    tf.nn.conv1d(dec_inputs_without_first, input_weight, 1, "VALID")

      self.dec_seq_length = tf.placeholder(
          tf.int32, [None], name="dec_seq_length")
      self.dec_idx_inputs = tf.placeholder(tf.int32,
          [None, self.max_dec_length], name="dec_inputs")

      idx_pairs = index_matrix_to_pairs(self.dec_idx_inputs)
      self.dec_inputs = tf.gather_nd(self.enc_inputs, idx_pairs)
      self.transformed_dec_inputs = \
          tf.gather_nd(self.transformed_enc_inputs, idx_pairs)

      #dec_inputs = [
      #    tf.expand_dims(self.first_decoder_input, 1),
      #    dec_inputs_without_first,
      #]
      #self.dec_inputs = tf.concat_v2(dec_inputs, axis=1)

      if self.use_terminal_symbol:
        dec_target_dims = [None, self.max_enc_length + 1]
      else:
        dec_target_dims = [None, self.max_enc_length]

      self.dec_targets = tf.placeholder(
          tf.int32, dec_target_dims, name="dec_targets")
      self.is_train = tf.placeholder(tf.bool, name="is_train")

      self.dec_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.dec_cell] * self.num_layers
        self.dec_cell = MultiRNNCell(cells)

      self.dec_output_logits, self.dec_states, _ = decoder_rnn(
          self.dec_cell, self.transformed_dec_inputs, 
          self.enc_outputs, self.enc_final_states,
          self.enc_seq_length, self.hidden_dim, self.num_glimpse,
          self.max_dec_length, batch_size, is_train=True,
          initializer=self.initializer)

    with tf.variable_scope("dencoder", reuse=True):
      self.dec_outputs, _, self.predictions = decoder_rnn(
          self.dec_cell, self.transformed_dec_inputs,
          self.enc_outputs, self.enc_final_states,
          self.enc_seq_length, self.hidden_dim, self.num_glimpse,
          self.max_dec_length, batch_size, is_train=False,
          initializer=self.initializer)

  def _build_optim(self):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.dec_targets, logits=self.dec_output_logits)

    weights = tf.ones(input_length, dtype=tf.int32)
    batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                        tf.reduce_sum(weights),
                        name="batch_loss")

    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()

    tf.summary.scalar("losses/batch_loss", batch_loss)
    tf.summary.scalar("losses/total_loss", total_loss)

    # TODO: length masking
    #mask = tf.sign(tf.to_float(targets_flat))
    #masked_losses = mask * self.loss

    self.lr = tf.train.exponential_decay(
        self.lr_start, self.global_step, self.lr_decay_step,
        self.lr_decay_rate, staircase=True, name="learning_rate")

    optimizer = tf.train.AdamOptimizer(self.lr)

    if self.max_grad_norm != None:
      grads_and_vars = optimizer.compute_gradients(self.loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars)
    else:
      self.optim = optimizer.minimize(self.loss)
