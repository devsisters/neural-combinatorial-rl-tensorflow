import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *
from utils import show_all_variables

class Model(object):
  def __init__(self, config, data_loader):
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

    self.use_terminal_symbol = config.use_terminal_symbol

    self.reg_scale = config.reg_scale
    self.lr_start = config.lr_start
    self.max_grad_norm = config.max_grad_norm
    self.batch_size = config.batch_size

    self.layer_dict = {}

    with arg_scope([linear, LSTMCell], \
        initializer=tf.random_normal_initializer(0, 0.001)):
      self._build_model()

    self._build_optim()

    show_all_variables()

  def _build_model(self):
    self.global_step = tf.Variable(0, trainable=False)

    initializer = None
    input_weight = tf.get_variable(
        "input_weight", [1, self.input_dim, self.hidden_dim])

    with tf.variable_scope("encoder"):
      self.enc_seq_length = tf.placeholder(
          tf.int32, [None], name="enc_seq_length")
      self.enc_inputs = tf.placeholder(
          tf.float32, [None, self.max_enc_length, self.input_dim],
          name="enc_inputs")
      self.transformed_enc_inputs = tf.nn.conv1d(
          self.enc_inputs, input_weight, 1, "VALID")

    batch_size = tf.shape(self.enc_inputs)[0]
    tiled_zeros = tf.tile(tf.zeros(
      [1, self.hidden_dim]), [batch_size, 1], name="tiled_zeros")

    with tf.variable_scope("encoder"):
      self.enc_cell = LSTMCell(self.hidden_dim)
      if self.num_layers > 1:
        cells = [self.enc_cell] * self.num_layers
        self.enc_cell = MultiRNNCell(cells)
      self.enc_init_state = trainable_initial_state(batch_size, self.enc_cell.state_size)

      # self.encoder_outputs : [None, max_time, output_size]
      self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
          self.enc_cell, self.transformed_enc_inputs, self.enc_seq_length, self.enc_init_state)

      if self.use_terminal_symbol:
        self.enc_outputs = [tiled_zeros] + self.enc_outputs

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
      self.transformed_dec_inputs = tf.gather_nd(self.transformed_enc_inputs, idx_pairs)

      #dec_inputs = [
      #    tf.expand_dims(self.first_decoder_input, 1),
      #    dec_inputs_without_first,
      #]
      #self.dec_inputs = tf.concat_v2(dec_inputs, axis=1)

      self.dec_targets = tf.placeholder(tf.float32,
          [None, self.max_enc_length + 1], name="dec_targets")
      self.is_train = tf.placeholder(tf.bool, name="is_train")

      self.dec_cell = LSTMCell(self.hidden_dim)
      if self.num_layers > 1:
        cells = [self.dec_cell] * self.num_layers
        self.dec_cell = MultiRNNCell(cells)

      self.dec_output_logits, self.dec_states, _ = decoder_rnn(
          self.dec_cell, self.transformed_dec_inputs, 
          self.enc_outputs, self.enc_final_states,
          self.enc_seq_length, self.hidden_dim, self.num_glimpse,
          self.max_dec_length, batch_size, is_train=True)

    with tf.variable_scope("dencoder", reuse=True):
      self.dec_outputs, _, self.predictions = decoder_rnn(
          self.dec_cell, self.transformed_dec_inputs,
          self.enc_outputs, self.enc_final_states,
          self.enc_seq_length, self.hidden_dim, self.num_glimpse,
          self.max_dec_length, batch_size, is_train=False)

  def _build_optim(self):
    self.loss = tf.reduce_mean(self.output - self.targets)

    self.lr = tf.Variable(self.lr_start)
    optimizer = tf.train.AdamOptimizer(self.lr)

    if self.max_grad_norm != None:
      grads_and_vars = optimizer.compute_gradients(self.loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars)
    else:
      self.optim = optimizer.minimize(self.loss)
