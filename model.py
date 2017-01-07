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

    self.max_length = config.max_length
    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers

    self.reg_scale = config.reg_scale
    self.learning_rate = config.learning_rate
    self.max_grad_norm = config.max_grad_norm
    self.batch_size = config.batch_size

    self.layer_dict = {}

    self._build_model()
    self._build_optim()

    show_all_variables()

  def _build_model(self):
    self.global_step = tf.Variable(0, trainable=False)

    initializer = None

    with tf.variable_scope("encoder"):
      self.inputs = tf.placeholder(
          tf.float32, [None, self.max_length, self.input_dim], name="inputs")
      self.seq_length = tf.placeholder(
          tf.float32, [None], name="seq_length")

      self.cell = LSTMCell(self.hidden_dim)
      if self.num_layers > 1:
        cells = [self.cell] * self.num_layers
        self.cell = MultiRNNCell(cells)

      import ipdb; ipdb.set_trace() 

      self.enc_init_state = \
          get_tiled_state(self.cell, self.inputs, initializer)

      tiled_initial_state = (tiled_init_enc_h, tiled_init_enc_c)
      self.encoder_output = tf.nn.dynamic_rnn(
          self.cell, self.inputs, self.seq_length, tiled_initial_state)

    with tf.variable_scope("dencoder"):
      import ipdb; ipdb.set_trace() 
      self.first_decoder_input, tiled_decoder_input = \
          get_tiled_state(self.inputs, self.hidden_dim)

      self.targets = tf.placeholder(tf.float32, name="targets")
      self.is_train = tf.placeholder(tf.bool, name="is_train")

      self.initial_decoder_state, tiled_initial_state = \
          get_tiled_state(self.inputs, self.hidden_dim)

      self.cell = LSTMCell(self.hidden_dim)
      if self.num_layers > 1:
        cells = [self.cell] * self.num_layers
        self.cell = MultiRNNCell(cells)

      self.decoder_output = tf.nn.dynamic_rnn(
          self.cell, self.inputs, self.seq_length, tiled_initial_state)

  def _build_optim(self):
    self.loss = tf.reduce_mean(self.output - self.targets)

    self.learning_rate = tf.Variable(self.learning_rate)
    self.optim = tf.train.AdamOptimizer(self.learning_rate)
