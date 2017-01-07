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

    self.input_height = config.input_height
    self.input_width = config.input_width
    self.input_channel = config.input_channel

    self.reg_scale = config.reg_scale
    self.learning_rate = config.learning_rate
    self.max_grad_norm = config.max_grad_norm
    self.batch_size = config.batch_size

    self.layer_dict = {}

    self._build_placeholders()
    self._build_model()
    self._build_optim()

    show_all_variables()

  def _build_placeholders(self):
    self.inputs = tf.placeholder(tf.float32, name="inputs")
    self.lengths = tf.placeholder(tf.float32, name="lengths")
    self.targets = tf.placeholder(tf.float32, name="targets")
    self.is_train = tf.placeholder(tf.bool, name="is_train")

  def _build_encoder(self):
    self.global_step = tf.Variable(0, trainable=False)

    with tf.variable_scope("encoder"):
      self.cell = tf.nn.rnn_cell.LSTMCell(size)
      if num_layers > 1:
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

      self.rnn = tf.nn.dynamic_rnn(self.cell, self.inputs, self.seq_length)

    with tf.variable_scope("dencoder"):
      self.cell = tf.nn.rnn_cell.LSTMCell(size)
      if num_layers > 1:
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

      self.rnn = tf.nn.dynamic_rnn(self.cell, self.inputs, self.seq_length)

  def _build_optim(self):
    self.loss = tf.reduce_mean(self.output - self.targets)

    self.learning_rate = tf.Variable(self.learning_rate)
    self.optim = tf.train.AdamOptimizer(self.learning_rate)
