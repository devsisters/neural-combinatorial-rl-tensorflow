import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from utils import show_all_variables
from data_loader import TSPDataLoader

class Trainer(object):
  def __init__(self, config, rng):
    self.config = config
    self.rng = rng

    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step
    self.checkpoint_secs = config.checkpoint_secs

    self.summary_ops = {}

    if config.task.lower().startswith('tsp'):
      self.data_loader = TSPDataLoader(config, rng=self.rng)
    else:
      raise Exception("[!] Unknown task: {}".format(config.task))

    self.models = {}

    reuse = False
    for name in ['train', 'valid', 'test']:
      tf.logging.info("Create a [{}] model..".format(name))
      vs = tf.get_variable_scope()

      with tf.variable_scope(vs, reuse=reuse):
        self.models[name] = Model(
            config,
            inputs=self.data_loader.x[name],
            labels=self.data_loader.y[name],
            enc_seq_length=self.data_loader.seq_length[name],
            dec_seq_length=self.data_loader.seq_length[name],
            reuse=reuse)
      reuse = True

    self._build_session()

    show_all_variables()

  def _build_session(self):
    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(self.model_dir)

    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.global_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    tf.logging.info("Training starts...")

    self.data_loader.run_input_queue(self.sess)

  def test(self):
    tf.logging.info("Testing starts...")

  def _inject_summary(self, tag, feed_dict, step):
    summaries = self.sess.run(self.summary_ops[tag], feed_dict)
    self.summary_writer.add_summary(summaries['summary'], step)

    path = os.path.join(
        self.config.sample_model_dir, "{}.png".format(step))
    imwrite(path, img_tile(summaries['output'],
            tile_shape=self.config.sample_image_grid)[:,:,0])

  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None
