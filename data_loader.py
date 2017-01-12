import os
import numpy as np

class TSPDataLoader(object):
  def __init__(self, config, rng=None):
    self.config = config
    self.rng = rng

    self.task = config.task
    self.min_length = config.min_data_length
    self.max_length = config.max_data_length

    self.task_name = "{}_{}_{}".format(self.task, self.min_length, self.max_length)

    self.npz_path = os.path.join(config.data_dir, "{}.npz".format(self.task_name))

  def maybe_create_and_save(self):
    base_path = os.path.join(data_path, '{}/Data/Normalized'.format(config.real_image_dir))
    npz_path = os.path.join(data_path, DATA_FNAME)

  def generate_samples(self):
    pass
