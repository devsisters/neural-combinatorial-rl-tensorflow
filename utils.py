import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf

def prepare_dirs(config):
  if config.load_path:
    config.model_name = "{}_{}".format(config.task, config.load_path)
  else:
    config.model_name = "{}_{}".format(config.task, get_time())
  config.model_dir = os.path.join(config.log_dir, config.model_name)

  for path in [config.log_dir]:
    if not os.path.exists(path):
      os.makedirs(path)

def get_time():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def show_all_variables():
  print("")
  total_count = 0
  for idx, op in enumerate(tf.trainable_variables()):
    shape = op.get_shape()
    count = np.prod(shape)
    print("[%2d] %s %s = %s" % (idx, op.name, shape, "{:,}".format(int(count))))
    total_count += int(count)
  print("=" * 40)
  print("[Total] variable size: %s" % "{:,}".format(total_count))
  print("=" * 40)
  print("")

def save_config(model_dir, config):
  param_path = os.path.join(model_dir, "params.json")

  print("[*] MODEL dir: %s" % model_dir)
  print("[*] PARAM path: %s" % param_path)

  with open(param_path, 'w') as fp:
    json.dump(config.__dict__, fp,  indent=4, sort_keys=True)
