#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
  return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_dim', type=int, default=200, help='')
net_arg.add_argument('--num_layers', type=int, default=2, help='')
net_arg.add_argument('--max_length', type=int, default=20, help='')
net_arg.add_argument('--input_dim', type=int, default=2, help='')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--task', type=str, default='tps20')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--optimizer', type=str, default='rmsprop', help='')
train_arg.add_argument('--max_step', type=int, default=10000, help='')
train_arg.add_argument('--reg_scale', type=float, default=0.5, help='')
train_arg.add_argument('--batch_size', type=int, default=512, help='')
train_arg.add_argument('--learning_rate', type=float, default=0.001, help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=50, help='')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--sample_dir', type=str, default='samples')
misc_arg.add_argument('--output_dir', type=str, default='outputs')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--debug', type=str2bool, default=False)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--random_seed', type=int, default=123, help='')

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed
