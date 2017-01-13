# Most of the codes are from https://github.com/vshallc/PtrNets/blob/master/pointer/misc/tsp.py
import os
import itertools
import numpy as np
from tqdm import trange

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))

# https://gist.github.com/mlalevic/6222750
def solve_tsp_dynamic(points):
  #calc all lengths
  all_distances = [[length(x,y) for y in points] for x in points]
  #initial value - just distance from 0 to every other point + keep the track of edges
  A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
  cnt = len(points)
  for m in range(2, cnt):
    B = {}
    for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
      for j in S - {0}:
        B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
    A = B
  res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
  return np.asarray(res[1]) + 1

def generate_one_example(n_nodes):
  nodes = np.random.rand(n_nodes, 2)
  res = solve_tsp_dynamic(nodes)
  return nodes, res

def generate_examples(num, n_min, n_max, desc=""):
  examples = []
  for i in trange(num, desc=desc):
    n_nodes = np.random.randint(n_min, n_max + 1)
    nodes, res = generate_one_example(n_nodes)
    examples.append((nodes, res))
  return examples

class TSPDataLoader(object):
  def __init__(self, config, rng=None):
    self.config = config
    self.rng = rng

    self.task = config.task
    self.min_length = config.min_data_length
    self.max_length = config.max_data_length

    self.task_name = "{}_{}_{}".format(self.task, self.min_length, self.max_length)
    self.npz_path = os.path.join(config.data_dir, "{}.npz".format(self.task_name))

    self._maybe_generate_and_save()

  def _maybe_generate_and_save(self):
    if not os.path.exists(self.npz_path):
      print("[*] Creating dataset for {}".format(self.task))

      train = generate_examples(
          1000000, self.min_length, self.max_length, "Train data..")
      valid = generate_examples(
          1000, self.min_length, self.max_length, "Valid data..")
      test = generate_examples(
          1000, self.max_length, self.max_length, "Test data..")

      np.savez(self.npz_path, train=train, test=test, valid=valid)
    else:
      print("[*] Loading dataset for {}".format(self.task))
      data = np.load(self.npz_path)
      self.train, self.test, self.valid = \
          data['train'], data['test'], data['valid']
