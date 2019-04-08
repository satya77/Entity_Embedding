#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import threading
from io import open
from multiprocessing.pool import ThreadPool

import math
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import time


logger = logging.getLogger("deepwalk")


__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"


# def timing(f):
#     def wrap(*args,**kwargs):
#         time1 = time.time()
#         ret = f(*args,**kwargs)
#         time2 = time.time()
#         print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
#
#         return ret
#     return wrap
import bisect
import random

# class WeightedRandomGenerator(object):
#     def __init__(self, weights):
#         self.totals = []
#         running_total = 0
#
#         for w in weights:
#             running_total += w
#             self.totals.append(running_total)
#
#     def next(self):
#         rnd = random.random() * self.totals[-1]
#         return bisect.bisect_right(self.totals, rnd)
#
#     def __call__(self):
#         return self.next()
class WeightedRandomGenerator(object):
    def __init__(self, totals):
        self.totals = totals

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()
class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

  def __init__(self):
    super(Graph, self).__init__(list)
    self.probs = {}

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()

    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]

    return subgraph

  def make_undirected(self):

    t0 = time.time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)

    t1 = time.time()
    print('make_directed: added missing edges {}s'.format(t1 - t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time.time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))

    t1 = time.time()
    print('make_consistent: made consistent in {}s'.format(t1 - t0))

    # self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time.time()

    for x in self:
      if x in self[x]:
        self[x].remove(x)
        removed += 1

    t1 = time.time()

    print('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True

    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v: len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()]) / 2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order()

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    # time1 = time.time()
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.keys()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        # if rand.random() >= alpha:
        nodes, probs = G.probs[cur]
        path.append(nodes[WeightedRandomGenerator(probs).next()])
        # path.append(nodes[np.random.multinomial(1,probs).argmax()])-> a bit faster
          #path.append(np.random.choice(nodes, p=probs))-> slow
        # else:
        #   path.append(path[0])
      else:
        break
    # time2 = time.time()
    # print("Thread" + str(threading.get_ident()) + " took :{:.3f} ms".format((time2 - time1) * 1000.0))
    return [str(node) for node in path]


# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0),workers=6):
  walks = []

  nodes = list(G.nodes())

  with ThreadPool(workers) as pool:
    results = []
    for cnt in range(num_paths):
      rand.shuffle(nodes)
      results.append(pool.apply_async(random_Walk_for_each_node, (G, path_length, nodes, alpha, rand)))
    for async_result in results:
      try:
        walks.extend(async_result.get())
      except ValueError as e:
        print(e)

  # for cnt in range(num_paths):
  #   rand.shuffle(nodes)
  #   walks.append(random_Walk_for_each_node(G, path_length, nodes, alpha, rand))

  return walks

def random_Walk_for_each_node(G, path_length,nodes,  alpha=0,
                          rand=random.Random(0)):
    time1 = time.time()
    result=[]
    counter=0
    for node in nodes:
        counter=counter+1
        result.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
        if counter%50000==0:
            print("Thread" +str(threading.current_thread())+ " is on node :"+str(counter))
    time2 = time.time()
    print('{:s} function took {:.3f} ms'.format("all nodes: ", (time2 - time1) * 1000.0))
    print("Thread" + str( threading.current_thread()) + " if finished")

    return result


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
  "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def load_edgelist_weighted(file_,proximity="Plain", undirected=True):
  G = Graph()
  all_nodes=set()
  with open(file_) as f:
    for l in f:
      x, y, w = l.strip().split()[:3]
      x = int(x)
      y = int(y)
      w = float(w)
      all_nodes.add(x)
      all_nodes.add(y)
      G[x].append((y, w))
      if undirected:
        G[y].append((x, w))
  print(len(all_nodes))
  for cur in G.keys():
    # total = sum(n for _, n in G[cur])
    nodes = []
    probs = []
    totals=[]
    running_total = 0

    for node, weight in G[cur]:
      nodes.append(node)
      if proximity == "Plain":
        running_total += weight
      elif proximity == "log":
        running_total += math.log(weight + 1)
      elif proximity == "sqrt":
        running_total += math.sqrt(weight)
      totals.append(running_total)
      # probs.append(weight)
      # probs.append(weight / total)
    # G.probs[cur] = (nodes, probs)
      G.probs[cur]=(nodes,totals)

  G.make_consistent()
  return G


def from_networkx(G_input, undirected=True):
  G = Graph()

  for idx, x in enumerate(G_input.nodes_iter()):
    for y in iterkeys(G_input[x]):
      G[x].append(y)

  if undirected:
    G.make_undirected()

  return G



