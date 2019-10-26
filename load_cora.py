#!/usr/bin/python3

from os.path import exists;
from shutil import rmtree;
import wget;
import tarfile;
import numpy as np;
import tensorflow as tf;
from Model import dense_to_sparse;

def normalize(m):

  rowsum = tf.sparse.reduce_sum(m, axis = 1, output_is_sparse = True);
  r_inv = tf.sparse.SparseTensor(
    indices = rowsum.indices,
    values = tf.math.pow(rowsum.values, -1),
    dense_shape = rowsum.dense_shape);
  mask = tf.math.logical_not(tf.math.is_inf(r_inv.values));
  r_inv = tf.sparse.SparseTensor(
    indices=tf.boolean_mask(r_inv.indices, mask),
    values=tf.boolean_mask(r_inv.values, mask),
    dense_shape=r_inv.dense_shape);
  r_mat_inv = tf.sparse.SparseTensor(
    indices=tf.concat([r_inv.indices, r_inv.indices], axis=-1),
    values=r_inv.values,
    dense_shape=tf.concat([r_inv.dense_shape, r_inv.dense_shape], axis=0));
  return dense_to_sparse(tf.sparse.sparse_dense_matmul(r_mat_inv, tf.sparse.to_dense(m)));

def load_cora():

  # download cora dataset
  if False == exists('cora.tgz'):
    filename = wget.download('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz', out = '.');
  else:
    filename = "cora.tgz";
  # unzip the dataset
  if exists('cora'): rmtree('cora');
  tar = tarfile.open(filename);
  subdir_and_files = [tarinfo for tarinfo in tar.getmembers()];
  tar.extractall('.',subdir_and_files);
  # convert to dataset
  idx_features_labels = np.genfromtxt("cora/cora.content", dtype = np.dtype(str));
  # 1)features.shape = (1, N, Din)
  features = dense_to_sparse(idx_features_labels[:,1:-1].astype('float32'));
  features = normalize(features);
  features = tf.sparse.expand_dims(features, axis = 0);
  # 2)labels.shape = (1, N, C)
  labels = idx_features_labels[:,-1];
  classes = set(labels);
  classes_dict = {c: tf.eye(len(classes))[i, :] for i, c in enumerate(classes)};
  labels = tf.constant(np.array(list(map(classes_dict.get, labels)), dtype = np.int32));
  labels = tf.expand_dims(labels, axis = 0);
  # 3)adjacent matrix.shape = (N,N)
  idx = np.array(idx_features_labels[:, 0], dtype = np.int32);
  idx_map = {j: i for i, j in enumerate(idx)};
  edges_unordered = np.genfromtxt("cora/cora.cites", dtype = np.int32);
  edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype = np.int32).reshape(edges_unordered.shape);
  adj = tf.sparse.reorder(tf.sparse.SparseTensor(indices = edges, values = tf.ones(edges.shape[0]), dense_shape = [labels.shape[1], labels.shape[1]]));
  adjT = tf.sparse.transpose(adj);
  mask = tf.cast(tf.math.greater(tf.sparse.to_dense(adjT), tf.sparse.to_dense(adj)), dtype = tf.float32);
  adj = tf.sparse.add(tf.sparse.add(adj,adjT.__mul__(mask)),adj.__mul__(mask).__mul__(-1));
  adj = tf.sparse.add(adj, tf.sparse.eye(adj.shape[0]));
  adj = normalize(adj);
  # generate dataset
  return features, labels, adj;

if __name__ == "__main__":
    
  assert True == tf.executing_eagerly();
  load_cora();
