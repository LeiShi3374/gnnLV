import numpy as np
from matplotlib import pyplot as plt

## BeamData
# real_node_displacement_train_proc = np.load('data/beamData/processedData/train/real-node-displacement.npy')
# real_node_displacement_test_proc = np.load('data/beamData/processedData/test/real-node-displacement.npy')
# real_node_displacement_validation_proc = np.load('data/beamData/processedData/validation/real-node-displacement.npy')
#
# train_proc_nodes = real_node_displacement_train_proc.sum(axis=0).sum(axis=0) / real_node_displacement_train_proc.shape[0] / real_node_displacement_train_proc.shape[1]
# test_proc_nodes = real_node_displacement_test_proc.sum(axis=0).sum(axis=0) / real_node_displacement_test_proc.shape[0] / real_node_displacement_test_proc.shape[1]
# validation_proc_nodes = real_node_displacement_validation_proc.sum(axis=0).sum(axis=0) / real_node_displacement_validation_proc.shape[0] / real_node_displacement_validation_proc.shape[1]
#
# real_node_displacement_train_raw = np.load('data/beamData/rawData/train/real-node-displacement.npy')
# real_node_displacement_test_raw = np.load('data/beamData/rawData/test/real-node-displacement.npy')
# real_node_displacement_validation_raw = np.load('data/beamData/rawData/validation/real-node-displacement.npy')
#
# train_raw_nodes = real_node_displacement_train_raw.sum(axis=0).sum(axis=0) / real_node_displacement_train_raw.shape[0] / real_node_displacement_train_raw.shape[1]
# test_raw_nodes = real_node_displacement_test_raw.sum(axis=0).sum(axis=0) / real_node_displacement_test_raw.shape[0] / real_node_displacement_test_raw.shape[1]
# validation_raw_nodes = real_node_displacement_validation_raw.sum(axis=0).sum(axis=0) / real_node_displacement_validation_raw.shape[0] / real_node_displacement_validation_raw.shape[1]


## Process vs Raw Test Data
# augmented_node_coords_test_proc = np.load('data/beamData/processedData/test/augmented-node-coords.npy')
# augmented_node_features_test_proc = np.load('data/beamData/processedData/test/augmented-node-features.npy')
# edge_features_test_proc = np.load('data/beamData/processedData/test/edge-features.npy')
# global_features_test_proc = np.load('data/beamData/processedData/test/global-features.npy')
# real_node_coords_test_proc = np.load('data/beamData/processedData/test/real-node-coords.npy')
# real_node_displacement_test_proc = np.load('data/beamData/processedData/test/real-node-displacement.npy')
# # real_node_features_test_proc = np.load('data/beamData/processedData/test/real-node-features.npy')
# shape_coeffs_test_proc = np.load('data/beamData/processedData/test/shape-coeffs.npy')
#
# augmented_node_coords_test_raw = np.load('data/beamData/rawData/test/augmented-node-coords.npy')
# augmented_node_features_test_raw = np.load('data/beamData/rawData/test/augmented-node-features.npy')
# edge_features_test_raw = np.load('data/beamData/rawData/test/edge-features.npy')
# global_features_test_raw = np.load('data/beamData/rawData/test/global-features.npy')
# real_node_coords_test_raw = np.load('data/beamData/rawData/test/real-node-coords.npy')
# real_node_displacement_test_raw = np.load('data/beamData/rawData/test/real-node-displacement.npy')
# real_node_features_test_raw = np.load('data/beamData/rawData/test/real-node-features.npy')
# shape_coeffs_test_raw = np.load('data/beamData/rawData/test/shape-coeffs.npy')

## Process vs Raw Train Data
# augmented_node_coords_train_proc = np.load('data/beamData/processedData/train/augmented-node-coords.npy')
# augmented_node_features_train_proc = np.load('data/beamData/processedData/train/augmented-node-features.npy')
# edge_features_train_proc = np.load('data/beamData/processedData/train/edge-features.npy')
# global_features_train_proc = np.load('data/beamData/processedData/train/global-features.npy')
# real_node_coords_train_proc = np.load('data/beamData/processedData/train/real-node-coords.npy')
# real_node_displacement_train_proc = np.load('data/beamData/processedData/train/real-node-displacement.npy')
# shape_coeffs_train_proc = np.load('data/beamData/processedData/train/shape-coeffs.npy')

augmented_node_coords_train_raw = np.load('data/beamData/rawData/train/augmented-node-coords.npy')
augmented_node_features_train_raw = np.load('data/beamData/rawData/train/augmented-node-features.npy')
edge_feature_train_raw = np.load('data/beamData/rawData/train/edge-features.npy')
global_features_train_raw = np.load('data/beamData/rawData/train/global-features.npy')
real_node_coords_train_raw = np.load('data/beamData/rawData/train/real-node-coords.npy')
real_node_displacement_train_raw = np.load('data/beamData/rawData/train/real-node-displacement.npy')
real_node_features_train_raw = np.load('data/beamData/rawData/train/real-node-features.npy')
shape_coeffs_train_raw = np.load('data/beamData/rawData/train/shape-coeffs.npy')
node_layer_labels = np.load('data/beamData/topologyData/node-layer-labels.npy')
sparse_topology = np.load('data/beamData/topologyData/sparse-topology.npy')
augmented_topology = np.load('data/beamData/topologyData/augmented-topology.npy')


np.savetxt("sparse_topology.csv", sparse_topology, fmt='%f', delimiter=",")



print()


#
# sparse_topology = np.load('data/beamData/topologyData/sparse-topology.npy')
# representative_nodes = np.load('data/beamData/topologyData/representative-nodes.npy')
# representative_augmented_nodes = np.load('data/beamData/topologyData/representative-augmented-nodes.npy')
# augmented_topology = np.load('data/beamData/topologyData/augmented-topology.npy')
# kmeans_labels_list = np.load('data/beamData/topologyData/kmeans-labels-list.npy', allow_pickle=True)
# node_layer_labels = np.load('data/beamData/topologyData/node-layer-labels.npy')
# real_node_topology = np.load('data/beamData/topologyData/real-node-topology.npy')
# kmeans_labels_list1 = kmeans_labels_list[0]
# kmeans_labels_list2 = kmeans_labels_list[1]
#
# # kmeans_labels_list1.tofile('bgttt.csv', sep = ',')
# np.savetxt("bgttt.csv", kmeans_labels_list2, fmt='%d', delimiter=",")
# # plt.scatter(real_node_coords[0, :, 0], real_node_coords[0, :, 1])
# # plt.show()

## lvData

# real_node_topology = np.load('data/lvData/topologyData/real-node-topology.npy')
# real_node_coords = np.load('data/lvData/rawData/test/real-node-coords.npy')
# representative_nodes = np.load('data/lvData/topologyData/representative-nodes.npy')
# np.savetxt("LVreal_node_topology.csv", real_node_topology, fmt='%d', delimiter=",")
# np.savetxt("LVrepresentative_nodes.csv", representative_nodes, fmt='%f', delimiter=",")
# shape_coeffs = np.load('data/lvData/rawData/test/shape-coeffs.npy')
# vol = np.load('data/lvData/rawData/test/vol.npy')
# real_node_features = np.load('data/lvData/rawData/test/real-node-features.npy')
# real_node_displacement = np.load('data/lvData/rawData/test/real-node-displacement.npy')
# global_features = np.load('data/lvData/rawData/test/global-features.npy')
# edge_features = np.load('data/lvData/rawData/test/edge-features.npy')
# np.savetxt("LVreal_node_topology.csv", real_node_topology, fmt='%f', delimiter=",")
# np.savetxt("LVreal_node_features.csv", real_node_features[0], fmt='%f', delimiter=",")
# kmeans_labels_lists = np.load('data/lvData/topologyData/kmeans-labels-list.npy', allow_pickle=True)
# kmeans_labels_lists_1 = kmeans_labels_lists[0]
# kmeans_labels_lists_2 = kmeans_labels_lists[1]
# kmeans_labels_lists_3 = kmeans_labels_lists[2]










print()