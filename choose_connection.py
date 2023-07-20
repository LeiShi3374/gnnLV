import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np

real_node_coords_al = np.load('data/beamData/rawData/train/real-node-coords.npy')
real_node_coords = real_node_coords_al[0]

tree = KDTree(real_node_coords)

# Find indices and distances of nearest neighbors
k = 4  # Number of nearest neighbors
distances, indices = tree.query(real_node_coords, k=k+1)  # +1 to include self

# Reshape the indices array for easier processing
indices = indices[:, 1:].reshape(-1)

# Create connections using vectorized operations
points_count = len(real_node_coords)
connections = np.column_stack((np.arange(points_count).repeat(k), indices))

# Filter out self-connections
connections = connections[connections[:, 0] != connections[:, 1]]

# Ensure symmetry in connections
reciprocal_connections = np.column_stack((connections[:, 1], connections[:, 0]))
connections = np.concatenate((connections, reciprocal_connections))

# Remove duplicate connections
connections = np.unique(connections, axis=0)

dense_flag = False
if dense_flag:
    dense_connections = np.zeros((154, 2))
    k = 0
    for i in range(7):
        for j in range(11):
            connection_tm = np.array([i * 12 + j + 0, i * 12 + j+13])
            dense_connections[k] = connection_tm
            k = k + 1
            connection_tm = np.array([i * 12 + j + 1, i * 12 + j + 12])
            dense_connections[k] = connection_tm
            k = k + 1
    dense_connections = dense_connections.astype(int)
    connections = np.concatenate((connections, dense_connections))

# Center Method
center_flag = False
if center_flag:
    center_point = np.mean(real_node_coords, axis=0)
    distances_to_center_point = np.sqrt(np.sum((real_node_coords - center_point) ** 2, axis=1))
    closest_index_to_center = np.argmin(distances_to_center_point)
    leaf_center_indices = sorted(
        np.random.choice(np.linspace(0, real_node_coords.shape[0] - 1, real_node_coords.shape[0], dtype=int),
                         size=int(np.round(1 * real_node_coords.shape[0])), replace=False))
    leaf_center_indices = [x for x in leaf_center_indices if x != closest_index_to_center]
    root_center_indices = np.full_like(leaf_center_indices, closest_index_to_center)
    sparce_center_topology = np.column_stack((leaf_center_indices, root_center_indices))
    connections = np.vstack((connections, sparce_center_topology))

# Create scatter plot and plot connections
plt.scatter(real_node_coords[:, 0], real_node_coords[:, 1], color='blue')
for connection in connections:
    node1 = real_node_coords[connection[0]]
    node2 = real_node_coords[connection[1]]
    plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='black')

# connections_or = np.load('data/beamData/topologyData/real-node-topology.npy')
# for connection in connections_or:
#     node1 = real_node_coords[connection[0]]
#     node2 = real_node_coords[connection[1]]
#     plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='black')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Nodes with Connection Lines')
plt.axis('equal')

plt.show()