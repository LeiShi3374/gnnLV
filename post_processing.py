import numpy as np
import pyvista as pv
from absl import logging
import data_utils as data_utils
from main import *

def create_vtu_tri(data_path, K, n_shape_coeff, n_epochs, lr, trained_params_dir, fixed_geom, dir_label, case_number):

    logging.info('Beginning Evaluation All')
    logging.info(f'Data path: {data_path}')
    logging.info(f'Message passing steps (K): {K}')
    logging.info(f'Num. shape coeffs: {n_shape_coeff}')
    logging.info(f'Training epochs: {n_epochs}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Trained Params Dir: {trained_params_dir}')
    logging.info(f'Fixed LV geom: {fixed_geom}\n')

    # load train data
    train_data = data_utils.DataLoader(data_path, 'train', n_shape_coeff, fixed_geom)
    train_data._displacement = train_data._displacement * train_data._displacement_std + train_data._displacement_mean

    receivers = train_data._receivers
    senders   = train_data._senders
    node_coords = train_data._real_node_coords[case_number, :, :]

    receivers_real = receivers[(receivers < node_coords.shape[0]) & (senders < node_coords.shape[0])]
    senders_real = senders[(receivers < node_coords.shape[0]) & (senders < node_coords.shape[0])]
    sparse_topology_real = np.concatenate((receivers_real.reshape(-1,1), senders_real.reshape(-1,1)), axis = 1)

    # create dictionary of hyperparameters of the GNN emulator
    config_dict = create_config_dict(K, n_shape_coeff, n_epochs, lr, train_data._output_dim)

    # create directory to store the trained parameters of the network
    results_save_dir = create_savedir(data_path, K, n_shape_coeff, n_epochs, lr, dir_label)
    logging.info(f'Results save directory: {results_save_dir}\n')

    # if trained_params_dir is not set, parameters are read from results_save_dir
    if trained_params_dir == "None": trained_params_dir = results_save_dir

    # initialise GNN emulator and read trained network parameters
    model, trained_params = utils.initialise_emulator(config_dict, train_data, results_save_dir, fixed_geom,
                                                      trained_params_dir)

    # select the emulator predict function based on whether we consider fixed or varying LV geometry data
    emul_pred_fn = select_pred_fn(fixed_geom)

    # jit prediction for faster execution
    prediction_fn_jit = jit(functools.partial(emul_pred_fn,
                                              net=model,
                                              params=trained_params,
                                              Umean=train_data._displacement_mean,
                                              Ustd=train_data._displacement_std))

    logging.info('Predicting on test data set using trained emulator')
    displacement_pred_all = predict_dataset(train_data, prediction_fn_jit)

    element_real = []
    for i, seg0 in enumerate(sparse_topology_real):
        for j, seg1 in enumerate(sparse_topology_real[i + 1:, :]):
            if seg1[0] == seg0[0]:
                for k, seg2 in enumerate(sparse_topology_real[i + 1:, :]):
                    if all(seg2 != seg0):
                        if ((seg2[0] == seg0[1]) & (seg2[1] == seg1[1])) or (
                                (seg2[1] == seg0[1]) & (seg2[0] == seg1[1])):
                            elem = np.array([seg0[0], seg0[1], seg1[1]])
                            element_real.append(elem.reshape(1, -1))
            if seg1[1] == seg0[0]:
                for k, seg2 in enumerate(sparse_topology_real[i + 1:, :]):
                    if (seg2 != seg0).all():
                        if ((seg2[0] == seg0[1]) & (seg2[1] == seg1[0])) or (
                                (seg2[1] == seg0[1]) & (seg2[0] == seg1[0])):
                            elem = np.array([seg0[0], seg0[1], seg1[0]])
                            element_real.append(elem.reshape(1, -1))

    element_real = np.concatenate(element_real, axis=0)

    element_real_final = (np.concatenate((np.ones((len(element_real), 1)) * 3, element_real), axis = 1)).astype(int)

    node_coords = np.concatenate((node_coords, np.zeros((len(node_coords), 1))), axis = 1)

    result_mesh = pv.PolyData(node_coords, element_real_final)

    displacement_orig = train_data._displacement[case_number, :, :]
    displacement_pred = displacement_pred_all[case_number, :, :]

    displacement_orig = np.concatenate((displacement_orig, np.zeros((len(displacement_orig), 1))), axis = 1)
    displacement_pred = np.concatenate((displacement_pred, np.zeros((len(displacement_pred), 1))), axis=1)

    result_mesh.point_data['displacement_orig'] = displacement_orig

    result_mesh.point_data['displacement_pred'] = displacement_pred

    result_mesh.save('result.vtp')

    result_mesh.points =  node_coords + displacement_orig

    result_mesh.save('result_wrapped_orig.vtp')

    result_mesh.points =  node_coords + displacement_pred

    result_mesh.save('result_wrapped_pred.vtp')

    print()

K = 2
data_path = 'beamData'
dir_label = ''
fixed_geom = False
lr = 5e-05
n_epochs = 300
n_shape_coeff = 2
trained_params_dir = 'None'
case_number = 16


create_vtu_tri(data_path, K, n_shape_coeff, n_epochs, lr, trained_params_dir, fixed_geom, dir_label, case_number)

# Set up node coordinates and connectivity
# node_coords = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
# connectivity = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
#
# # Create PolyData object
# mesh = pv.PolyData(node_coords, connectivity)
#
# mesh.save('result.vtp')