import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import transforms.coordinate_transforms
import random

from keras.models import model_from_json
from keras.applications.imagenet_utils import preprocess_input

from CNN_train_package.utils.gt_fns import give_gt_cart_reg
from CNN_train_package.fast_image_rotation import rectify_image_given_phi_theta
from CNN_train_package.utils.data_augment import moderate_combo

pathToMatrices = '/home/ben/projects/comparison_leveller/data/matrices221_10000'
phiTheta = np.load(pathToMatrices + '/phiTheta_10000.npy')
num_points = phiTheta.shape[0]


def forward_spherical(estimator, image):
    # input: An instance of the CNN and image
    # output: phi (=elevation), theta (=elevation) for specific convention please refer to spherical_to_cartesian in fast_image_rotation.py
    # to estimate the up-vector direction given the instance of the network
    # for: (spherical angles and regression)
    densenet_input_shape = (221, 442, 3)
    inputH, inputW = densenet_input_shape[0], densenet_input_shape[1]

    batch_x = np.zeros((1,) + densenet_input_shape, dtype='float32')
    HEIGHT, WIDTH, CHANNEL = image.shape

    if HEIGHT != inputH or WIDTH != inputW:
        resized_image = cv2.resize(image, (inputW, inputH), interpolation=cv2.INTER_CUBIC)
    else:
        resized_image = image

    batch_x[0] = resized_image

    output = estimator.predict(batch_x)[0]
    phi_output = np.asscalar(output[0])
    theta_output = np.asscalar(output[1])
    z_axis = give_gt_cart_reg(phi_output, theta_output)

    return phi_output, theta_output, z_axis


def build_model(path_to_network_model, path_to_weights):
    """
        path_to_network_model = 'path to network model (model.json)'
        path_to_weight = 'path to network weight weight_xx_xx.h5'
    """

    # with tf.device('/gpu:0'):
    json_file = open(path_to_network_model, 'r')
    model_json = json_file.read()
    json_file.close()
    # custom_objects={"backend": K, "tf": tf}
    model = model_from_json(model_json, custom_objects={"tf": tf})
    model.compile(
        loss='logcosh',
        optimizer='adam')
    model.load_weights(path_to_weights)
    return model


def get_random_rotation(s):
    rotationIdx = np.random.randint(0, num_points)
    xMapName = pathToMatrices + '/' + str("%05d" % rotationIdx) + "_x.npy"
    yMapName = pathToMatrices + '/' + str("%05d" % rotationIdx) + "_y.npy"
    xMap = np.load(xMapName)
    yMap = np.load(yMapName)
    phiGT = phiTheta[rotationIdx, 0]
    thetaGT = phiTheta[rotationIdx, 1]
    xMap = xMap * float(s / 221)
    yMap = yMap * float(s / 221)
    return phiGT, thetaGT, xMap, yMap


def get_validation_image_paths(d):
    random.seed(1)

    valid_filenames = [os.path.join(d, x) for x in os.listdir(d)]
    random.shuffle(valid_filenames)
    if 'sun360' in d:
        n = int((1. - 0.11111111111111)*len(valid_filenames))
    else:
        n = int(0.9*len(valid_filenames))
    if 'test' in d:
        n = 0
    valid_filenames = valid_filenames[n:]
    return valid_filenames


def show_model_performance_histogram(est_axes, true_axes):
    thetas = get_angle_between(est_axes, true_axes)
    bins, counts = get_bins_and_counts(thetas)
    for b, c in zip(bins, counts):
        print(b, c)

    print('<10 {0:.4f}'.format( sum(counts[:4])))


def get_angle_between(est_axes, true_axes):
    true_axes = np.stack(true_axes)
    est_axes = np.stack(est_axes)
    dotted = (true_axes * est_axes).sum(axis=-1)
    theta = np.rad2deg(np.arccos(dotted))
    return theta


def get_bins_and_counts(thetas):
    bins = [0, 2.5, 5, 7.5, 10, 15, 20, 25, 30]
    counts = []
    s = thetas.shape[0]

    for i in range(len(bins) - 1):
        mask = np.logical_and(thetas < bins[i + 1], thetas >= bins[i])
        count = mask.sum()
        counts.append(count)
        thetas = thetas[~mask]
    greater_vals = thetas.shape[0]
    counts.append(greater_vals)
    counts = [x/s for x in counts]
    bins = ['< {}'.format(x) for x in bins[1:]] + ['> 30']
    return bins, counts


def validate(path_to_network_model, path_to_weights, directory, flat):

    validation_paths = get_validation_image_paths(directory)
    model = build_model(path_to_network_model=path_to_network_model, path_to_weights=path_to_weights)
    n = len(validation_paths)
    data = []
    for k, x in enumerate(validation_paths):
        if k % 100 == 0:
            print(k, n)
        orig = cv2.imread(x)
        if flat:
            phi_gt = theta_gt = 0.
            rotatedImage = orig
        else:
            phi_gt, theta_gt, xMap, yMap = get_random_rotation(orig.shape[0])
            rotatedImage = cv2.remap(orig, xMap, yMap, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        img = moderate_combo(rotatedImage)
        img = preprocess_input(img)
        phi, theta, z = forward_spherical(model, img)
        data.append((give_gt_cart_reg(phi_gt, theta_gt), z))

    show_model_performance_histogram(*zip(*data))


def to_r3(polygons):
    unnormalise = np.array([360., 180.])
    array = np.array(polygons)
    unnormal = array * np.broadcast_to(unnormalise, array.shape)
    orig_shape = unnormal.shape
    unnormal = unnormal.reshape([-1, 2])
    equi_ps = transforms.coordinate_transforms.EquirectangularImagePoints(180, unnormal)
    unnormal = equi_ps.to_r3().points
    return unnormal.reshape(orig_shape[:-1] + (3,))


def estimate_camera_z_axis(polygons):
    ps = to_r3(polygons)
    normals = np.cross(ps[:, 0], ps[:, 1])
    plane_basis = gram_schmidt(normals)
    z = np.cross(plane_basis[:, 1], plane_basis[:, 0])
    return np.concatenate([plane_basis[:, ::-1], z[:, None]], axis=1)


def gram_schmidt(vs):
    v_0 = vs[0]
    v_1 = vs[1]
    u_0 = v_0
    u_1 = v_1 - np.dot(u_0, v_1)/np.dot(u_0, u_0)*u_0
    return np.stack([u_0/np.linalg.norm(u_0), u_1/np.linalg.norm(u_1)], axis=1)


def get_test_data():
    annotations_pickle = '/home/ben/datasets/levelDataV2/test_annotations.pickle'
    image_dir = '/home/ben/datasets/levelDataV2/test_images'
    with open(annotations_pickle, 'rb') as pfile:
        annotations, ids, _ = pickle.load(pfile)
    data = []
    for k, im_id in enumerate(annotations):
        im_path = os.path.join(image_dir, im_id + '.JPG')
        if len(annotations[im_id]) == 2:
            z_axis = estimate_camera_z_axis(annotations[im_id])[:, -1]
            data.append((im_path, z_axis))
    return data


def test(path_to_network_model, path_to_weights):
    data = get_test_data()
    model = build_model(path_to_network_model=path_to_network_model, path_to_weights=path_to_weights)
    n = len(data)
    result = []
    for k, (p, axis) in enumerate(data):
        if k%100 == 0:
            print(k)
        orig = cv2.imread(p)
        orig = cv2.resize(orig, (442, 221))
        img = moderate_combo(orig)
        img = preprocess_input(img)
        phi, theta, z = forward_spherical(model, img)
        z[1] *= -1.

        result.append((axis, z))

    show_model_performance_histogram(*zip(*result))


if __name__ == '__main__':
    import random
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    # # disperse
    # path_to_network_model = '/home/ben/projects/comparison_leveller/data/training/regression/paper_model_disperse/model.json'
    # path_to_weights = '/home/ben/projects/comparison_leveller/data/training/regression/paper_model_disperse/Best/weights_150_5.08.h5'
    # directory = '/home/ben/datasets/levelDataV2/levelled_images'
    # validate(path_to_network_model, path_to_weights, directory, flat=True)
    # validate(path_to_network_model, path_to_weights, directory, flat=False)
    # test(path_to_network_model, path_to_weights)

    # sun360
    path_to_network_model = '/home/ben/projects/comparison_leveller/data/training/regression/paper_model_sun360/model.json'
    path_to_weights = '/home/ben/projects/comparison_leveller/data/training/regression/paper_model_sun360/Best/weights_360_4.43.h5'
    directory = '/home/ben/datasets/sun360/levelled_images'
    # validate(path_to_network_model, path_to_weights, directory, flat=True)
    validate(path_to_network_model, path_to_weights, directory, flat=False)
    directory = '/home/ben/datasets/sun360/test_images'
    validate(path_to_network_model, path_to_weights, directory, flat=True)


