import numpy as np
from os.path import *
PROJECT_FILE_PATH = dirname(dirname(dirname(dirname((dirname(abspath(__file__)))))))+"/"
UTILS_PATH = PROJECT_FILE_PATH+"src/utils/rotation/"
import sys
sys.path.insert(0,UTILS_PATH)
import CNN_train_package.fast_image_rotation as bp


# this script has functions for giving ground truth for the networks of different characteristics (for example: classification network with CE+geodesic loss function and
# spherical angles parameterization)

##### regressions
def give_gt_sph_reg(phi,theta):
    # use this for regression with spherical angles parameterization
    batch_y = np.zeros(2, dtype='float32')
    batch_y[0] = phi
    batch_y[1] = theta
    return batch_y

def give_gt_cart_reg(phi,theta):
    # use this for regression with cartesian parameterization
    batch_y = np.zeros(3,dtype = 'float32')
    [gt_x, gt_y, gt_z] = bp.spherical_to_cartesian(phi, theta)
    batch_y[0] = gt_x
    batch_y[1] = gt_y
    batch_y[2] = gt_z
    return batch_y

##### classifications
def give_gt_sph_cls_CE_geodesic(phi,theta,binsize):
    # use this for classification with spherical angles parameterization and CE+geodesic loss function
    # center of mass exactly equals to ground truth up-vector
    # given phi,theta, this function outputs a vector
    # you can adjust binsize (use 1,3 or 9)
    phiGT = phi / binsize
    thetaGT = theta / binsize

    integer_phi = int(np.floor(phiGT))
    small_number_phi = phiGT - integer_phi
    integer_theta = int(np.floor(thetaGT))
    small_number_theta = thetaGT - integer_theta

    phi_array_len = int(180/binsize+1)
    phi_array = np.zeros(phi_array_len, dtype=np.float32)

    if integer_phi == 180/binsize:
        phi_array[integer_phi] = 1
    else:
        phi_array[integer_phi] = 1 - small_number_phi
        phi_array[integer_phi + 1] = small_number_phi

    theta_array_len = int(360/binsize+1)
    theta_array = np.zeros(theta_array_len, dtype=np.float32)
    # theta_array 0 to 359
    # so theta_array[idx] represents idx degrees
    # 0 mod 360 = 360 mod 360

    if integer_theta == int(360/binsize):
        theta_array[integer_theta] = 1
    else:
        theta_array[integer_theta] = 1 - small_number_theta
        theta_array[integer_theta + 1] = small_number_theta

    batch_y = np.zeros( phi_array_len+theta_array_len, dtype=np.float32)
    final_array = np.concatenate([phi_array, theta_array])
    batch_y[:] = final_array

    return batch_y


def give_gt_cart_cls_CE_geodesic(phi,theta,binsize):
    # use this for classification with cartesian parameterization and CE+geodesic loss function
    # center of mass exactly equals to ground truth up-vector
    # given phi,theta, this function outputs a vector
    # you can adjust binsize (use 1,3 or 9)
    gt_x, gt_y, gt_z = bp.spherical_to_cartesian(phi, theta)

    num_bins = int(180/binsize+1)
    offset = int((num_bins - 1) / 2)
    step_size = 1 / ((num_bins - 1) / 2)

    [quo_x, rem_x] = np.divmod(gt_x, step_size )
    quo_x = int(quo_x)
    [quo_y, rem_y] = np.divmod(gt_y, step_size )
    quo_y = int(quo_y)
    [quo_z, rem_z] = np.divmod(gt_z, step_size )
    quo_z = int(quo_z)

    x_array = np.zeros(num_bins, dtype=np.float32)
    y_array = np.zeros(num_bins, dtype=np.float32)
    z_array = np.zeros(num_bins, dtype=np.float32)

    if quo_x+offset == num_bins-1:
        x_array[quo_x+offset] = 1
    else:
        x_array[quo_x + offset] = 1 - rem_x * (1 / step_size )
        x_array[quo_x + offset + 1] = rem_x * (1 / step_size )

    if quo_x + offset == num_bins - 1:
        y_array[quo_y+offset] = 1
    else:
        y_array[quo_y + offset] = 1 - rem_y * (1 / step_size )
        y_array[quo_y + offset + 1] = rem_y * (1 / step_size )

    if quo_z+offset == num_bins-1:
        z_array[quo_z+offset] = 1
    else:
        z_array[quo_z + offset] = 1 - rem_z * (1 / step_size )
        z_array[quo_z + offset + 1] = rem_z * (1 / step_size )

    batch_y = np.zeros( num_bins * 3, dtype=np.float32)
    final_array = np.concatenate([x_array, y_array, z_array])
    batch_y[:] = final_array

    return batch_y
