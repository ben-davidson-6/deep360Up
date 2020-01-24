import tensorflow as tf
import keras.backend as K
import sys
from os.path import *
PROJECT_FILE_PATH = dirname(dirname(dirname(dirname((dirname(abspath(__file__)))))))+"/"
HYPER_PARAM = dirname(dirname(abspath(__file__)))+'/'
sys.path.insert(0,HYPER_PARAM)
from hyper_params import batch_size
import math

clamp_high = 0.999999
clamp_low = -0.999999

# this script has loss functions.


def spherical_2_cartesian(inputs):
    #Change it into Radians
    phi = inputs[0] * math.pi / float(180)
    theta = inputs[1] * math.pi / float(180)
    rho = 1

    #Change them into x, y, z cartesian coordinates
    # x = tf.clip_by_value(rho * tf.sin(phi) * tf.cos(theta), -0.999999999998, 0.9999999999998)
    # y = tf.clip_by_value(rho * tf.sin(phi) * tf.sin(theta), -0.999999999998, 0.9999999999998)
    # z = tf.clip_by_value(rho * tf.cos(phi), -0.999999999998, 0.9999999999998)
    x = tf.sin(phi) * tf.cos(theta)
    y = tf.sin(phi) * tf.sin(theta)
    z = tf.cos(phi)
    #Get them into
    cartesian = tf.stack([x, y, z])

    return cartesian

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def cvpr_geodesic_cart_bin1(y_true, y_pred):
    # CE+geodesic for cartesian coordinate and binsize of 1
    alpha = 0.1
    gt_x,gt_y,gt_z = tf.split(y_true, [181, 181,181], axis=1)
    # computing cross entropy
    pred_x,pred_y,pred_z = tf.split(y_pred,[181,181,181],axis = 1)

    CE_x = categorical_crossentropy(gt_x, pred_x)
    CE_x = tf.reduce_mean(CE_x)
    CE_y = categorical_crossentropy(gt_y, pred_y)
    CE_y = tf.reduce_mean(CE_y)
    CE_z = categorical_crossentropy(gt_z, pred_z)
    CE_z = tf.reduce_mean(CE_z)

    #computing center of mass of xyz gt
    batch_size_ = [batch_size]
    xyz_idx = tf.range(0, 181, 1.0)
    xyz_idx = tf.tile(xyz_idx, tf.convert_to_tensor(batch_size_))
    xyz_idx = tf.reshape(xyz_idx, [batch_size, 181])

    gt_x_val = tf.reduce_sum(tf.multiply(gt_x, xyz_idx), axis=1)
    gt_x_val = (gt_x_val - 1) / 90 - 1
    gt_y_val = tf.reduce_sum(tf.multiply(gt_y, xyz_idx), axis=1)
    gt_y_val = (gt_y_val - 1) / 90 - 1
    gt_z_val = tf.reduce_sum(tf.multiply(gt_z, xyz_idx), axis=1)
    gt_z_val = (gt_z_val - 1) / 90 - 1

    pred_x_val = tf.reduce_sum(tf.multiply(pred_x, xyz_idx), axis=1)
    pred_x_val = (pred_x_val - 1) / 90 - 1
    pred_y_val = tf.reduce_sum(tf.multiply(pred_y, xyz_idx), axis=1)
    pred_y_val = (pred_y_val - 1) / 90 - 1
    pred_z_val = tf.reduce_sum(tf.multiply(pred_z, xyz_idx), axis=1)
    pred_z_val = (pred_z_val - 1) / 90 - 1

    gt_val = tf.stack([gt_x_val, gt_y_val,gt_z_val], axis=1)
    gt_val = K.l2_normalize(gt_val,axis = 1)
    pred_val = tf.stack([pred_x_val, pred_y_val,pred_z_val], axis=1)
    pred_val = K.l2_normalize(pred_val,axis = 1)

    dot_prod = tf.reduce_sum(tf.multiply(gt_val, pred_val), axis=1)
    dot_prod = tf.clip_by_value(dot_prod, clamp_low, clamp_high)
    geodesic_distance = tf.acos(dot_prod)
    geodesic_distance = tf.reduce_mean(geodesic_distance) * 180 / math.pi
    final_loss = CE_x + CE_y + CE_z  + alpha * geodesic_distance

    return final_loss

def cvpr_geodesic_cart_bin3(y_true, y_pred):
    # CE+geodesic for cartesian coordinate and binsize of 3
    alpha = 0.1
    gt_x,gt_y,gt_z = tf.split(y_true, [61, 61,61], axis=1)
    # computing cross entropy
    pred_x,pred_y,pred_z = tf.split(y_pred,[61,61,61],axis = 1)

    CE_x = categorical_crossentropy(gt_x, pred_x)
    CE_x = tf.reduce_mean(CE_x)
    CE_y = categorical_crossentropy(gt_y, pred_y)
    CE_y = tf.reduce_mean(CE_y)
    CE_z = categorical_crossentropy(gt_z, pred_z)
    CE_z = tf.reduce_mean(CE_z)

    #computing center of mass of xyz gt
    batch_size_ = [batch_size]
    xyz_idx = tf.range(0, 61, 1.0)
    xyz_idx = tf.tile(xyz_idx, tf.convert_to_tensor(batch_size_))
    xyz_idx = tf.reshape(xyz_idx, [batch_size, 61])

    gt_x_val = tf.reduce_sum(tf.multiply(gt_x, xyz_idx), axis=1)
    gt_x_val = (gt_x_val - 1) / 30 - 1
    gt_y_val = tf.reduce_sum(tf.multiply(gt_y, xyz_idx), axis=1)
    gt_y_val = (gt_y_val - 1) / 30 - 1
    gt_z_val = tf.reduce_sum(tf.multiply(gt_z, xyz_idx), axis=1)
    gt_z_val = (gt_z_val - 1) / 30 - 1

    pred_x_val = tf.reduce_sum(tf.multiply(pred_x, xyz_idx), axis=1)
    pred_x_val = (pred_x_val - 1) / 30 - 1
    pred_y_val = tf.reduce_sum(tf.multiply(pred_y, xyz_idx), axis=1)
    pred_y_val = (pred_y_val - 1) / 30 - 1
    pred_z_val = tf.reduce_sum(tf.multiply(pred_z, xyz_idx), axis=1)
    pred_z_val = (pred_z_val - 1) / 30 - 1

    gt_val = tf.stack([gt_x_val, gt_y_val,gt_z_val], axis=1)
    gt_val = K.l2_normalize(gt_val,axis = 1)
    pred_val = tf.stack([pred_x_val, pred_y_val,pred_z_val], axis=1)
    pred_val = K.l2_normalize(pred_val,axis = 1)

    dot_prod = tf.reduce_sum(tf.multiply(gt_val, pred_val), axis=1)
    dot_prod = tf.clip_by_value(dot_prod, clamp_low, clamp_high)
    geodesic_distance = tf.acos(dot_prod)
    geodesic_distance = tf.reduce_mean(geodesic_distance) * 180 / math.pi
    final_loss = CE_x + CE_y + CE_z  + alpha * geodesic_distance

    return final_loss

def cvpr_geodesic_cart_bin9(y_true, y_pred):
    # CE+geodesic for cartesian coordinate and binsize of 9
    alpha = 0.1
    gt_x,gt_y,gt_z = tf.split(y_true, [21, 21,21], axis=1)
    # computing cross entropy
    pred_x,pred_y,pred_z = tf.split(y_pred,[21,21,21],axis = 1)

    CE_x = categorical_crossentropy(gt_x, pred_x)
    CE_x = tf.reduce_mean(CE_x)
    CE_y = categorical_crossentropy(gt_y, pred_y)
    CE_y = tf.reduce_mean(CE_y)
    CE_z = categorical_crossentropy(gt_z, pred_z)
    CE_z = tf.reduce_mean(CE_z)

    #computing center of mass of xyz gt
    batch_size_ = [batch_size]
    xyz_idx = tf.range(0, 21, 1.0)
    xyz_idx = tf.tile(xyz_idx, tf.convert_to_tensor(batch_size_))
    xyz_idx = tf.reshape(xyz_idx, [batch_size, 61])

    gt_x_val = tf.reduce_sum(tf.multiply(gt_x, xyz_idx), axis=1)
    gt_x_val = (gt_x_val - 1) / 10 - 1
    gt_y_val = tf.reduce_sum(tf.multiply(gt_y, xyz_idx), axis=1)
    gt_y_val = (gt_y_val - 1) / 10 - 1
    gt_z_val = tf.reduce_sum(tf.multiply(gt_z, xyz_idx), axis=1)
    gt_z_val = (gt_z_val - 1) / 10 - 1

    pred_x_val = tf.reduce_sum(tf.multiply(pred_x, xyz_idx), axis=1)
    pred_x_val = (pred_x_val - 1) /10 - 1
    pred_y_val = tf.reduce_sum(tf.multiply(pred_y, xyz_idx), axis=1)
    pred_y_val = (pred_y_val - 1) / 10 - 1
    pred_z_val = tf.reduce_sum(tf.multiply(pred_z, xyz_idx), axis=1)
    pred_z_val = (pred_z_val - 1) / 10 - 1

    gt_val = tf.stack([gt_x_val, gt_y_val,gt_z_val], axis=1)
    gt_val = K.l2_normalize(gt_val,axis = 1)
    pred_val = tf.stack([pred_x_val, pred_y_val,pred_z_val], axis=1)
    pred_val = K.l2_normalize(pred_val,axis = 1)

    dot_prod = tf.reduce_sum(tf.multiply(gt_val, pred_val), axis=1)
    dot_prod = tf.clip_by_value(dot_prod, clamp_low, clamp_high)
    geodesic_distance = tf.acos(dot_prod)
    geodesic_distance = tf.reduce_mean(geodesic_distance) * 180 / math.pi
    final_loss = CE_x + CE_y + CE_z  + alpha * geodesic_distance

    return final_loss



def cvpr_geodesic_sph_bin1(y_true,y_pred):
    # CE+geodesic for spherical angles and binsize of 1
    alpha = 0.5
    # computing cross entropy
    gt_phi,gt_theta = tf.split(y_true,[181,361],axis = 1)
    pred_phi,pred_theta = tf.split(y_pred,[181,361],axis = 1)
    CE_phi = categorical_crossentropy(gt_phi,pred_phi)
    CE_phi = tf.reduce_mean(CE_phi)
    CE_theta = categorical_crossentropy(gt_theta,pred_theta)
    CE_theta = tf.reduce_mean(CE_theta)

    #computing geodesic distance
    gt_phi_val = tf.argmax(gt_phi,axis = 1)
    gt_phi_val = tf.cast(gt_phi_val,dtype = tf.float32)
    gt_theta_val = tf.argmax(gt_theta,axis = 1)
    gt_theta_val = tf.cast(gt_theta_val, dtype=tf.float32)
    #computing center of mass of phi and theta
    batchsize = [batch_size]
    phi_idx = tf.range(0, 181, 1.0)

    phi_idx = tf.tile(phi_idx, tf.convert_to_tensor(batchsize))
    phi_idx = tf.reshape(phi_idx, [batch_size, 181])
    pred_phi_val = tf.reduce_sum(tf.multiply(pred_phi, phi_idx), axis=1)

    theta_idx = tf.range(0, 361, 1.0)
    theta_idx = tf.tile(theta_idx, tf.convert_to_tensor(batchsize))
    theta_idx = tf.reshape(theta_idx, [batch_size, 361])
    pred_theta_val = tf.reduce_sum(tf.multiply(pred_theta, theta_idx), axis=1)

    gt_val = tf.stack([gt_phi_val,gt_theta_val],axis = 1)
    pred_val = tf.stack([pred_phi_val,pred_theta_val],axis = 1)

    gt_cart = tf.map_fn(spherical_2_cartesian,gt_val)
    pred_cart = tf.map_fn(spherical_2_cartesian,pred_val)

    dot_prod = tf.reduce_sum(tf.multiply(gt_cart,pred_cart),axis = 1)
    dot_prod = tf.clip_by_value(dot_prod,clamp_low,clamp_high)
    geodesic_distance = tf.acos(dot_prod)
    geodesic_distance = tf.reduce_mean(geodesic_distance)*180/math.pi
    final_loss = CE_phi+CE_theta+alpha*geodesic_distance

    return final_loss


def cvpr_geodesic_sph_bin3(y_true, y_pred):
    # CE+geodesic for spherical angles and binsize of 3
    alpha = 0.1
    # computing cross entropy
    gt_phi, gt_theta = tf.split(y_true, [61, 121], axis=1)
    pred_phi, pred_theta = tf.split(y_pred, [61, 121], axis=1)
    CE_phi = categorical_crossentropy(gt_phi, pred_phi)
    CE_phi = tf.reduce_mean(CE_phi)
    CE_theta = categorical_crossentropy(gt_theta, pred_theta)
    CE_theta = tf.reduce_mean(CE_theta)

    # computing geodesic distance
    # gt_phi_val = tf.argmax(gt_phi, axis=1)
    # gt_phi_val = tf.cast(gt_phi_val, dtype=tf.float32)
    # gt_theta_val = tf.argmax(gt_theta, axis=1)
    # gt_theta_val = tf.cast(gt_theta_val, dtype=tf.float32)

    #computing center of mass of phi and theta gt
    batchsize = [batch_size]
    phi_idx = tf.range(0, 181, 3.0)
    phi_idx = tf.tile(phi_idx, tf.convert_to_tensor(batchsize))
    phi_idx = tf.reshape(phi_idx, [batch_size, 61])
    gt_phi_val = tf.reduce_sum(tf.multiply(gt_phi,phi_idx),axis = 1)

    theta_idx = tf.range(0, 361, 3.0)
    theta_idx = tf.tile(theta_idx, tf.convert_to_tensor(batchsize))
    theta_idx = tf.reshape(theta_idx, [batch_size, 121])
    gt_theta_val = tf.reduce_sum(tf.multiply(gt_theta, theta_idx), axis=1)

    # computing center of mass of phi and theta
    # batch_size = [40]
    # phi_idx = tf.range(0, 181, 1.0)
    # phi_idx = tf.tile(phi_idx, tf.convert_to_tensor(batch_size))
    # phi_idx = tf.reshape(phi_idx, [40, 181])
    pred_phi_val = tf.reduce_sum(tf.multiply(pred_phi, phi_idx), axis=1)

    # theta_idx = tf.range(0, 360, 1.0)
    # theta_idx = tf.tile(theta_idx, tf.convert_to_tensor(batch_size))
    # theta_idx = tf.reshape(theta_idx, [40, 360])
    pred_theta_val = tf.reduce_sum(tf.multiply(pred_theta, theta_idx), axis=1)

    gt_val = tf.stack([gt_phi_val, gt_theta_val], axis=1)
    pred_val = tf.stack([pred_phi_val, pred_theta_val], axis=1)

    gt_cart = tf.map_fn(spherical_2_cartesian, gt_val)
    pred_cart = tf.map_fn(spherical_2_cartesian, pred_val)

    dot_prod = tf.reduce_sum(tf.multiply(gt_cart, pred_cart), axis=1)
    dot_prod = tf.clip_by_value(dot_prod, clamp_low, clamp_high)
    geodesic_distance = tf.acos(dot_prod)
    geodesic_distance = tf.reduce_mean(geodesic_distance) * 180 / math.pi
    final_loss = CE_phi + CE_theta + alpha * geodesic_distance

    return final_loss

def cvpr_geodesic_sph_bin9(y_true, y_pred):
    # CE+geodesic for spherical angles and binsize of 9
    alpha = 0.1
    # computing cross entropy
    gt_phi, gt_theta = tf.split(y_true, [61, 120], axis=1)
    pred_phi, pred_theta = tf.split(y_pred, [61, 120], axis=1)
    CE_phi = categorical_crossentropy(gt_phi, pred_phi)
    CE_phi = tf.reduce_mean(CE_phi)
    CE_theta = categorical_crossentropy(gt_theta, pred_theta)
    CE_theta = tf.reduce_mean(CE_theta)

    # computing geodesic distance
    # gt_phi_val = tf.argmax(gt_phi, axis=1)
    # gt_phi_val = tf.cast(gt_phi_val, dtype=tf.float32)
    # gt_theta_val = tf.argmax(gt_theta, axis=1)
    # gt_theta_val = tf.cast(gt_theta_val, dtype=tf.float32)

    #computing center of mass of phi and theta gt
    batchsize = [batch_size]
    phi_idx = tf.range(0, 181, 3.0)
    phi_idx = tf.tile(phi_idx, tf.convert_to_tensor(batchsize))
    phi_idx = tf.reshape(phi_idx, [batch_size, 61])
    gt_phi_val = tf.reduce_sum(tf.multiply(gt_phi,phi_idx),axis = 1)

    theta_idx = tf.range(0, 360, 3.0)
    theta_idx = tf.tile(theta_idx, tf.convert_to_tensor(batchsize))
    theta_idx = tf.reshape(theta_idx, [batch_size, 120])
    gt_theta_val = tf.reduce_sum(tf.multiply(gt_theta, theta_idx), axis=1)

    # computing center of mass of phi and theta
    # batch_size = [40]
    # phi_idx = tf.range(0, 181, 1.0)
    # phi_idx = tf.tile(phi_idx, tf.convert_to_tensor(batch_size))
    # phi_idx = tf.reshape(phi_idx, [40, 181])
    pred_phi_val = tf.reduce_sum(tf.multiply(pred_phi, phi_idx), axis=1)

    # theta_idx = tf.range(0, 360, 1.0)
    # theta_idx = tf.tile(theta_idx, tf.convert_to_tensor(batch_size))
    # theta_idx = tf.reshape(theta_idx, [40, 360])
    pred_theta_val = tf.reduce_sum(tf.multiply(pred_theta, theta_idx), axis=1)

    gt_val = tf.stack([gt_phi_val, gt_theta_val], axis=1)
    pred_val = tf.stack([pred_phi_val, pred_theta_val], axis=1)

    gt_cart = tf.map_fn(spherical_2_cartesian, gt_val)
    pred_cart = tf.map_fn(spherical_2_cartesian, pred_val)

    dot_prod = tf.reduce_sum(tf.multiply(gt_cart, pred_cart), axis=1)
    dot_prod = tf.clip_by_value(dot_prod, clamp_low, clamp_high)
    geodesic_distance = tf.acos(dot_prod)
    geodesic_distance = tf.reduce_mean(geodesic_distance) * 180 / math.pi
    final_loss = CE_phi + CE_theta + alpha * geodesic_distance

    return final_loss



# for metrics
def arc_error(y_true, y_pred):
    # geodesic distance metric for spherical angles

    #spherical to cartesian [x, y, z] normalized vector
    y_true = tf.map_fn(spherical_2_cartesian, y_true)
    y_pred = tf.map_fn(spherical_2_cartesian, y_pred)

    #dot product
    dot = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    dot = tf.clip_by_value(dot, -0.999999999998, 0.9999999999998)

    #acos to get angles
    theta = tf.acos(dot) * float(180) / math.pi

    return tf.reduce_mean(theta)


def arc_error_cart(y_true, y_pred):
    # goedesic distance metric for cartesian coordinates
    #spherical to cartesian [x, y, z] normalized vector


    #dot product
    dot = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    dot = tf.clip_by_value(dot, -0.999999999998, 0.9999999999998)

    #acos to get angles
    theta = tf.acos(dot) * float(180) / math.pi

    return tf.reduce_mean(theta)


# for regressions

def _logcosh(x):
    # logcosh = log(cosh(x))
    # cosh(x) = (e^x+e^(-x))/2

    return x + K.softplus(tf.constant(-2.,dtype=tf.float32) * x) - K.log(tf.constant(2.,dtype=tf.float32))


def geodesic_spherical(y_true,y_pred):
    # y_true and y_pred respectively correspond to (phi,theta) GT and predicted

    # converstion to cartesian coordinates
    y_true = tf.map_fn(spherical_2_cartesian, y_true)
    y_pred = tf.map_fn(spherical_2_cartesian, y_pred)

    # cosTheta = K.dot(y_true,y_pred)
    dot_prod = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    dot_prod = tf.clip_by_value(dot_prod,clamp_low,clamp_high)

    dot_prod = tf.clip_by_value(dot_prod,clamp_low,clamp_high)
    geodesic_distance = tf.acos(dot_prod)*180/math.pi

    logcosh_geodesic = _logcosh(geodesic_distance)

    return tf.reduce_mean(logcosh_geodesic)


def angle_spherical(y_true,y_pred):
    # |phi_{gt}-phi{pred}|+min(|theta_{gt}-theta_{pred}|,180-|theta_{gt}-theta_{pred}|)

    gt_phi,gt_theta = tf.split(y_true, [1, 1], 1)
    pred_phi,pred_theta = tf.split(y_pred,[1,1],1)
    pred_theta_mod = tf.floormod(pred_theta,360)
    theta_dist = tf.abs(gt_theta-pred_theta_mod)
    theta_dist = tf.minimum(360-theta_dist,theta_dist)
    phi_dist = tf.abs(gt_phi-pred_phi)
    angle_dist = (theta_dist+phi_dist)/2
    # angle_dist = angle_dist*180/math.pi
    logcosh_angle_dist = _logcosh(angle_dist)

    return tf.reduce_mean(logcosh_angle_dist)

def geodesic_cartesian(y_true,y_pred):

    # cosTheta = K.dot(y_true,y_pred)
    y_pred = K.l2_normalize(y_pred, axis = -1)
    cosTheta = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    # cosTheta = tf.clip_by_value(cosTheta, -0.999999, 0.999999)
    cosTheta = tf.reduce_mean(cosTheta,0)
    alpha = 0.99-cosTheta
    geodesicDistance = tf.acos(cosTheta)

    logcoshGeodesic = _logcosh(geodesicDistance)
    return logcoshGeodesic