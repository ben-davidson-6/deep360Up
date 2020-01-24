import cv2
import numpy as np
from numpy.linalg import inv, norm
from numba import jit
import math
from os.path import *
import os
from pathlib import Path
PROJECT_FILE_PATH = dirname(dirname(dirname((dirname(abspath(__file__)))))) + "/"
CURRENT_FILE_PATH = dirname(abspath(__file__)) + "/"
# IMAGE_FILE_PATH = PROJECT_FILE_PATH + "data/images/"
# REMAP_FILE_PATH = PROJECT_FILE_PATH + "data/remapping_matrix/"
# SAVE_FILE_PATH = PROJECT_FILE_PATH + "data/remapping_matrix/originalCartCoord/"

def predict_pt(R):
    ori = np.array([[0], [0], [1]])
    dir = np.matmul(R, ori)
    phi,theta = cartesian_to_spherical(dir[0], dir[1], dir[2])
    return phi, theta

def spherical_to_cartesian(phi, theta):
    # Input:
    #  rho (radius of sphere = 1), phi and theta in degree
    # Output:
    #  x,y,z (cartesian) that corresponds to rho,phi,theta
    # Goal:
    # convert spherical coordinates to cartesian coordinate
    # About convention :
    #  phi = arccos(z/r) , theta = arctan(y/x) range of phi [0,pi], range of theta [0,2pi]
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)

    x =  math.sin(phi) * math.cos(theta)
    y =  math.sin(phi) * math.sin(theta)
    z =  math.cos(phi)

    return [x, y, z]

def cartesian_to_spherical(x, y, z):
    # Input:
    #  x,y,z
    # Output:
    #  x,y,z (cartesian) that corresponds to rho,phi,theta
    # Goal:
    # convert spherical coordinates to cartesian coordinate
    # About convention :
    #  phi = arccos(z/r) , theta = arctan(y/x) range of phi [0,pi], range of theta [0,2pi]
    x_2 = math.pow(x, 2)
    y_2 = math.pow(y, 2)
    z_2 = math.pow(z, 2)

    theta = float(math.atan2(y, x))
    # atan2 returns value of which range is [-pi,pi], range of theta is [0,2pi] so if theta is negative value,actual value is theta+2pi
    if theta < 0:
        theta = theta + 2 * math.pi

    # theta = theta % (2* PI) # potential ERROR : phi [0,pi] theta [0,2pi] but atan2 returns value [-pi,pi]

    rho = x_2 + y_2 + z_2
    rho = math.sqrt(rho)
    phi = math.acos(z / rho)
    phi = np.rad2deg(phi)
    theta = np.rad2deg(theta)

    return [phi,theta]

@jit(nopython=True,cache=True)
def rotate_sphere_given_phi_theta(R, spherePoints):
    # Input:
    #       phi,theta in degrees and spherePoints(x,y,z of on sphere dimension (height,width,3) )
    # Output:
    #       spherePointsRotated of which dimension is (h,w,3) and contains (x',y',z' )
    #  (x',y',z')=R*(x,y,z) where R maps (0,0,1) to (vx,vy,vz) defined by theta,phi (i.e. R*(0,0,1)=(vx,vy,vz))
    # Goal:
    #      apply R to every point on sphere

    h, w, c = spherePoints.shape
    spherePointsRotated = np.zeros((h, w, c),dtype=np.float64)

    for y in range(0, h):
        for x in range(0, w):
            pointOnSphere = spherePoints[y, x, :]
            pointOnSphereRotated = np.dot(R, pointOnSphere)
            spherePointsRotated[y, x, :] = pointOnSphereRotated
            # spherePointsRotated[y, x, :] = np.dot(R, pointOnSphere)

    return spherePointsRotated




def restore_sphere_given_phi_theta(phi, theta, spherePoints):
    # Input:
    #       phi,theta in degrees and sourceCartCoord(x,y,z of on sphere dimension (height,width,3) )
    # Output:
    #       rotated sourceCartCoord (x,y,z)
    # Goal:
    #       Current up vector is (x,y,z) which corresponds to phi,theta
    #       This function puts (x,y,z) to (0,0,1)

    # Inverse of a rotation matrix (phi,theta)
    R = calculate_Rmatrix_from_phi_theta(phi, theta)

    R_inv = inv(R)
    h, w, c = spherePoints.shape
    spherePointsRotated = np.zeros((h, w, c))
    # Multiply all x,y,z by inverse rotation matrix
    for y in range(0, h):
        for x in range(0, w):
            pointOnSphere = spherePoints[y, x, :]
            movedPoint = np.dot(R_inv, pointOnSphere)
            spherePointsRotated[y, x, :] = movedPoint

    return spherePointsRotated

@jit(nopython=True,cache=True)
def sphere_to_flat(spherePointsRotated, height, width):
    # Input:
    #       y,x coordinate on 2d flat image,numpy nd array of dimension (height,width,3). ndarray(y,x) has x,y,z value on sphere ,height and width of an image
    # Output:
    #       x,y coordinate of 2d flat image
    # Goal:
    #       calculate destination x,y coordinate given information x,y(2d flat) <-> x,y,z(sphere)
    map_y = np.zeros((height, width), dtype=np.float32)
    map_x = np.zeros((height, width), dtype=np.float32)

    factor_phi = (height-1)/np.pi
    factor_theta = (width-1)/(2*np.pi)

    # Get multiplied(by inverted rotation matrix) x,y,z coordinates
    for image_y in range(0, height):
        for image_x in range(0, width):
            pointOnRotatedSphere_x = spherePointsRotated[image_y, image_x, 0]
            pointOnRotatedSphere_y = spherePointsRotated[image_y, image_x, 1]
            pointOnRotatedSphere_z = spherePointsRotated[image_y, image_x, 2]

            x_2 = np.power(pointOnRotatedSphere_x, 2)
            y_2 = np.power(pointOnRotatedSphere_y, 2)
            z_2 = np.power(pointOnRotatedSphere_z, 2)

            theta = float(np.arctan2(pointOnRotatedSphere_y, pointOnRotatedSphere_x))
            # atan2 returns value of which range is [-pi,pi], range of theta is [0,2pi] so if theta is negative value,actual value is theta+2pi
            if theta < 0:
                theta = theta + np.multiply(2,np.pi)


            rho = x_2 + y_2 + z_2
            rho = np.sqrt(rho)
            phi = np.arccos(pointOnRotatedSphere_z / rho)


            map_y[image_y, image_x] = phi*factor_phi
            map_x[image_y, image_x] = theta*factor_theta


    return [map_x, map_y]

# def pano360_to_cartCoord(height, width):
#     # Input:
#     #      height and width of image
#     # Output:
#     #      write (height,width,3) numpy ndarray. (y,x) of array has (x,y,z) value which is on sphere. This is saved in specific directory
#     # Goal:
#     #      get a relation between (y,x) on flat image and (x,y,z) on sphere
#     global SAVE_FILE_PATH
#     # Create matrix that contains x,y,z coordinates
#     save = np.empty([height, width, 3])
#
#     X_TO_THETA = []
#     Y_TO_PHI = []
#
#     # Calculate theta for all x
#     for x in range(0, width):
#         theta_calculated = linear_map_theta(x, width)
#         X_TO_THETA.append(theta_calculated)
#
#     # Calculate phi for all y
#     for y in range(0, height):
#         phi_calculated = linear_map_phi(y, height)
#         Y_TO_PHI.append(phi_calculated)
#
#     # For every pixel coordinates, create a matrix that contains the
#     # corresponding (x,y,z) coordinates
#     for y in range(0, height):
#         for x in range(0, width):
#             theta = X_TO_THETA[x]
#             phi = Y_TO_PHI[y]
#             cartesian = spherical_to_cartesian(phi, theta)
#             # Save the carteian value to the matrix
#             save[y, x] = cartesian
#
#     if not os.path.exists(SAVE_FILE_PATH):
#         os.makedirs(SAVE_FILE_PATH)
#
#     np.save(SAVE_FILE_PATH + 'pano360_to_cartCoord(' + str(width) + "_" + str(height) + ").npy", save)

@jit(nopython=True,cache = True)
def flat_to_sphere(height, width):
    # Input:
    #      height and width of image
    # Output:
    #      return (height,width,3) numpy ndarray. (y,x) of array has (x,y,z) value which is on sphere.
    # Goal:
    #      return sphere points
    # Create matrix that contains x,y,z coordinates

    sphere = np.zeros((height, width, 3))
    x_to_theta = np.zeros(width)
    y_to_phi = np.zeros(height)

    theta_slope = 2*np.pi/(width-1)
    phi_slope = np.pi/(height-1)


    #linear map from [y,x] to [phi,theta]
    for x in range(0, width):
        x_to_theta[x] = np.rad2deg(np.multiply(x,theta_slope))

    for y in range(0,height):
        y_to_phi[y] = np.rad2deg(np.multiply(y,phi_slope))


    # For every pixel coordinates, create a matrix that contains the
    # corresponding (x,y,z) coordinates
    for y_f in range(0, height):
        for x_f in range(0, width):
            theta = x_to_theta[x_f]
            phi = y_to_phi[y_f]

            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
            x_s = np.sin(phi) * np.cos(theta)
            y_s = np.sin(phi) * np.sin(theta)
            z_s = np.cos(phi)
            sphere[y_f,x_f,0] = x_s
            sphere[y_f,x_f,1] = y_s
            sphere[y_f,x_f,2] = z_s

    return sphere


@jit(nopython=True,cache=True)
def skewSymmetricCrossProduct(v):
    # Input:
    #   a vector in R^3
    # Output:
    #   [ 0 -v3 v2 ; v3 0 -v1; -v2 v1 0]
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    # v_x1 = np.array([0, -v3, v2])
    # v_x2 = np.array([v3, 0, -v1])
    # v_x3 = np.array([-v2, v1, 0])
    skewSymmetricMatrix = np.array([[0, -v3, v2],[v3, 0, -v1],[-v2, v1, 0]],dtype=np.float64)
    # skewSymmetricMatrix = np.vstack((v_x1, v_x2))
    # skewSymmetricMatrix = np.vstack((skewSymmetricMatrix, v_x3))
    return skewSymmetricMatrix

@jit(nopython=True,cache=True)
def calculate_Rmatrix_from_phi_theta(phi, theta):
    # Inputs:
    #       phi,theta value in degrees
    # Outputs:
    #       rotation matrix that moves [0,0,1] to ([x,y,z] that is equivalent to (phi,theta))
    # Goal:
    #    A = [0,0,1] B = [x,y,z] ( = phi,theta) the goal is to find rotation matrix R where R*A == B
    # please refer to this website https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    # v = a cross b ,s = ||v|| (sine of angle), c = a dot b (cosine of angle)

    epsilon = 1e-7
    A = np.array([0, 0, 1],dtype=np.float64)  # original up-vector
    # B = spherical_to_cartesian(phi,theta)  # target vector

    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    B = np.array([x,y,z],dtype=np.float64)

    desiredResult = B
    # dot(R,A) == B
    # If A == B then return identity(3)
    if A[0] - B[0] < epsilon and A[0] - B[0] > -epsilon and A[1] - B[1] < epsilon and A[1] - B[1] > -epsilon and A[2] - B[2] < epsilon and A[2] - B[2] > -epsilon:
        # print('Identity matrix is returned')
        return np.identity(3)

    # v = np.cross(A, B)
    # In the numba, numpy.cross is not supported
    cross_1 = np.multiply(A[1],B[2])-np.multiply(A[2],B[1])
    cross_2 = np.multiply(A[2],B[0])-np.multiply(A[0],B[2])
    cross_3 = np.multiply(A[0],B[1])-np.multiply(A[1],B[0])
    v = np.array([cross_1,cross_2,cross_3])

    c = np.dot(A, B)
    skewSymmetric = skewSymmetricCrossProduct(v)

    if -epsilon < c + 1 and c + 1 < epsilon:
        R = -np.identity(3)
    else:
        R = np.identity(3) + skewSymmetric + np.dot(skewSymmetric, skewSymmetric) * (
                    1 / (1 + c))  # what if 1+c is 0?


    return R



# @jit(nopython=True,cache=True)
def compute_R_v1_v2(v1, v2):
    # Inputs:
    #       phi,theta value in degrees
    # Outputs:
    #       rotation matrix that moves [0,0,1] to ([x,y,z] that is equivalent to (phi,theta))
    # Goal:
    #    A = [0,0,1] B = [x,y,z] ( = phi,theta) the goal is to find rotation matrix R where R*A == B
    # please refer to this website https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    # v = a cross b ,s = ||v|| (sine of angle), c = a dot b (cosine of angle)

    epsilon = 1e-7
    A = v1  # original up-vector
    # B = spherical_to_cartesian(phi,theta)  # target vector
    B = v2

    desiredResult = B
    # dot(R,A) == B
    # If A == B then return identity(3)
    if A[0] - B[0] < epsilon and A[0] - B[0] > -epsilon and A[1] - B[1] < epsilon and A[1] - B[1] > -epsilon and A[2] - B[2] < epsilon and A[2] - B[2] > -epsilon:
        # print('Identity matrix is returned')
        return np.identity(3)

    # v = np.cross(A, B)
    # In the numba, numpy.cross is not supported
    cross_1 = np.multiply(A[1],B[2])-np.multiply(A[2],B[1])
    cross_2 = np.multiply(A[2],B[0])-np.multiply(A[0],B[2])
    cross_3 = np.multiply(A[0],B[1])-np.multiply(A[1],B[0])
    v = np.array([cross_1,cross_2,cross_3])

    c = np.dot(A, B)
    skewSymmetric = skewSymmetricCrossProduct(v)

    if -epsilon < c + 1 and c + 1 < epsilon:
        R = -np.identity(3)
    else:
        R = np.identity(3) + skewSymmetric + np.dot(skewSymmetric, skewSymmetric) * (
                    1 / (1 + c))  # what if 1+c is 0?
    return R

# @jit(nopython=True,cache=True)
def rotate_map_given_phi_theta(phi, theta, height, width):
    # Inputs:
    #       phi,theta in degrees , height and width of an image
    # output:
    #       rotation map for x and y coordinate
    # goal:
    #       calculating rotation map for corresponding image dimension and phi,theta value.
    #       (1,0,0)(rho,phi,theta) on sphere goes to (1,phi,theta)


    # if not original_file.is_file():
    # step1

    spherePoints = flat_to_sphere(height, width)
    R = calculate_Rmatrix_from_phi_theta(phi,theta)
    R_inv = inv(R)
    #step2
    spherePointsRotated = rotate_sphere_given_phi_theta(R_inv, spherePoints)
    #Create two mapping variable
    #step3
    [map_x,map_y] = sphere_to_flat(spherePointsRotated,height,width)

    # dst(y,x) = src(map_x(y,x),map_y(y,x))
    return [map_x, map_y]


def rotate_map_given_R(R, height, width):
    # Inputs:
    #       phi,theta in degrees , height and width of an image
    # output:
    #       rotation map for x and y coordinate
    # goal:
    #       calculating rotation map for corresponding image dimension and phi,theta value.
    #       (1,0,0)(rho,phi,theta) on sphere goes to (1,phi,theta)

    def pos_conversion(x, y, z):
        # given postech protocol
        # return my protocol

        return z, -x, y

    def inv_conversion(x, y, z):
        # given my conversion
        # convert it to postech system.

        return -y, z, x

    # if not original_file.is_file():
    # step1
    spherePoints = flat_to_sphere(height, width)
    # R = calculate_Rmatrix_from_phi_theta(phi,theta)
    R_inv = inv(R)
    #step2
    spherePointsRotated = rotate_sphere_given_phi_theta(R_inv, spherePoints)

    #Create two mapping variable
    #step3
    [map_x,map_y] = sphere_to_flat(spherePointsRotated,height,width)

    # dst(y,x) = src(map_x(y,x),map_y(y,x))
    return [map_x, map_y]


def rotate_map_given_R_postech(R, height, width):
    # Inputs:
    #       phi,theta in degrees , height and width of an image
    # output:
    #       rotation map for x and y coordinate
    # goal:
    #       calculating rotation map for corresponding image dimension and phi,theta value.
    #       (1,0,0)(rho,phi,theta) on sphere goes to (1,phi,theta)

    def pos_conversion(x, y, z):
        # given postech protocol
        # return my protocol

        return z, -x, y

    def inv_conversion(x, y, z):
        # given my conversion
        # convert it to postech system.

        return -y, z, x

    # if not original_file.is_file():
    # step1

    spherePoints = flat_to_sphere(height, width)
    postech_sphere = np.zeros((height,width,3),dtype = np.float64)
    # mine to postech
    x_s = spherePoints[:,:,0]
    y_s = spherePoints[:,:,1]
    z_s = spherePoints[:,:,2]
    postech_sphere[:,:,0] = -y_s
    postech_sphere[:,:,1] = z_s
    postech_sphere[:,:,2] = x_s

    # R = calculate_Rmatrix_from_phi_theta(phi,theta)
    R_inv = inv(R)
    #step2
    spherePointsRotated_postech = rotate_sphere_given_phi_theta(R_inv, postech_sphere)
    spherePointsRotated = np.zeros((height,width,3),dtype = np.float32)
    #potech to mine
    x_s = spherePointsRotated_postech[:,:,0]
    y_s = spherePointsRotated_postech[:,:,1]
    z_s = spherePointsRotated_postech[:,:,2]
    spherePointsRotated[:,:,0] = z_s
    spherePointsRotated[:,:,1] = -x_s
    spherePointsRotated[:,:,2] = y_s

    #Create two mapping variable
    #step3
    [map_x,map_y] = sphere_to_flat(spherePointsRotated,height,width)

    # dst(y,x) = src(map_x(y,x),map_y(y,x))
    return [map_x, map_y]

@jit(nopython=True,cache=True)
def restore_map_given_phi_theta(phi, theta, height, width):
    # Inputs:
    #       phi,theta in degrees , height and width of an image
    # output:
    #       rotation map for x and y coordinate
    # goal:
    #       calculating rotation map for corresponding image dimension and phi,theta value.
    #       (1,0,0)(rho,phi,theta) on sphere goes to (1,phi,theta)

    # originalFilePath = SAVE_FILE_PATH+'pano360_to_cartCoord('+str(width)+"_"+str(height)+").npy"
    # original_file = Path(originalFilePath)
    # if not original_file.is_file():
    # step1
    spherePoints = flat_to_sphere(height, width)
    # spherePoints = np.load(SAVE_FILE_PATH+'pano360_to_cartCoord('+str(width)+"_"+str(height)+").npy")
    R = calculate_Rmatrix_from_phi_theta(phi,theta)
    #step2
    spherePointsRotated = rotate_sphere_given_phi_theta(R, spherePoints)
    #Create two mapping variable
    #step3
    [map_x,map_y] = sphere_to_flat(spherePointsRotated,height,width)

    # dst(y,x) = src(map_x(y,x),map_y(y,x))
    return [map_x, map_y]


def rotate_image_given_phi_theta(image,phi, theta):
    h,w,c = image.shape
    [map_x,map_y] = rotate_map_given_phi_theta(phi,theta,h,w)
    rotated_image = cv2.remap(image,map_x,map_y,cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)

    return rotated_image

def rectify_image_given_phi_theta(image,phi, theta):
    h,w,c = image.shape
    [map_x,map_y] = restore_map_given_phi_theta(phi,theta,h,w)
    rotated_image = cv2.remap(image,map_x,map_y,cv2.INTER_CUBIC,borderMode = cv2.BORDER_TRANSPARENT)

    return rotated_image

if __name__ == "__main__":
    x,skew = calculate_Rmatrix_from_phi_theta(12,31)
    np.save('skew.npy',skew)
    exit()
    import sys
    sys.path.insert(0,'/home/cmlkaist/PycharmProjects/Aidenbackup/VR_360_Project/src/models')
    import network_eval as ne
    from keras.models import model_from_json
    import keras.backend as K
    import tensorflow as tf
    from keras import optimizers

    model_path = '/home/cmlkaist/PycharmProjects/Aidenbackup/VR_360_Project/data/training/regression/cart_classification_cubic_transplant/'
    with tf.device('/gpu:0'):
        json_file = open(model_path + 'model.json', 'r')
        model_json = json_file.read()
        json_file.close()
        learning_rate = 0.00001
        adam = optimizers.Adam(lr=learning_rate)
        estimator = model_from_json(model_json, custom_objects={"backend": K, "tf": tf})
        # estimator = get_model()
        estimator.compile(loss='logcosh', optimizer=adam)
        # load weights into new model
        estimator.load_weights(model_path + 'Best/' + 'weights_74_7.53.h5')


    def forward_classification_cart_cm(estimator, image):
        batch_x = np.zeros((1,) + (221,442,3), dtype='float32')
        HEIGHT, WIDTH, CHANNEL = image.shape

        if HEIGHT != 221 or WIDTH != 442:
            resized_image = cv2.resize(image, (442, 221), interpolation=cv2.INTER_CUBIC)
        else:
            resized_image = image

        batch_x[0] = resized_image
        output = estimator.predict(batch_x)
        output = np.squeeze(output)
        pred_x = output[0:181]
        pred_y = output[181:362]
        pred_z = output[362:543]
        # pred = np.split(output,[181,181,181])

        # pred_x = np.squeeze(pred_x)
        # pred_y = np.squeeze(pred_y)
        # pred_z = np.squeeze(pred_z)

        x_idx = np.asarray(range(0, 181))
        y_idx = np.asarray(range(0, 181))
        z_idx = np.asarray(range(0, 181))

        # phi_summation = np.sum(phi_idx)
        # theta_summation = np.sum(theta_idx)

        cm_x = np.sum(pred_x * x_idx)
        cm_y = np.sum(pred_y * y_idx)
        cm_z = np.sum(pred_z * z_idx)

        cm_x = (cm_x - 1) / 90 - 1
        cm_y = (cm_y - 1) / 90 - 1
        cm_z = (cm_z - 1) / 90 - 1
        up_vec = np.asarray([cm_x, cm_y, cm_z])
        norm = np.sqrt(cm_x ** 2 + cm_y ** 2 + cm_z ** 2)
        up_vec = up_vec / norm
        phi, theta = cartesian_to_spherical(up_vec[0], up_vec[1], up_vec[2])

        # phi = np.argmax(phi)
        # theta = np.argmax(theta)

        return phi, theta

    path_img = '/home/cmlkaist/PycharmProjects/Aidenbackup/HAHE360_dataset/RicohDB_our_orig_CVPR18/backup/RicohDB_orig/'
    to_save = '/home/cmlkaist/PycharmProjects/Aidenbackup/HAHE360_dataset/RicohDB_our_orig_CVPR18/backup/rectified2/'
    for idx in range(0,140):
        print(idx)
        # img_path = 'omni_00004_input.png'
        img_path = 'omni_'+str('%05d'%idx)+'_input.png'
        if os.path.isfile(path_img+img_path):
            img = cv2.imread(path_img+img_path)
            pred_phi,pred_theta = forward_classification_cart_cm(estimator,img)
            rect_img = rectify_image_given_phi_theta(img,pred_phi,pred_theta)
            save_path = to_save+str('%05d'%idx)+".png"
            cv2.imwrite(save_path,rect_img)