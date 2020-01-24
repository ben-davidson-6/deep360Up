from __future__ import division

from os.path import *

PROJECT_FILE_PATH = (dirname(dirname((dirname(abspath(__file__)))))) + "/"
CURRENT_FILE_PATH = dirname(abspath(__file__)) + "/"

import numpy as np
import CNN_train_package.fast_image_rotation as bp
import os


if __name__ == "__main__":


    # path to points
    pathToPoints=PROJECT_FILE_PATH+"data/10000_points.npy"
    spherePoints = np.load(pathToPoints).T
    numPoints = spherePoints.shape[0]

    folder_path = PROJECT_FILE_PATH + "data/"+"matrices221_"+str("%d"%numPoints)+"/"

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    h,w = 221,442
    phiList = []
    thetaList = []
    for i in range(0,numPoints):
        if i%100 == 0:
            print(i)
        x = spherePoints[i,0]
        y = spherePoints[i,1]
        z = spherePoints[i,2]
        # if np.rad2deg(math.acos(z)) <= 90:
        [phi, theta] = bp.cartesian_to_spherical(x, y, z)
        phiList.append(phi)
        thetaList.append(theta)
        rotate_map = bp.rotate_map_given_phi_theta(phi, theta, h, w)
        x_filename = folder_path + str('%05d' % i) + "_x.npy"
        y_filename = folder_path + str('%05d' % i) + "_y.npy"
        np.save(x_filename, rotate_map[0])
        np.save(y_filename, rotate_map[1])


    phiList = np.asarray(phiList)
    thetaList = np.asarray(thetaList)
    phiListName = folder_path+"phiList.npy"
    thetaListName = folder_path+"thetaList.npy"
    phiThetaName = folder_path+"phiTheta"+"_"+str('%d'%numPoints)+".npy"
    phiList = np.transpose(phiList)
    thetaList = np.transpose(thetaList)
    phiTheta = np.column_stack((phiList,thetaList))
    np.save(phiThetaName,phiTheta)
