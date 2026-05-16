# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:22:36 2020

Covariance estimate for results given by openMVG bundle adjustment.
One extra iteration is performed on the results, thus giving us the ability to 
compute covariance for parameters and 
@author: zdang2
"""
import sfm_IO as IO
import numpy as np
import math as m
from scipy.linalg import inv
from numpy.linalg import multi_dot


<<<<<<< HEAD
root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = IO.read_sfm('sfm.json', 'D:/SFM/covestimator/Data/')
=======
root_path, extrinsics, intrinsics, coords_3d, views_meta, control_points = read_sfm('sfm.json', 'D:/Documents/Research/covestimator/Data/')
>>>>>>> 4b61234516656f6b3d70d9cd72cd0d60a0de8ff5


# 3-D coordinates corresponding to 2-D image observations


def initParams(views_meta, coords_3d, extrinsics, intrinsics):
    """
    This function initializes bundle adjustment parameters and data inputs.
    Returns: Design matrices Ae, Ao, EO initial values (XYZc, rotation)
    2D image observations (merged_obs), 3D object point coordinates (XYZ_3D)
    """
    img_obs = []
    n_imgs = len(views_meta)

<<<<<<< HEAD
    # XYZ_3D - flattened 3D object point coordinates
    XYZ_3D = np.empty((0, 0))
    for i in range(0, len(coords_3d)):
        coords_3d[i]['index_3D'] = i
        XYZ_3D = np.append(XYZ_3D, np.array(coords_3d[i]['X']))

    for i in range(0, len(coords_3d)):
        pnt_3d = coords_3d[i]['observations']
        index_3d = coords_3d[i]['index_3D']  
        for img in pnt_3d:
            img.update({"index_3D": index_3d})
        img_obs.extend(pnt_3d)


    # list of 2-D image observations for all images
    # merged_obs is a list with length of (n_imgs), each value for each image
    # corresponds to number of observations for that image, (x, y, index of the 3D
    # point which can be found in coords_3d)
    merged_obs = []
    obs_count = []
    for i in range(0, n_imgs):
        temp = []
        for d in range(0, len(img_obs)):
            a = []
            b = []
            keyID = img_obs[d]['key']
            if keyID == i:
                a = img_obs[d]['value']['x']
                b = img_obs[d]['index_3D']
                a.append(b)
                temp.append(a)
        obs_count.append(len(temp))
        merged_obs.append(temp)
=======
# list of 2-D image observations for all images
    
merged_obs = []
for i in range(0, n_imgs):
    temp = np.array([])
    a = []
    b = 0
    for d in range(0, len(img_obs)):
        keyID = img_obs[d]['key']
        if keyID == i:
            a = img_obs[d]['value']['x']
            b = img_obs[d]['index_3D']
            c = np.array(a.append(b))
            temp = np.append(temp, c)
    merged_obs.append(temp)


            
    
# 3-D coordinates corresponding to 2-D image observations
>>>>>>> 4b61234516656f6b3d70d9cd72cd0d60a0de8ff5


    # Number of 2D image observations
    n = 2 * sum(obs_count) 
    
    # Number of exterior parameters
    ue = 6 * n_imgs
    
    # Number of 3D object points
    uo = len(coords_3d)
    
    # Exterior design matrix
    Ae = np.empty([n, ue])
    
    # Object point design matrix
    Ao = np.empty([n, uo])
    
    # dxe, dxo
    dxe = np.ones((ue, 1))
    
    dxo = np.ones((uo, 1))
    
    # Exterior - Xc, Yc, Zc, Rotation
    XYZc = []
    rotation = []
    for i in range(0, n_imgs):
        XYZc.append(extrinsics[i]['center'])
        rotation.append(extrinsics[i]['rotation'])
    
    # Intrinsics unpack
    interior = intrinsics['ptr_wrapper']['data']
        
    return Ae, Ao, dxe, dxo, XYZc, rotation, merged_obs, XYZ_3D, interior

def apply_distortion(interior, transformed_3D_point):
    k1, k2, k3 = interior['disto_k3']
    c = interior['focal_length']
    xp, yp = interior['principal_point']
    
    # 3D homogeneous -> 2D euclidean
    projected_point = transformed_3D_point[0:2] / transformed_3D_point[-1]
    
    # distortion
    r2 = np.inner(projected_point, projected_point)
    r4 = r2 * r2
    r6 = r4 * r2
    r_coeff = (1 + k1 * r2 + k2 * r4 + k3 * r6)
    return xp, yp, projected_point, r_coeff, c
    
        
def cost_function(intrinsics, R, XYZc, XYZ_3D, interior, obs_2D):
    """
    This evaluates the image x, y based on collinearity.
    Input: camera parameters, exterior orientation parameters, points' coordinates
    in real world frame.
    Return: x, y on the image (pixel)
    """    
    transformed_3D_point = np.dot(R, XYZ_3D) + XYZc
    xp, yp, projected_point, r_coeff, c = apply_distortion(interior, transformed_3D_point)
    
    # collinearity equations
    # x = xp - c*(m11*(X - Xc) + m12*(Y - Yc) + m13*(Z - Zc))/(m31*(X - Xc) + m32*(Y - Yc) + m33*(Z - Zc))
    # y = yp - c*(m21*(X - Xc) + m22*(Y - Yc) + m23*(Z - Zc))/(m31*(X - Xc) + m32*(Y - Yc) + m33*(Z - Zc))
    residuals_x = xp + projected_point[0] * r_coeff * c - obs_2D[0]
    residuals_y = yp + projected_point[1] * r_coeff * c - obs_2D[1]
    return residuals_x, residuals_y
        

    
    