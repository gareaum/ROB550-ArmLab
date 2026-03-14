"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials


TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""


import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
from scipy.optimize import fsolve


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]


    @param      angle  The angle


    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle




def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world


                TODO: implement this function


                Calculate forward kinematics for rexarm using DH convention


                return a transformation matrix representing the pose of the desired link


                note: phi is the euler angle about the y-axis in the base frame
                         thetoffset = np.array([0.09, 0.79, 3.08, 0.62])a]
    @param      joint_angles  The joint angles of the links
    @param
    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
           link          The link to transform from


    @return     a transformation matrix representing the pose of the desired link
    """
    # offset = np.array([0.09, 0.88, 3.52, 0.62, -0.09])  # Offset for each joint
    A_i = np.eye(4)
    for i in range(link):
        a, alpha, d, theta = dh_params[i]
        theta += joint_angles[i]
        A_i = np.dot(A_i, get_transform_from_dh(a, alpha, d, theta))
    return A_i






def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.


    TODO: Find the T matrix from a row of a DH table


    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians


    @return     The 4x4 transformation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ], dtype=np.float32)




def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.


                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)


    @param      T     transformation matrix


    @return     The euler angles from T.
    """
    pass
    R = T[:3, :3]
   
    a = np.arctan2(R[1, 2], R[0, 2])
    b = np.arctan2(np.sqrt(1 - R[2, 2]**2), R[2, 2])
    r = np.arctan2(R[2, 1], -R[2, 0])
       
    return a, b, r


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.


    TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T


    @param      T     transformation matrix


    @return     The pose vector from T.
    """
    xyz = T[:3, 3]
    y,p,r = get_euler_angles_from_T(T)
    pose = np.hstack((xyz, [y,p,r] ))
    return pose.tolist()

def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link


                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link


    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors


    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    # offset = np.array([0.09, 0.88, 3.52, 0.62, -0.09])  # Offset for each joint
    offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Offset for each joint
    T = np.eye(4)
    s_lst = s_lst.T
    for i in range(len(joint_angles)):
        w=s_lst[i, :3]
        v=s_lst[i, 3:]
        S_matrix = to_s_matrix(w,v)
        T = np.dot(T, expm(S_matrix * (joint_angles[i] - offset[i])))
    T= np.dot(T, m_mat)

    return T






def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.


    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)


    @param      w     { parameter_description }
    @param      v     { parameter_description }


    @return     { description_of_the_return_value }
    """
    E = np.array([
        [ 0,    -w[2],  w[1], v[0]],
        [ w[2],     0, -w[0], v[1]],
        [-w[1],  w[0],     0, v[2]],
        [0, 0, 0, 0]
    ])
    return E




def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.


                TODO: Convert a desired end-effector pose vector as np.array to joint angles


    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array


    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    print("dh_params:", dh_params)
    joint_configs = np.zeros((4, 5))
    print("pose:", pose)
    
    joint_configs[:, 0] = np.array([np.arctan2(pose[1], pose[0]),
                                    np.arctan2(pose[1], pose[0]),
                                    np.arctan2(pose[1], pose[0]) + np.pi,
                                    np.arctan2(pose[1], pose[0]) + np.pi])
    print("theta1:", joint_configs[:, 0])
    r = np.sqrt(pose[0]**2 + pose[1]**2)
    print("r:", r)
    s = np.sqrt(r**2 + pose[2]**2)
    print("s:", s)
    cos_theta3 = (r**2 + s**2 - dh_params[1][0]**2 - dh_params[2][0]**2) / (2 * dh_params[1][0] * dh_params[2][0])
    cos_theta3 = np.clip(cos_theta3, -1, 1)  # Clip to avoid numerical issues
    joint_configs[:, 2] = np.array([np.arccos(cos_theta3) - dh_params[2][3],
                                    -np.arccos(cos_theta3) - dh_params[2][3],
                                    np.arccos(cos_theta3) - dh_params[2][3],
                                    -np.arccos(cos_theta3) - dh_params[2][3]])
    print("theta3:", joint_configs[:, 2])
    joint_configs[:, 1] = np.array([np.arctan2(s, r) - np.arctan2(dh_params[2][0] * np.sin(joint_configs[0, 2]), dh_params[1][0] + dh_params[2][0] * np.cos(joint_configs[0, 2])),
                                    np.arctan2(s, r) - np.arctan2(dh_params[2][0] * np.sin(joint_configs[1, 2]), dh_params[1][0] + dh_params[2][0] * np.cos(joint_configs[1, 2])),
                                    np.arctan2(s, r) - np.arctan2(dh_params[2][0] * np.sin(joint_configs[2, 2]), dh_params[1][0] + dh_params[2][0] * np.cos(joint_configs[2, 2])),
                                    np.arctan2(s, r) - np.arctan2(dh_params[2][0] * np.sin(joint_configs[3, 2]), dh_params[1][0] + dh_params[2][0] * np.cos(joint_configs[3, 2]))])
    print("theta2:", joint_configs[:, 1])


def IK_geometric2(dh_params: np.ndarray, pose: np.ndarray) -> np.ndarray:
    # Unpack pose
    x, y, z, phi = map(float, pose)
    l1 = float(dh_params[0][2])
    l2 = float(dh_params[1][0])
    l3 = float(dh_params[2][0])
    l4 = float(dh_params[4][2])


    theta1 = np.arctan2(-x, y)
   
    xc, yc, zc = pose[0:3] - l4 * np.array([-np.sin(theta1)*np.cos(phi), np.cos(theta1)*np.cos(phi), -np.sin(phi)], dtype=np.float32)


    r = np.sqrt(xc * xc + yc * yc)  
    s = zc - l1


    theta3 = - np.arccos((r * r + s * s - l2 * l2 - l3 * l3)/(2 * l2 * l3)) + np.pi/2 - np.arctan2(50, 200)
    theta2 = - np.arctan2(s, r) - np.arctan2(l3*np.sin(theta3), l2 + l3*np.cos(theta3)) + np.pi/2 - np.arctan2(50, 200)
    theta4 = (theta2 + theta3) - phi
    theta5 = 0 #TODO


    if np.sqrt(xc*xc + yc*yc + (zc - l1)*(zc - l1)) > (l2 + l3):
        return [0, 0, 0, 0, 0]
    return [theta1, theta2, theta3, theta4, theta5]
   
def EQ_numerical_IK(theta, xd, yd, zd, pitch):
    l0, l1, l2, l3 = 103.91, 206.155281281, 200.0, 174.15


    #Goal
    objectives = [np.sqrt(xd**2 + yd**2), zd, pitch]


    #Kinematics
    K1 = (
        l1 * np.cos(-theta[0] + np.arctan(4))
        + l2 * np.cos(-(theta[0] + theta[1]))
        + l3 * np.cos(-(theta[0] + theta[1] + theta[2]))
    )


    K2 = (
        l0
        + l1 * np.sin(-theta[0] + np.arctan(4))
        + l2 * np.sin(-(theta[0] + theta[1]))
        + l3 * np.sin(-(theta[0] + theta[1] + theta[2]))
    )


    K3 = theta[0] + theta[1] + theta[2]


    #COST
    return [K1 - objectives[0], K2 - objectives[1], K3 - objectives[2]]




def IK_numerical(target_position, initial_theta):


    xd, yd, zd, roll, pitch = target_position


    solve = fsolve(
        EQ_numerical_IK, initial_theta, args=(xd, yd, zd, pitch)
    )


    # Base rotation
    theta1 = -np.arctan2(xd, yd)


    # End-effector rotation (placeholder)
    theta5 = theta1 + roll
    #theta5 = 0.0


    # Recursion if solution is outside range
    if np.max(np.abs(np.array([theta1, solve[0], solve[1], solve[2], theta5]))) < 3.14:
        return np.array([theta1, solve[0], solve[1], solve[2], theta5])
    else:
        new_target = target_position - np.array([1, 1, 1, 10, 0])
        return IK_numerical(new_target, initial_theta)






def IK_error(dh_params, pose):
    x, y, z, theta, psi = pose


    l1 = dh_params[0][2]
    l2 = dh_params[1][0]
    l3 = dh_params[2][0]
    l4 = dh_params[4][2]
    l5 = dh_params[3][0]


    Lbase = 100  # Base radius
    point_length = np.sqrt(x**2 + y**2 + (z - l1)**2)
    arm_length = l1 + l2 + l3 + l4 + l5


    if((point_length > arm_length) or (point_length < Lbase) or (z < 0) or (z > arm_length)):
        return True
    else:
        return False
   

def IK_geometric3(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.


                TODO: Convert a desired end-effector pose vector as np.array to joint angles


    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array


    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    joint_configs = np.zeros((4, 5))


    if(IK_error(dh_params, pose)):
        print("Pose is unreachable")
        return [0, 0, 0, 0, 0]


    x, y, z, theta, psi = pose
    l1 = dh_params[0][2]
    l2 = dh_params[1][0]
    l3 = dh_params[2][0]
    l4 = dh_params[3][0]
    l5 = dh_params[4][2]
    theta1_offset = dh_params[0][3]
    angle_offset = (np.pi/2) - np.arctan2(200, 50) #14.036243467926477


    theta1 = np.array([np.arctan2(y, x) - theta1_offset,
                       np.arctan2(y, x) + np.pi - theta1_offset])


    r = np.sqrt(x**2 + y**2)
    s = z - l1

    w_r = r - l5 * np.cos(theta)
    w_s = s + l5 * np.sin(theta)

    cos_theta3 = (w_r**2 + w_s**2 - l2**2 - l3**2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1, 1)


    theta3 = np.array([np.arccos(cos_theta3) - (np.pi/2 - angle_offset),
                       -np.arccos(cos_theta3) - (np.pi/2 - angle_offset)])

    theta2 = np.zeros(2)
    theta4 = np.zeros(2)
    for i in range(len(theta3)):
        theta2[i] = np.arctan2(w_s, w_r) - np.arctan2(l3 * np.sin(theta3[i]), l2 + l3 * np.cos(theta3[i]))
        theta2[i] -= angle_offset


        theta4[i] = theta - (theta2[i] + theta3[i])

    joint_configs = np.array([[theta1[0], theta2[0], theta3[0], theta4[0], psi],
                              [theta1[0], theta2[1], theta3[1], theta4[1], psi],
                              [theta1[1], theta2[0], theta3[0], theta4[0], psi],
                              [theta1[1], theta2[1], theta3[1], theta4[1], psi]])


    ik_angle = checkIKAngles_2(dh_params, pose, joint_configs)


    ik_angle_deg = []
    for angle in ik_angle:
        ik_angle_deg.append(angle * 180.0 / np.pi)
    print("ik_angle:", ik_angle_deg)


    return ik_angle


def checkIKAngles_2(dh_params, pose, joint_configs):
    smallest_dist = 99999
    corr_angle = [joint_configs[0][0], joint_configs[0][1], joint_configs[0][2], joint_configs[0][3], joint_configs[0][4]]


    for i in range(4):
        if (((-180/180)*np.pi <= joint_configs[i][0] <= (180/180)*np.pi) and
            ((-108/180)*np.pi <= joint_configs[i][1] <= (113/180)*np.pi) and
            ((-108/180)*np.pi <= joint_configs[i][2] <= (93/180)*np.pi) and
            ((-100/180)*np.pi <= joint_configs[i][3] <= (123/180)*np.pi)):
            potential_angle = [joint_configs[i][0], joint_configs[i][1], joint_configs[i][2], joint_configs[i][3], joint_configs[i][4]]
            fk = FK_dh(dh_params, potential_angle, 5)
            fk_position = fk[:3, 3]
            desired_position = np.array(pose[:3])
            distance = np.linalg.norm(fk_position - desired_position)

            if distance < smallest_dist:
                corr_angle = potential_angle
                smallest_dist = distance
    return corr_angle


def checkIKAngles_1(dh_params, pose, theta1, theta2, theta3):
    smallest_dist = 99999
    # corr_angle = [0, 0, 0, 0, 0]
    corr_angle = [theta1[0], theta2[0], theta3[0], pose[3], pose[4]]


    for i in range(len(theta1)):
        for j in range(len(theta2)):
            for k in range(len(theta3)):
                theta4 = pose[3] - (theta2[j] + theta3[k])
                if (((-180/180)*np.pi <= theta1[i] <= (180/180)*np.pi) and
                    ((-108/180)*np.pi <= theta2[j] <= (113/180)*np.pi) and
                    ((-108/180)*np.pi <= theta3[k] <= (93/180)*np.pi) and
                    ((-100/180)*np.pi <= theta4 <= (123/180)*np.pi)):
                    potential_angle = [theta1[i], theta2[j], theta3[k], theta4, pose[4]]
                    # print("potential_angle (deg):", [angle * 180.0 / np.pi for angle in potential_angle])
                    fk = FK_dh(dh_params, potential_angle, 5)
                    fk_position = fk[:3, 3]
                    desired_position = np.array(pose[:3])
                    distance = np.linalg.norm(fk_position - desired_position)
                    # print(distance)


                    if distance < smallest_dist:
                        corr_angle = potential_angle
                        smallest_dist = distance
    return corr_angle
