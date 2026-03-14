"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from kinematics import *

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.calibration_done = False
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """

        # IMPORTANT: This function runs in a loop. If you make a new state, it will be run every iteration.
        #            The function (and the state functions within) will continuously be called until the state changes.

        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate_pnp":
            self.calibrate_pnp()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "record":
            self.record()

        if self.next_state == "custom_execute":
            self.custom_execute()

        if self.next_state == "pick":
            self.pick()

        if self.next_state == "drop":
            self.drop()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"

        ## iterate throught poses
        for waypoint in self.waypoints:
            self.rxarm.set_positions(waypoint)
            time.sleep(5)

        self.next_state = "idle"
    
    def xyz_from_uvd(self, uvd):
        click_xyz = self.camera.pixel_to_world(uvd).flatten().tolist()
        return click_xyz


    def pick(self):
        """!
        @brief      Pick function to pick up a block
        """
        self.status_message = "State: Pick - Picking up block"
        self.current_state = "pick"
        dh_params = self.rxarm.get_dh_parameters()

        self.camera.new_click = False
        while not self.camera.new_click:
            time.sleep(0.1)

        self.camera.new_click = False
        pt = self.camera.last_click
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)

        target_world_pos = self.xyz_from_uvd(click_uvd)

        pose1 = [target_world_pos[0], target_world_pos[1], target_world_pos[2], np.pi/2]
        sleep_time = 3

        target_position = np.array([(pose1[0]), (pose1[1]), (pose1[2] + 150), (pose1[3]), np.pi/2])
        init_angle = self.rxarm.get_positions()
        target_angle = IK_numerical(target_position, init_angle[1:4])
        self.rxarm.set_positions(target_angle)
        time.sleep(sleep_time)

        target_position = np.array([(pose1[0]), (pose1[1]), (pose1[2] - 15), (pose1[3]), np.pi/2])
        init_angle = self.rxarm.get_positions()
        target_angle = IK_numerical(target_position, init_angle[1:4])
        self.rxarm.set_positions(target_angle)
        time.sleep(sleep_time)
        self.rxarm.gripper.grasp()
        time.sleep(0.5)

        target_position = np.array([(pose1[0]), (pose1[1]), (pose1[2] + 150), (pose1[3]), np.pi/2])
        init_angle = self.rxarm.get_positions()
        target_angle = IK_numerical(target_position, init_angle[1:4])
        self.rxarm.set_positions(target_angle)
        time.sleep(sleep_time)

        point = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.rxarm.set_positions(point)
        time.sleep(sleep_time)

        if not self.next_state == "estop":
            self.next_state = "idle"

    def drop(self):
        """!
        @brief      Drop function to drop a block
        """
        self.status_message = "State: Drop - Dropping block"
        self.current_state = "drop"

        dh_params = self.rxarm.get_dh_parameters()

        self.camera.new_click = False
        while not self.camera.new_click:
            time.sleep(0.1)

        self.camera.new_click = False
        pt = self.camera.last_click
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)

        target_world_pos = self.xyz_from_uvd(click_uvd)

        pose1 = [target_world_pos[0], target_world_pos[1], target_world_pos[2], np.pi/2]
        sleep_time = 3

        target_position = np.array([(pose1[0]), (pose1[1]), (pose1[2] + 150), (pose1[3]), np.pi/2])
        init_angle = self.rxarm.get_positions()
        target_angle = IK_numerical(target_position, init_angle[1:4])
        self.rxarm.set_positions(target_angle)
        time.sleep(sleep_time)

        target_position = np.array([(pose1[0]), (pose1[1]), (pose1[2] + 0), (pose1[3]), np.pi/2])
        init_angle = self.rxarm.get_positions()
        target_angle = IK_numerical(target_position, init_angle[1:4])
        self.rxarm.set_positions(target_angle)
        time.sleep(sleep_time)
        self.rxarm.gripper.release()
        time.sleep(0.5)

        target_position = np.array([(pose1[0]), (pose1[1]), (pose1[2] + 150), (pose1[3]), np.pi/2])
        init_angle = self.rxarm.get_positions()
        target_angle = IK_numerical(target_position, init_angle[1:4])
        self.rxarm.set_positions(target_angle)
        time.sleep(sleep_time)

        point = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.rxarm.set_positions(point)

        if not self.next_state == "estop":
            self.next_state = "idle"


    def custom_execute(self):
        self.status_message = "State: Custom Execute - Executing cutsom motion plan"
        self.current_state = "custom_execute"
        counter = 0
        open_steps = [4, 10, 16]
        close_steps = [1,7, 13]

        count = 0
        try:
            while count < 2:
                with open("recorded_pose.txt", "r") as f:
                    lines = f.readlines()
                    for line in lines:

                        pose = line.strip().strip('[]').split()
                        pose = [float(angle) for angle in pose]
                        self.status_message = "Executing pose: " + str(pose)
                        self.rxarm.set_positions(pose)
                        time.sleep(3)

                        if counter in close_steps:
                            self.rxarm.gripper.grasp()
                            time.sleep(1)

                        if counter in open_steps:
                            self.rxarm.gripper.release()
                            time.sleep(1)
                        
                        counter += 1
                counter = 0
                count += 1

            self.next_state = "idle"
        except:
            self.status_message = "Error: Could not read recorded_pose.txt"
            self.next_state = "idle"

    def record(self):
        self.status_message = "State: Record - recording position"
        self.current_state = "record"

        desired_pose = self.rxarm.get_positions()

        with open("recorded_pose.txt", "a") as f:
            f.write(str(desired_pose) + "\n")

        self.next_state = "idle"


    def find_apriltags(self): #Returns corners and centers of detected april tags
        '''
        Find AprilTags in the current camera frame and return their corners and centers
        '''
        aprilTags_corners = []
        aprilTags_centers = []

        for detection in self.camera.tag_detections.detections:
            corners = []

            # Puts corners in a list
            for corner in detection.corners:
                x = int(corner.x)
                y = int(corner.y)
                corners.append((x, y))

            aprilTags_corners.append(corners)
            
            # centers x, y (aprile tags)
            cx = int(detection.centre.x)
            cy = int(detection.centre.y)

            aprilTags_centers.append((cx, cy))

        return (np.array(aprilTags_corners, dtype=np.float32), 
                np.array(aprilTags_centers, dtype=np.float32))


    def recover_homogenous_transform_pnp(self, image_points, world_points, K, D): #Recovers extrinsic matrix using PnP
        '''
        Use SolvePnP to find the rigidbody transform representing the camera pose in
        world coordinates
        '''
        distCoeffs = D
        [_, R_exp, t] = cv2.solvePnP(world_points,
                                    image_points,
                                    K,
                                    distCoeffs,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(R_exp)
        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))

    def homography_transform(self,aprilTags_centers): #Computes homography matrix
        '''
        Use homography to find the perspective transform matrix
        '''
        src_pts = np.array(aprilTags_centers)
        
        cx = 640
        cy = 360 

        dx = 250
        dy = dx/5 * 3 
        offset_y = 0

        dest_pts = np.array([
            [cx - dx, cy + dy + offset_y],
            [cx + dx, cy + dy + offset_y],
            [cx + dx, cy - dy + offset_y],
            [cx - dx, cy - dy + offset_y]
        ], dtype=np.float32)


        H, _ = cv2.findHomography(src_pts.astype(np.float32), dest_pts.astype(np.float32))
        
        return H

    def calibrate_pnp(self):
        self.current_state = "calibrate_pnp"

        K = self.camera.intrinsic_matrix
        D = np.array([0.15564486384391785, -0.48568257689476013, -0.0019681642297655344, 0.0007267732871696353, 0.44230175018310547], dtype=np.float32)

        aprilTags_corners, aprilTags_centers = self.find_apriltags() #Get corners and centers of detected april tags
        image_pts = aprilTags_centers
        world_pts = np.array(self.camera.tag_locations, dtype=np.float32)


        # append a z=0 coordinate to world points
        world_pts = np.hstack((world_pts, np.zeros((world_pts.shape[0], 1), dtype=np.float32)))
        
        print("Image Points:\n", image_pts)
        print("World Points:\n", world_pts)
        print("Image pts shape:", image_pts.shape)
        print("World pts shape:", world_pts.shape)

        try:
            A_pnp = self.recover_homogenous_transform_pnp(image_pts, world_pts, K, D) #Get extrinsic matrix using pnp

            self.camera.extrinsic_matrix_pnp = A_pnp #Set extrinsic matrix in camera class
            
            homography_matrix = self.homography_transform(aprilTags_centers) #Get homography matrix
            self.camera.H = homography_matrix #Set homography matrix in camera class

            self.status_message = "Calibration - Completed"

            print("Extrinsic Matrix from PnP:\n", A_pnp)
            print("Extrinsic Matrix shape:", A_pnp.shape)
            print("Homography Matrix:\n", homography_matrix)
            print("Homography Matrix shape:", homography_matrix.shape)
            self.calibration_done = True
            self.camera.camera_calibrated = True

        except:
            self.status_message = "Calibration - Failed: Could not compute extrinsics"
            print("Calibration failed: Could not compute extrinsics")

        self.next_state = "idle"

    
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)