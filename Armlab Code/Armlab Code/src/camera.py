#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        # TODO: Remove
        self.temp = 0

        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)

        # calibration
        self.H = None


        # mouse clicks & calibration variables
        self.camera_calibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.extrinsic_matrix_pnp = np.eye(4)
        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def pixel_to_world(self, uvz):
        offset = [0, 0, 0]
        index = np.array([uvz[0], uvz[1], 1]).reshape((3, 1))
        pos_camera = uvz[2] * np.matmul(np.linalg.inv(self.intrinsic_matrix), index)
        temp_pos = np.array([pos_camera[0][0] + offset[0], pos_camera[1][0] + offset[1], pos_camera[2][0] + offset[2], 1]).reshape((4, 1))
        pos_world = np.matmul(np.linalg.inv(self.extrinsic_matrix), temp_pos)
        return pos_world.flatten()[:3]

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)   


    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        if not self.camera_calibrated:
            print("Camera not calibrated.")
            return

        

        # # 1. SETUP & TRANSFORMATIONS
        # rgbImage = self.VideoFrame.copy()
        # depthImage = self.DepthFrameRaw
        # H, W = depthImage.shape
        
        # # Pre-define color library for vectorized search
        # # (Move this to __init__ if you want maximum efficiency)
        # color_lib = [
        #     ('red', (19, 15, 110)), ('red', (46, 23, 183)), ('red', (62, 41, 134)), 
        #     ('red', (64, 54, 187)), ('red', (52, 40, 86)), ('orange', (25, 80, 170)),
        #     ('orange', (63, 90, 209)), ('orange', (58, 75, 173)), ('orange', (23, 68, 150)),
        #     ('yellow', (30, 150, 200)), ('green', (89, 124, 48)), ('green', (118, 131, 83)),
        #     ('green', (74, 85, 25)), ('green', (48, 74, 33)), ('blue', (116, 83, 38)),
        #     ('blue', (110, 64, 2)), ('violet', (118, 70, 52)), ('violet', (100, 40, 80)),
        #     ('violet', (47, 35, 40)), ('violet', (72, 57, 65))
        # ]
        # labels, targets = zip(*color_lib)
        # targets = np.array(targets)

        # # Vectorized 3D Projection (Same math as yours, just faster)
        # u, v = np.meshgrid(np.arange(W), np.arange(H))
        # pixels = np.stack([u.ravel(), v.ravel(), np.ones(H*W)], axis=0)
        
        # camera_points = (np.linalg.inv(self.intrinsic_matrix) @ pixels) * depthImage.ravel()
        # world_pts = np.linalg.inv(self.extrinsic_matrix) @ np.vstack([camera_points, np.ones(H*W)])
        
        # # Reshape and Warp
        # world_grid = (world_pts[:3] / world_pts[3]).T.reshape(H, W, 3)
        # world_grid = cv2.warpPerspective(world_grid, self.H, (W, H))
        # z_map = world_grid[:, :, 2]

        # # 2. MASKING
        # mask = np.zeros((H, W), dtype=np.uint8)
        # cv2.rectangle(mask, (119, 0), (1147, 713), 255, -1)
        # cv2.rectangle(mask, (510, 363), (720, 713), 0, -1)
        
        # # Identify potential blocks by height range
        # detection_mask = cv2.bitwise_and(cv2.inRange(z_map, 13, 140), mask)
        # contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # 3. DETECTION LOOP
        # self.block_detections = []
        # for cnt in contours:
        #     if cv2.contourArea(cnt) < 100: continue
            
        #     # Shape Analysis
        #     (rcx, rcy), (bw, bh), theta = cv2.minAreaRect(cnt)
        #     if max(bw, bh) == 0 or abs(bw - bh) / max(bw, bh) > 0.2: continue

        #     # Center Coordinates
        #     ix, iy = int(np.clip(rcx, 0, W-1)), int(np.clip(rcy, 0, H-1))
        #     world_xyz = world_grid[iy, ix]

        #     # Vectorized Color Matching (Replaces your get_color function)
        #     c_mask = np.zeros((H, W), dtype=np.uint8)
        #     cv2.drawContours(c_mask, [cnt], -1, 255, -1)
        #     mean_bgr = np.array(cv2.mean(rgbImage, mask=c_mask)[:3])
            
        #     # Calculate Euclidean distance to all targets at once
        #     dist = np.linalg.norm(targets - mean_bgr, axis=1)
        #     color_name = labels[np.argmin(dist)]

        #     # Store Results
        #     self.block_detections.append([*world_xyz, theta, bw*bh, color_name])

        #     # Visuals
        #     cv2.drawContours(rgbImage, [cnt], -1, (0, 255, 255), 1)
        #     cv2.putText(rgbImage, color_name, (ix-30, iy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        # self.DetectFrame = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2BGR)

        # self.VideoFrame = self.DetectFrame.copy()


        # 0. Create a board mask
        board_mask = np.zeros(self.VideoFrame.shape[:2], dtype=np.uint8)

        board_mask_uv = np.array([[150, 10], [1130, 650]]) # upper left, bottom right
        arm_mask_uv = np.array([[550, 310], [723, board_mask_uv[1][1]]]) # upper left, bottom right

        # fill mask
        cv2.rectangle(board_mask, board_mask_uv[0], board_mask_uv[1], 255, cv2.FILLED)
        cv2.rectangle(board_mask, arm_mask_uv[0], arm_mask_uv[1], 0, cv2.FILLED)

        # draw mask on image
        # self.VideoFrame = cv2.bitwise_and(self.VideoFrame, self.VideoFrame, mask=board_mask)


        # 1. Load the image
        if self.temp == 0:
            # save the depth image to a file:
            if self.H is not None:
                modified_image = cv2.warpPerspective(
                    self.DepthFrameRaw, self.H,
                    (self.DepthFrameRaw.shape[1], self.DepthFrameRaw.shape[0]))
            cv2.imwrite("raw_depth2.png", modified_image)

            # save the numpy array
            np.save("raw_depth2.npy", modified_image)
            
            self.temp = 1

        img = self.VideoFrame.copy()

        # 2. white balance the image (normalize each channel by its mean, then clip to 255)
        img_board_only = cv2.bitwise_and(img, img, mask=board_mask)
        img_board_only_temp = np.ma.masked_equal(img_board_only, [0, 0, 0]) # mask out zeros for mean calculation
        
        mean_channels = np.ma.mean(img_board_only_temp, axis=(0, 1)) # majority of board grey, should work
        mean_grey = np.mean(mean_channels)
        channel_scale_factor = mean_channels / mean_grey

        img_bal = img.copy()
        for n in range(3):
            img_bal[:, :, n] = np.clip(img_bal[:, :, n] / channel_scale_factor[n], 0, 255)
        

        # just look at img_bal
        # self.TagImageFrame = img_bal 


        # 3. boost img brightness, mask block color
        # norm/brighten colors in image
        img_norm = np.zeros(np.shape(img)) # , dtype=np.uint8
        norm_factor_channel = np.mean(img_bal, axis=(0, 1)) # change to img_bal (white balance image)

        for i in range(3): # 3 channels
            img_norm[:,:, i] = img[:, :, i]/norm_factor_channel[i] * 255
        
        img_norm = np.clip(img_norm, 1, 255) # , dtype=np.uint8
        img_norm = np.array(img_norm, dtype=np.uint8)

        # mask by sat*val
        kernsize = 7

        img_blurred = cv2.GaussianBlur(img_norm, (kernsize, kernsize), 0)
        img_norm_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_RGB2HSV)

        img_mask_sat = cv2.inRange(img_norm_hsv[:, :, 1], 80, 255) # 80 subject to tuning
        img_mask_val = cv2.inRange(img_norm_hsv[:, :, 2], 100, 255)
        
        hue_mask = img_mask_sat * img_mask_val * 255
        hue_mask = cv2.bitwise_and(hue_mask, board_mask) # only look at blocks on board

        # vv run through depth mask to filter april tags vv

        img_masked = np.zeros(np.shape(img_blurred), dtype=np.uint8)

        for i in range(3):
            img_masked[:, :, i] = cv2.bitwise_and(img_norm[:, :, i], hue_mask, mask=None)

        # self.TagImageFrame = img_masked 
        # self.TagImageFrame = cv2.cvtColor(hue_mask, cv2.COLOR_GRAY2RGB)


        # 4. ID and draw around blocks
        
        # block by col, draw around blocks
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_red = np.array([0, 20])
        upper_red = np.array([160, 179])

        color_ranges = {
            'orange': [0, 29, [255, 150, 0]],
            'yellow': [30, 49, [255, 255, 0]],
            'green':  [50, 85, [0, 255, 0]],
            'purple': [126, 149,[150, 0, 150]],
            'red':    [150, 179, [255, 0, 0]],
            'blue':   [86, 125, [0, 0, 255]],
        }

        # check img_masked for color
        # draw on img_bal

        for name, (hue_low, hue_high, rgb_color) in color_ranges.items():
            # print(name)
            # print(rgb_color)
            img_masked_hsv = cv2.cvtColor(img_masked, cv2.COLOR_RGB2HSV)
            hue_mask_specific = cv2.inRange(img_masked_hsv[:, :, 0], np.array(hue_low), np.array(hue_high))
            hue_mask_specific = hue_mask_specific * cv2.inRange(img_masked_hsv[:, :, 1], np.array(50), np.array(255)) * 255 # sat clip (everything is orange problem)

            # hue_mask_specific = cv2.bitwise_and(hue_mask_specific, hue_mask) # combine with overall mask, orange
            
            # self.TagImageFrame = cv2.cvtColor(hue_mask_specific, cv2.COLOR_GRAY2RGB)
            
            # Clean the mask: Remove small dots and bridge small gaps in the blocks
            # kernel = np.ones((5,5), np.uint8)
            # hue_mask_specific = cv2.morphologyEx(hue_mask_specific, cv2.MORPH_OPEN, kernel) 
            # hue_mask_specific = cv2.morphologyEx(hue_mask_specific, cv2.MORPH_CLOSE, kernel)

            # Find blocks
            contours, _ = cv2.findContours(hue_mask_specific, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # if cv2.contourArea(cnt) > 300: # Adjust if tiny blocks are missed (300)
                if np.abs(w - h) < 100 and cv2.contourArea(cnt) > 300: # Filter out non-square contours (adjust threshold as needed)

                    # cv2.drawContours(img_masked, [cnt], -1, [200, 200, 255], 1) # img_bal
                    # draws straight rect, ignoring orientation
                    # x, y, w, h = cv2.boundingRect(cnt) # 
                    cv2.rectangle(img_bal, (x, y), (x+w, y+h), rgb_color, 3)
                    cv2.putText(img_bal, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb_color, 2)
        
        self.TagImageFrame = img_bal

        # red blocks
        # print("Red mask shape:", img_masked_hsv[:, :, 0].shape)
        # print("Lower red:", lower_red.shape, lower_red[0], lower_red[1])
        # print("Upper red:", upper_red.shape, upper_red[0], upper_red[1])
        # # print (np.shape(cv2.inRange(img_masked_hsv[:, :, 0], np.array(lower_red[0]), np.array(lower_red[1]))))
        # # print (np.shape(cv2.inRange(img_masked_hsv[:, :, 0], upper_red[0], upper_red[1])))

        # red_mask = cv2.bitwise_or(  cv2.inRange(img_masked_hsv[:, :, 0], np.array(lower_red[0]), np.array(lower_red[1]) ), 
        #                             cv2.inRange(img_masked_hsv[:, :, 0], np.array(upper_red[0]), np.array(upper_red[1])))
        # # red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        # contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in contours:
        #     if cv2.contourArea(cnt) > 300:
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         cv2.rectangle(img_bal, (x, y), (x+w, y+h), (0, 0, 255), 3)
        #         cv2.putText(img_bal, 'red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
       
        # output onto img_bal

        # self.VideoFrame = output



    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        if self.H is not None:
            modified_image = cv2.warpPerspective(
                self.DepthFrameRaw, self.H,
                (self.DepthFrameRaw.shape[1], self.DepthFrameRaw.shape[0]))
            self.DepthFrameRaw = modified_image

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        if self.H is not None:
            extrinsic = self.extrinsic_matrix_pnp
            intrinsic = self.intrinsic_matrix

            points = self.grid_points

            points = np.stack([points[0].ravel(), points[1].ravel(), np.zeros(points[0].size)], axis=-1)

            homo = np.c_[points, np.ones(len(points))]
            cam_coords = (extrinsic @ homo.T).T[:, :3]
            pixel_coords = (intrinsic @ cam_coords.T).T

            temp = pixel_coords[:, :2] / pixel_coords[:, 2:3]
            
            homo = np.c_[temp, np.ones(len(temp))]
            transformed = (self.H @ homo.T).T

            temp = transformed[:, :2] / transformed[:, 2:3]

            modified_image = self.VideoFrame.copy()
            # Write your code here

            for point in temp:
                x = int(point[0])
                y = int(point[1])
                cv2.circle(modified_image, (x, y), 5, (0, 255, 0), -1)

            self.GridFrame = modified_image

        else:
            extrinsic = self.extrinsic_matrix
            intrinsic = self.intrinsic_matrix

            points = self.grid_points

            points = np.stack([points[0].ravel(), points[1].ravel(), np.zeros(points[0].size)], axis=-1)

            homo = np.c_[points, np.ones(len(points))]
            cam_coords = (extrinsic @ homo.T).T[:, :3]
            pixel_coords = (intrinsic @ cam_coords.T).T

            temp = pixel_coords[:, :2] / pixel_coords[:, 2:3]

            modified_image = self.VideoFrame.copy()
            # Write your code here

            for point in temp:
                if np.abs(point[0]) > 2000 or np.abs(point[1]) > 2000:
                    continue
                
                x = int(point[0])
                y = int(point[1])
                cv2.circle(modified_image, (x, y), 5, (0, 255, 0), -1)

            self.GridFrame = modified_image

            if self.H is not None:
                modified_image = cv2.warpPerspective(self.DepthFrameRGB, self.H, (self.DepthFrameRGB.shape[1], self.DepthFrameRGB.shape[0]))
                self.DepthFrameRGB = modified_image

     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """

        
        modified_image = self.VideoFrame.copy()
        # Write your code here

        for detection in msg.detections:
            corners = []

            # Puts corners in a list
            for corner in detection.corners:
                x = int(corner.x)
                y = int(corner.y)
                corners.append((x, y))
            
            # draw tag outline
            cv2.polylines(modified_image,
                          [np.array(corners)],
                          isClosed=True,
                          color=(0, 255, 0),
                          thickness=2)
            
            # draw center
            cx = int(detection.centre.x)
            cy = int(detection.centre.y)

            # Draw center point
            cv2.circle(modified_image,
              (cx,cy),
               5, (255, 0, 0), -1)
            
            # draw tag ID
            cv2.putText(modified_image,
                        f"ID: {detection.id}",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255), 2)

        # output video (?) to self.TagImageFrame (?) 
        if self.H is not None:
            modified_image = cv2.warpPerspective(
                modified_image, self.H,
                (modified_image.shape[1], modified_image.shape[0]))

        # self.TagImageFrame = modified_image  ##### UNCOMMENT THIS TO GO BACK TO NORMAL
        
class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image

        if self.camera.H is not None:
            modified_image = cv2.warpPerspective(self.camera.VideoFrame, self.camera.H, (self.camera.VideoFrame.shape[1], self.camera.VideoFrame.shape[0]))
            self.camera.VideoFrame = modified_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                self.camera.blockDetector()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()