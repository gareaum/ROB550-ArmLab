# 🤖🦾 ArmLab: Autonomous 5-DOF Robotic Arm

This project, developed for **ROB550 – Spring 2026**, focuses on building an autonomous 5-DOF **ReactorX 200 robotic arm** capable of detecting, grasping, and arranging colored blocks. The system integrates **computer vision, kinematics, and motion planning** to enable precise end-effector manipulation.

![ArmLab Setup](Images/Setup.png)

---

## 🧠 Key Features

### Computer Vision
![Block Detection](Images/Block_Detection.png)
The system uses an **overhead Intel RealSense LiDAR camera** to detect and localize blocks in the workspace. Images are first white-balanced and brightness-normalized to account for lighting variations, then converted to **HSV color space** to segment blocks by color. Contours are filtered to remove noise, allowing the robot to accurately determine each block’s position in pixel coordinates, which are then transformed into **3D world coordinates** for autonomous manipulation.

---

### Calibration & Homography
![Homography](Images/Homography.png)
The camera is calibrated using **intrinsic parameters** and **AprilTag-based extrinsic calibration** to map pixels to real-world coordinates. A **homography transformation** corrects for the angled camera view, producing a top-down rectified workspace. This ensures the robot can precisely locate and interact with blocks for grasping and placement tasks.

---

### Kinematics
![DH Kinematics](Images/DH_Kinematics.png)
The robotic arm uses **forward and inverse kinematics** to translate between joint angles and end-effector positions. Forward kinematics calculates the end-effector pose from joint angles, while inverse kinematics computes the joint angles needed to reach a desired position. Together, these models enable the arm to accurately grasp and place blocks within its workspace.

---

### Autonomous Motion

The robot performs autonomous block manipulation using a **three-stage motion sequence**: approach, grasp, and retract. This strategy ensures stable and precise handling of objects, allowing the arm to reliably pick up blocks and place them in predefined configurations without manual intervention.

---

## 📊 Results

- **Forward Kinematics Accuracy:** Average positional error ~6.25 mm, ensuring reliable grasping and placement.  
- **Block Detection:** Successfully detects and localizes colored blocks in real-time; performance is highest near the center of the workspace.  
- **Autonomous Task Performance:** Completed multiple block manipulation cycles with stable and repeatable execution.  
- **Calibration & Homography:** Factory intrinsic matrix combined with AprilTag extrinsic calibration produced the most accurate coordinate mapping.  

For a detailed explanation of methods, calibration data, and full experimental results, see the [ROB550 ArmLab Report](ROB550_ArmLab_Report.pdf) in this repository. 

