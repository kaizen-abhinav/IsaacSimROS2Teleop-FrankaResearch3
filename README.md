# IsaacSimROS2Teleop-FrankaResearch3
# Panda Hand Controller — Vision Teleoperation

This project provides a complete vision-based teleoperation pipeline to control a **Franka Emika Panda** robot in **NVIDIA Isaac Sim** using just a webcam and your hand gestures.

It uses **MediaPipe** for real-time 3D hand tracking, maps the hand coordinates to the robot's reachable workspace, recognizes discrete gestures (Pinch, Fist, Open Palm), and streams everything into Isaac Sim via **ROS 2**.

## Features

- **Real-Time 3D Hand Tracking**: Uses a standard webcam to detect 21 hand landmarks.
- **Amplified Workspace Mapping**: Scales small hand movements across the camera into full-range robot motions.
- **Gesture Recognition Engine**:
  - 🖐️ **Open Palm**: `MOVE` (Robot follows hand)
  - ✊ **Fist**: `FREEZE` (Robot holds its current position)
  - 🤏 **Pinch**: `PICK` (Closes the gripper while following)
  - ✋ **Release Pinch**: `PLACE` (Opens the gripper, returns to `MOVE`)
  - ☝️ **Point Up**: `WRIST` (Index finger angle controls robot wrist rotation)
- **Direct Isaac Sim Integration**: Bypasses complex control loops by streaming target poses straight into Isaac Sim's built-in Lula Inverse Kinematics solver.

---

## Prerequisites

1. **Ubuntu** (tested on 22.04)
2. **ROS 2 Humble** installed and sourced.
3. **NVIDIA Isaac Sim** (tested on 5.1.0).
4. **Python 3** with `cv2` and `mediapipe` installed in your ROS2 environment:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

---

## System Architecture

1. **`hand_pose_publisher.py`**: A ROS2 Node running outside of Isaac Sim. It captures webcam data, runs MediaPipe inference, maintains the teleop state machine, and publishes:
   - `/franka/target_pose` (`geometry_msgs/PoseStamped`): The target EE Cartesian pose.
   - `/franka/gripper_command` (`std_msgs/Float64`): The target gripper width (0.0 to 0.04m).
   - `/franka/teleop_state` (`std_msgs/String`): The current gesture state.
2. **Isaac Sim Action Graph (OmniGraph)**:
   - Subscribes to the ROS2 topics.
   - Feeds the pose and gripper data into a custom **Script Node**.
   - The Script Node runs the **Lula Kinematics Solver** to compute joint angles.
   - The joint angles are sent to an **Articulation Controller** to drive the `/World/Franka` prim.

---

## How to Launch

### Step 1: Start the Vision Publisher
Open a new terminal and run:

```bash
# 1. Source ROS 2
source /opt/ros/humble/setup.bash

# 2. Navigate to the project directory
cd ~/PandaHandController

# 3. Run the publisher
python3 hand_pose_publisher.py
```
*A window should appear showing your webcam feed with a HUD overlay indicating the current state, coordinates, and finger status.*

### Step 2: Configure Isaac Sim

Make sure your Isaac Sim environment is launched with ROS 2 sourced in the terminal beforehand!

1. Load your `.usd` environment containing the Franka robot.
2. Open the **Action Graph** and ensure the following flow exists:
   - `On Playback Tick` → `ROS2 Subscriber` (Topic: `/franka/target_pose`, Type: `PoseStamped`)
   - `On Playback Tick` → `ROS2 Subscriber` (Topic: `/franka/gripper_command`, Type: `Float64`)
3. Connect the outputs of the Subscribers to the custom **Script Node**.
   - Pose goes into `posePositionX`, `Y`, `Z` and `poseOrientationX`, `Y`, `Z`, `W`.
   - Gripper command goes into `gripperWidth`.
4. Connect the Script Node's `jointPositions` output to the **Articulation Controller**'s `positionCommand`.
5. Ensure the **Articulation Controller**'s `targetPrim` is pointing directly at the Franka robot root in the Stage (e.g., `/World/Franka`).

### Step 3: Run the Simulation
Press **▶ Play** in Isaac Sim. 

Watch the Isaac Sim Console. Once you see:
```text
[IK] Lula solver ready!
```
You can hold your hand up to the camera (Open Palm) and the robot will immediately snap to your hand's position.

---

## Troubleshooting

- **"No module named 'rclpy'"**: You started Isaac Sim from the UI launcher instead of a terminal with `source /opt/ros/humble/setup.bash`. You must launch Isaac Sim via terminal.
- **Robot is jittery/shaking**: Ensure your room is well-lit for MediaPipe. If necessary, increase the `alpha` value in the `LowPassFilter` inside `hand_pose_publisher.py` (e.g., from `0.4` to `0.2` for smoother but slightly slower movement).
- **Robot doesn't move but IK is computing**: Check the Articulation Controller's `targetPrim` path. It must exactly match the path to the Franka articulation root in the Stage tree.
- **IK failing (success=False)**: Hand coordinates might be pushing the arm outside its reachable workspace. Make sure your hand remains relatively centered in the camera frame.


