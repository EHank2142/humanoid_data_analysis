# Copyright information
#
# © [2025] LimX Dynamics Technology Co., Ltd. All rights reserved.
# 
# Modified by Bao Guo, The University of Hong Kong, 2025-2026. All modifications are licensed under the same terms as the original code.

import os
import sys
import time
import subprocess
import atexit
import datetime
import mujoco
import mujoco.viewer as viewer
import numpy as np
from functools import partial
import limxsdk
from limxsdk.robot.Rate import Rate
from limxsdk.robot.Robot import Robot
from limxsdk.robot.RobotType import RobotType
import limxsdk.datatypes as datatypes

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEPLOY_DIR)

# Vicon Plug-in Gait Full Body marker set (39 markers), matching the <site>
# elements added to HU_D04_01.xml. Sites are looked up by name at load time,
# so this list is safe to use for any robot xml -- names simply won't
# resolve (and are skipped) on robots that don't define them.
MARKER_NAMES = [
    "LFHD", "RFHD", "LBHD", "RBHD",
    "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI",
    "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
]

# Marker trajectory is recorded at this rate (Hz), independent of the
# simulation's own control-loop rate (typically 1000 Hz), to keep the log
# a manageable size and to match typical mocap capture rates.
MARKER_LOG_FPS = 100.0

MARKER_LOG_DIR = os.path.join(DEPLOY_DIR, "marker_logs")

KINEMATIC_PROJECTION_DIR = os.path.join(
    DEPLOY_DIR,
    "prebuild",
)

ROBOT_DESCRIPTION_DIR = os.path.join(
    PROJECT_ROOT,
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "assets",
)


class SimulatorMujoco:
    def __init__(self, asset_path, robot, floating_base): 
        self.robot = robot
        self.floating_base = floating_base
        
        # Load the MuJoCo model and data from the specified XML asset path
        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        # Get the number of actuators
        self.actuator_count = self.mujoco_model.nu

        # Get the number of joints
        self.joint_num = self.mujoco_model.njnt

        # Get the joint names
        self.joint_sensor_names = []
        for i in range(self.joint_num):
            joint_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_sensor_names.append(joint_name)
        
        # Launch the MuJoCo viewer in passive mode with custom settings
        self.viewer = viewer.launch_passive(self.mujoco_model, self.mujoco_data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
        self.viewer.cam.distance = 10  # Set camera distance
        self.viewer.cam.elevation = -20  # Set camera elevation
    
        self.dt = self.mujoco_model.opt.timestep  # Get simulation timestep
        self.fps = 1 / self.dt  # Calculate frames per second (FPS)

        # Resolve marker site ids present in this model (robots other than
        # HU_D04_01 may not define all/any of them -- skip missing ones).
        self.marker_names = []
        self.marker_site_ids = []
        for name in MARKER_NAMES:
            sid = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                self.marker_names.append(name)
                self.marker_site_ids.append(sid)
        if self.marker_names:
            print(f"[MuJoCo] Logging {len(self.marker_names)}/{len(MARKER_NAMES)} marker sites "
                  f"at {MARKER_LOG_FPS:.0f} Hz.")
        else:
            print("[MuJoCo] No marker sites found on this model; marker trajectory logging disabled.")

        self.marker_log_decimation = max(1, round(self.fps / MARKER_LOG_FPS))
        self.marker_log_t = []
        self.marker_log_pos = []
        atexit.register(self.save_marker_log)

        # Initialize robot command data with default values
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.actuator_count)]
        self.robot_cmd.q = [0. for x in range(0, self.actuator_count)]
        self.robot_cmd.dq = [0. for x in range(0, self.actuator_count)]
        self.robot_cmd.tau = [0. for x in range(0, self.actuator_count)]
        self.robot_cmd.Kp = [0. for x in range(0, self.actuator_count)]
        self.robot_cmd.Kd = [0. for x in range(0, self.actuator_count)]

        # Initialize robot state data with default values
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.actuator_count)]
        self.robot_state.q = [0. for x in range(0, self.actuator_count)]
        self.robot_state.dq = [0. for x in range(0, self.actuator_count)]
        self.robot_state.motor_names = ['' for x in range(0, self.actuator_count)]

        # Initialize IMU data structure
        self.imu_data = datatypes.ImuData()

        # Robot-command diagnostics. Controllers can publish at 1 kHz, so log
        # at most once per second instead of printing every callback.
        self.robot_cmd_count = 0
        self.last_robot_cmd_log_time = 0.0

        # Set up callback for receiving robot commands in simulation mode
        self.robotCmdCallbackPartial = partial(self.robotCmdCallback)
        self.robot.subscribeRobotCmdForSim(self.robotCmdCallbackPartial)

    # Callback function for receiving robot command data
    def robotCmdCallback(self, robot_cmd: datatypes.RobotCmd):
        self.robot_cmd = robot_cmd
        self.robot_cmd_count += 1

        now = time.monotonic()
        if self.robot_cmd_count == 1 or now - self.last_robot_cmd_log_time >= 1.0:
            self.last_robot_cmd_log_time = now
            q = list(robot_cmd.q)
            kp = list(robot_cmd.Kp)
            kd = list(robot_cmd.Kd)
            tau = list(robot_cmd.tau)
            print(
                "[MuJoCo] RobotCmd received "
                f"(count={self.robot_cmd_count}, q={len(q)}, Kp={len(kp)}, "
                f"Kd={len(kd)}, tau={len(tau)}): "
                f"q[:5]={[round(float(value), 4) for value in q[:5]]}, "
                f"Kp[:5]={[round(float(value), 2) for value in kp[:5]]}",
                flush=True,
            )

    # Callback for keypress events in the MuJoCo viewer (currently does nothing)
    def key_callback(self, keycode):
        pass

    def run(self):
        frame_count = 0
        self.rate = Rate(self.fps)  # Set the update rate according to FPS
        while self.viewer.is_running():
            # Step the MuJoCo physics simulation
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            # Record marker trajectory (downsampled to MARKER_LOG_FPS)
            if self.marker_names and frame_count % self.marker_log_decimation == 0:
                self.marker_log_t.append(self.mujoco_data.time)
                self.marker_log_pos.append(self.mujoco_data.site_xpos[self.marker_site_ids].copy())

            if not self.floating_base:
                # Extract IMU data (orientation, gyro, and acceleration) from simulation
                self.imu_data.quat[0] = self.mujoco_data.sensordata[0]
                self.imu_data.quat[1] = self.mujoco_data.sensordata[1]
                self.imu_data.quat[2] = self.mujoco_data.sensordata[2]
                self.imu_data.quat[3] = self.mujoco_data.sensordata[3]

                self.imu_data.gyro[0] = self.mujoco_data.sensordata[4]
                self.imu_data.gyro[1] = self.mujoco_data.sensordata[5]
                self.imu_data.gyro[2] = self.mujoco_data.sensordata[6]

                self.imu_data.acc[0] = self.mujoco_data.sensordata[7]
                self.imu_data.acc[1] = self.mujoco_data.sensordata[8]
                self.imu_data.acc[2] = self.mujoco_data.sensordata[9]

                # Set the timestamp for the current IMU data and publish it
                self.imu_data.stamp = time.time_ns()
                self.robot.publishImuDataForSim(self.imu_data)

                # Update robot state data from simulation
                for i in range(self.actuator_count):
                    self.robot_state.q[i] = self.mujoco_data.sensordata[i + 10]
                    self.robot_state.dq[i] = self.mujoco_data.sensordata[self.actuator_count + i + 10]
                    self.robot_state.tau[i] = self.mujoco_data.ctrl[i]

                    # Apply control commands to the robot based on the received robot command data
                    self.mujoco_data.ctrl[i] = (
                        self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) + 
                        self.robot_cmd.Kd[i] * (self.robot_cmd.dq[i] - self.robot_state.dq[i]) + 
                        self.robot_cmd.tau[i]
                    )
            else:
                # Update robot state data from simulation
                for i in range(self.actuator_count):
                    self.robot_state.q[i] = self.mujoco_data.sensordata[i]
                    self.robot_state.dq[i] = self.mujoco_data.sensordata[self.actuator_count + i]
                    self.robot_state.tau[i] = self.mujoco_data.ctrl[i]

                    # Apply control commands to the robot based on the received robot command data
                    self.mujoco_data.ctrl[i] = (
                        self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) + 
                        self.robot_cmd.Kd[i] * (self.robot_cmd.dq[i] - self.robot_state.dq[i]) + 
                        self.robot_cmd.tau[i]
                    )
        
            # Set the timestamp for the current robot state and publish it
            self.robot_state.stamp = time.time_ns()
            self.robot.publishRobotStateForSim(self.robot_state)

            # Sync the viewer every 20 frames for smoother visualization
            if frame_count % 20 == 0:
                self.viewer.sync()

            frame_count += 1
            self.rate.sleep()  # Maintain the simulation loop at the correct rate

    def save_marker_log(self):
        """Write the recorded marker trajectory to .npz on process exit.

        Output schema mirrors carepd_extract_keypoints.py /
        gmr_extract_keypoints.py (joint_names/positions/fps/unit/source_file)
        so Part B/C can read this file with the same code, using
        marker_names in place of joint_names since these are raw skin
        markers (position only, no orientation -- matches what a real c3d
        file records).
        """
        if not self.marker_names or not self.marker_log_pos:
            return
        os.makedirs(MARKER_LOG_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(MARKER_LOG_DIR, f"mujoco_markers_{timestamp}.npz")
        np.savez(
            out_path,
            marker_names=np.array(self.marker_names),
            positions=np.stack(self.marker_log_pos, axis=0).astype(np.float32),  # (T, J, 3)
            t=np.array(self.marker_log_t, dtype=np.float32),
            fps=np.float32(MARKER_LOG_FPS),
            unit="m",
            source_file=out_path,
        )
        print(f"[MuJoCo] Saved {len(self.marker_log_pos)} frames x {len(self.marker_names)} "
              f"markers to {out_path}")


def run_kinematic_projection():
    try:
        # Build the full path of the executable program
        program_path = os.path.join(KINEMATIC_PROJECTION_DIR, 'kinematic_projection')
        
        # Build the full path of the executable program etc
        program_etc_path = os.path.join(KINEMATIC_PROJECTION_DIR, 'etc')

        if not os.path.exists(program_path):
            raise FileNotFoundError(program_path)

        if not os.path.isdir(program_etc_path):
            raise FileNotFoundError(program_etc_path)

        # Copy the current environment variables
        env = os.environ.copy()
        env['MROS_ETC_PATH'] = program_etc_path
        env['MROS_LOG_LEVEL'] = "0"

        # Start the executable program and pass in the modified environment variables
        process = subprocess.Popen(program_path, env=env)

        def cleanup():
            # When the Python script exits, check if the subprocess is still running
            if process.poll() is None:
                print("Trying to terminate the subprocess...")
                # Send a termination signal to the subprocess
                process.terminate()
                try:
                    # Wait for the subprocess to terminate within 5 seconds
                    process.wait(timeout=5)
                    print("The subprocess has been successfully terminated.")
                except subprocess.TimeoutExpired:
                    # If it times out, force kill the subprocess
                    print("The subprocess did not terminate within the specified time, forcing termination...")
                    process.kill()
                    print("The subprocess has been force-killed.")

        # Register the cleanup function to ensure it is called when the script exits
        atexit.register(cleanup)
        return process
    except FileNotFoundError:
        print(
            "Error: kinematic_projection assets were not found under "
            f"{KINEMATIC_PROJECTION_DIR}. Please vendor the executable and its etc directory into HUB."
        )
    except PermissionError:
        print("Error: You do not have permission to execute the program.")
    except Exception as e:
        print(f"An unknown error occurred: {e}")
    return None

if __name__ == '__main__': 
    robot_type = os.getenv("ROBOT_TYPE")

    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)
        
    # Split from the right side of ROBOT_TYPE and get the content before the first underscore as the main type
    main_robot_type = robot_type.rsplit('_', 1)[0]

    # Is floating base
    floating_base = main_robot_type.startswith(('DA_', 'UB_'))

    # Default IP address for the robot
    robot_ip = "127.0.0.1"
    
    # Check if command-line argument is provided for robot IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]
    
    ip_parts = robot_ip.split('.')
    if len(ip_parts) == 4:
        os.environ["MROS_IP_LIST"] = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.x"
    else:
        print(f"ERROR: Invalid IPv4 format ({robot_ip})")
        sys.exit(1)

    # Create a Robot instance of the Humanoid type
    robot = Robot(RobotType.Humanoid, True)

    # Initialize the robot with the provided IP address
    if not robot.init(robot_ip):
        sys.exit(1)

    # Define the path to the robot model XML file based on the robot type
    model_path = os.path.join(
        ROBOT_DESCRIPTION_DIR,
        f'{main_robot_type}_description',
        'xml',
        f'{robot_type}.xml',
    )

    # Check if the model file exists, otherwise exit with an error
    if not os.path.exists(model_path):
        print(f"Error: The file {model_path} does not exist. Please ensure the ROBOT_TYPE is set correctly.")
        sys.exit(1)
    
    # Run kinematic_projection
    run_kinematic_projection()

    # Create and run the MuJoCo simulator instance
    print(f"*** Model File Loaded: {model_path} ***")
    simulator = SimulatorMujoco(model_path, robot, floating_base)
    simulator.run()
