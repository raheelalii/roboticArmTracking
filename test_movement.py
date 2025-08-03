import Robot
import math
import time

def main():
    ip = "192.168.58.2"
    max_attempts = 3

    # Try to connect via RPC (port 20003)
    robot_obj = None
    for i in range(max_attempts):
        try:
            print(f"Attempt {i+1}/{max_attempts}: connecting to {ip} …")
            robot_obj = Robot.RPC(ip)
            print("Connected.")
            break
        except Exception as e:
            print(f"Connection failed: {e}")
            if i < max_attempts - 1:
                time.sleep(2)
            else:
                raise SystemExit("Cannot establish RPC connection")

    # Switch to automatic mode with retry
    print("Switching to auto mode …")
    for _ in range(3):
        rc = robot_obj.Mode(0)  # 0 = auto, 1 = manual
        print("Mode() returned:", rc)
        if rc == 0:
            break
        print(f"Mode set failed: {rc}, retrying...")
        time.sleep(1)
    else:
        raise RuntimeError("Failed to set auto mode after retries")

    # Enable robot servo with retry
    print("Enabling robot servo …")
    for _ in range(3):
        rc = robot_obj.RobotEnable(1)  # 1 = enable, 0 = disable
        print("RobotEnable() returned:", rc)
        if rc == 0:
            break
        print(f"Enable failed: {rc}, retrying...")
        time.sleep(1)
    else:
        raise RuntimeError("Failed to enable robot after retries")

    # Try reading current joint angles; handle different return formats
    try:
        result = robot_obj.GetActualJointPosRadian()
        print("GetActualJointPosRadian result:", result)  # Debug
        if isinstance(result, int):
            err = result
            joints = None
        elif isinstance(result, tuple) and len(result) == 2:
            err, joints = result
        else:
            err = 0
            joints = result

        if err != 0 or not joints or not isinstance(joints, (list, tuple)):
            raise RuntimeError(f"GetActualJointPosRadian error {err} or invalid joints")

    except Exception as e:
        print("Failed to read joints:", e)
        # fallback angles in degrees converted to radians
        fallback_deg = [130.98, -52.884, -128.786, 27.451, 90.956, -43.87]
        joints = [math.radians(a) for a in fallback_deg]

    # Convert to degrees for GetForwardKin
    joint_deg = [math.degrees(rad) for rad in joints]

    # Compute forward kinematics to get Cartesian pose
    try:
        fk_result = robot_obj.GetForwardKin(joint_deg)
        print("GetForwardKin result:", fk_result)  # Debug
        if isinstance(fk_result, int):
            err = fk_result
            pose = None
        elif isinstance(fk_result, tuple) and len(fk_result) == 2:
            err, pose = fk_result
        else:
            err = 0
            pose = fk_result

        if err != 0 or not pose or not isinstance(pose, list):
            raise RuntimeError(f"GetForwardKin error {err} or invalid pose")

    except Exception as e:
        print("GetForwardKin failed:", e)
        pose = [200.0, 0.0, 300.0, 0.0, 0.0, 0.0]

    print("Current pose:", pose)
    print("Current joints (degrees):", joint_deg)

    # Plan linear movement: Move 10 cm (100 mm) forward along X-axis
    print("\n=== Moving 10 cm forward along X-axis using MoveL ===")
    
    # Create target pose by adding 100 mm to X
    target_pose = pose.copy()
    target_pose[0] += 100.0
    print(f"Moving to target pose: {target_pose}")

    # Perform MoveL using corrected parameters (after fixing Robot.py)
    # The XML-RPC call now expects 12 parameters (removed the extra '0')
    rc = robot_obj.MoveL(
        desc_pos=target_pose,
        tool=0,
        user=0,
        joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Triggers inverse kinematics
        vel=20.0,
        acc=0.0,
        ovl=100.0,
        blendR=-1.0,
        exaxis_pos=[0.0, 0.0, 0.0, 0.0],
        search=0,
        offset_flag=0,
        offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        overSpeedStrategy=0,
        speedPercent=10
    )
    print("MoveL() returned:", rc)
    if rc != 0:
        print(f"MoveL failed with error {rc}")
        return
    time.sleep(4)
    
    # Return to original pose
    print(f"Returning to original pose: {pose}")
    rc = robot_obj.MoveL(
        desc_pos=pose,
        tool=0,
        user=0,
        joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Triggers inverse kinematics
        vel=20.0,
        acc=0.0,
        ovl=100.0,
        blendR=-1.0,
        exaxis_pos=[0.0, 0.0, 0.0, 0.0],
        search=0,
        offset_flag=0,
        offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        overSpeedStrategy=0,
        speedPercent=10
    )
    print("MoveL() return to original returned:", rc)
    if rc != 0:
        print(f"MoveL return failed with error {rc}")
        return
    time.sleep(4)

    # Clean up
    robot_obj.RobotEnable(0)
    robot_obj.CloseRPC()
    print("Done.")

if __name__ == "__main__":
    main()