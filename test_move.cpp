#include "FRRobot.h"
#include <iostream>
#include <string.h>
#include <windows.h>
#include <math.h>

using namespace std;

#define PI 3.14159265358979323846

// Global robot object for reuse
static FRRobot* g_robot = nullptr;
static bool g_robot_initialized = false;

// Initialize robot connection (call once)
extern "C" __declspec(dllexport) int initialize_robot() {
    if (g_robot_initialized) {
        return 0; // Already initialized
    }
    
    try {
        g_robot = new FRRobot();
        
        int retval = g_robot->RPC("192.168.58.2");
        if (retval != 0) {
            cout << "Robot connection failed: " << retval << endl;
            delete g_robot;
            g_robot = nullptr;
            return retval;
        }
        
        // Enable the robot
        g_robot->RobotEnable(1);
        g_robot->SetSpeed(10);
        
        g_robot_initialized = true;
        cout << "Robot initialized successfully" << endl;
        return 0;
        
    } catch (...) {
        cout << "Robot initialization error" << endl;
        if (g_robot) {
            delete g_robot;
            g_robot = nullptr;
        }
        return -1;
    }
}

// Move robot to offset position in tool frame (transformed to base frame)
extern "C" __declspec(dllexport) int move_robot_to_offset(float offset_x, float offset_y, float offset_z) {
    if (!g_robot_initialized || !g_robot) {
        cout << "Robot not initialized. Call initialize_robot() first." << endl;
        return -1;
    }
    
    try {
        // Get current TCP pose
        DescPose current_pose;
        memset(&current_pose, 0, sizeof(DescPose));
        int retval = g_robot->GetActualTCPPose(0, &current_pose);
        if (retval != 0) {
            cout << "Failed to get current pose: " << retval << endl;
            return retval;
        }
        
        cout << "Current pose: X=" << current_pose.tran.x << ", Y=" << current_pose.tran.y << ", Z=" << current_pose.tran.z 
             << ", RX=" << current_pose.rpy.rx << ", RY=" << current_pose.rpy.ry << ", RZ=" << current_pose.rpy.rz << endl;
        
        // Extract RPY in degrees
        double roll = current_pose.rpy.rx;
        double pitch = current_pose.rpy.ry;
        double yaw = current_pose.rpy.rz;
        
        // Convert to radians
        double r_rad = roll * PI / 180.0;
        double p_rad = pitch * PI / 180.0;
        double y_rad = yaw * PI / 180.0;
        
        // Compute sine and cosine
        double cr = cos(r_rad), sr = sin(r_rad);
        double cp = cos(p_rad), sp = sin(p_rad);
        double cy = cos(y_rad), sy = sin(y_rad);
        
        // Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
        double R[3][3] = {
            {cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr},
            {sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr},
            {-sp,     cp * sr,               cp * cr}
        };
        
        // Offset vector in tool frame
        double offset_tool[3] = {offset_x, offset_y, offset_z};
        
        // Compute offset in base frame: R * offset_tool
        double offset_base[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                offset_base[i] += R[i][j] * offset_tool[j];
            }
        }
        
        // Calculate target pose
        DescPose target_pose = current_pose;
        target_pose.tran.x += static_cast<float>(offset_base[0]);
        target_pose.tran.y += static_cast<float>(offset_base[1]);
        target_pose.tran.z += static_cast<float>(offset_base[2]);
        
        cout << "Target pose: X=" << target_pose.tran.x << ", Y=" << target_pose.tran.y << ", Z=" << target_pose.tran.z 
             << ", RX=" << target_pose.rpy.rx << ", RY=" << target_pose.rpy.ry << ", RZ=" << target_pose.rpy.rz << endl;
        cout << "Moving robot to tool-frame offset: (" << offset_x << ", " << offset_y << ", " << offset_z << ") mm transformed to base frame" << endl;
        
        // Movement parameters
        int tool = 0;
        int user = 0;
        float vel = 50.0;
        float acc = 100.0;
        float ovl = 100.0;
        float blendT = -1.0;
        int config = -1;
        
        // Execute movement
        retval = g_robot->MoveCart(&target_pose, tool, user, vel, acc, ovl, blendT, config);
        if (retval != 0) {
            cout << "Movement failed: " << retval << endl;
            return retval;
        }
        
        cout << "Movement command sent successfully" << endl;
        
        // Wait for movement completion
        uint8_t state = 0;
        int timeout = 100; // 10 seconds timeout
        while (state == 0 && timeout > 0) {
            g_robot->GetRobotMotionDone(&state);
            Sleep(100);
            timeout--;
        }
        
        if (timeout <= 0) {
            cout << "Movement timeout" << endl;
            return -2;
        }
        
        cout << "Movement completed" << endl;
        return 0;
        
    } catch (...) {
        cout << "Movement error" << endl;
        return -1;
    }
}

// Cleanup robot connection
extern "C" __declspec(dllexport) void cleanup_robot() {
    if (g_robot_initialized && g_robot) {
        g_robot->RobotEnable(0);
        g_robot->CloseRPC();
        delete g_robot;
        g_robot = nullptr;
        g_robot_initialized = false;
        cout << "Robot cleanup completed" << endl;
    }
}

// Original main function for standalone testing
int main(void)
{
    // Test the new functions
    cout << "Testing robot functions..." << endl;
    
    int result = initialize_robot();
    if (result != 0) {
        cout << "Failed to initialize robot" << endl;
        return result;
    }
    
    // Test movement: move 100mm forward in tool Z (assuming Z is forward)
    result = move_robot_to_offset(0.0, 0.0, 100.0);
    if (result != 0) {
        cout << "Movement test failed" << endl;
    }
    
    Sleep(2000); // Wait 2 seconds
    
    // Return to original position by moving -100mm in tool Z
    result = move_robot_to_offset(0.0, 0.0, -100.0);
    if (result != 0) {
        cout << "Return movement failed" << endl;
    }
    
    cleanup_robot();
    return 0;
}