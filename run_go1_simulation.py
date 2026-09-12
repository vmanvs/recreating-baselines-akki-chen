"""
run_go1_simulation.py
Simulates the Unitree Go1 quadruped robot in MuJoCo using the unitree_mujoco model,
supporting both RL and MPC controllers with live perturbation injection and trajectory logging.

Usage:
  # Run RL controller in headless mode:
  python run_go1_simulation.py --controller rl --steps 1000 --headless

  # Run MPC controller in headless mode:
  python run_go1_simulation.py --controller mpc --steps 1000 --headless

  # Run with perturbation in positive x (+150 N for 0.2s):
  python run_go1_simulation.py --controller rl --perturb +x --force 150 --headless

  # Run with interactive 3D GUI viewer (press Space to pause/play):
  python run_go1_simulation.py --controller rl --gui
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import mujoco
try:
    import mujoco.viewer as mj_viewer
except ImportError:
    mj_viewer = None

from controllers import RLController, MPCController

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO_DIR, "unitree_mujoco", "unitree_robots", "go1", "scene.xml")
LOG_DIR = os.path.join(REPO_DIR, "sim_results")
os.makedirs(LOG_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate Unitree Go1 in MuJoCo with RL / MPC")
    parser.add_argument("--controller", type=str, choices=["rl", "mpc"], default="rl",
                        help="Locomotion controller: 'rl' or 'mpc'")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of simulation control steps (default: 1000 = ~20s)")
    parser.add_argument("--perturb", type=str, choices=["none", "+x", "-x", "-y", "+y"], default="none",
                        help="Direction of external perturbation force")
    parser.add_argument("--force", type=float, default=150.0,
                        help="Perturbation force magnitude in Newtons (default: 150.0)")
    parser.add_argument("--perturb_time", type=float, default=4.0,
                        help="Time in seconds when perturbation is triggered (default: 4.0s)")
    parser.add_argument("--perturb_duration", type=float, default=0.2,
                        help="Duration of perturbation impulse in seconds (default: 0.2s)")
    parser.add_argument("--gui", action="store_true",
                        help="Launch interactive 3D MuJoCo viewer")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI at maximum simulation speed")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save output trajectory CSV")
    return parser.parse_args()

def run_simulation(args):
    print("=" * 65)
    print(f"Starting Unitree Go1 Simulation ({args.controller.upper()} Controller)")
    print("=" * 65)
    print(f"Model Path         : {MODEL_PATH}")
    print(f"Simulation Steps   : {args.steps}")
    print(f"Perturbation Mode  : {args.perturb.upper()} ({args.force} N for {args.perturb_duration}s at t={args.perturb_time}s)")
    print(f"Execution Mode     : {'Interactive GUI' if args.gui else 'Headless Fast'}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data = mujoco.MjData(mj_model)
    
    # Reset to nominal home pose
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    
    trunk_body_id = mj_model.body("trunk").id
    dt = mj_model.opt.timestep # usually 0.002s
    sim_decimation = 10 # control loop at dt_ctrl = 0.02s (50 Hz)
    dt_ctrl = dt * sim_decimation
    
    # Initialize controller
    if args.controller == "rl":
        controller = RLController(target_vel=0.5, target_height=0.26)
    else:
        controller = MPCController(mj_model, horizon=6, num_samples=12, target_vel=0.5, target_height=0.26)
        
    logs = []
    
    # Viewer setup
    viewer = None
    if args.gui:
        if mj_viewer is not None:
            try:
                viewer = mj_viewer.launch_passive(mj_model, mj_data)
                # ── Camera tracking setup ──────────────────────────────────
                # lookat  = point the camera orbits around (robot CoM)
                # azimuth = horizontal angle (degrees)  – side view = 90°
                # elevation = vertical angle (degrees)  – slight downward
                # distance  = how far back the camera sits (metres)
                with viewer.lock():
                    viewer.cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat[:]  = mj_data.qpos[:3]   # start at robot pos
                    viewer.cam.azimuth    = 90.0   # view from the side (+x direction)
                    viewer.cam.elevation  = -20.0  # look slightly down
                    viewer.cam.distance   = 2.5    # 2.5 m back from the robot
                print("Interactive 3D viewer launched — camera tracks the robot. Close window to stop.")
            except Exception as e:
                print(f"Note: Could not launch GUI viewer ({e}), falling back to headless.")
                viewer = None
        else:
            print("Note: mujoco.viewer not available, falling back to headless.")

    start_real_time = time.time()
    
    for step_idx in range(args.steps):
        sim_time = step_idx * dt_ctrl
        
        # Determine perturbation force vector [fx, fy, fz]
        ext_force_mag = 0.0
        force_vec = np.zeros(3)
        if args.perturb != "none":
            if args.perturb_time <= sim_time < (args.perturb_time + args.perturb_duration):
                ext_force_mag = args.force
                if args.perturb == "+x":
                    force_vec[0] = args.force
                elif args.perturb == "-x":
                    force_vec[0] = -args.force
                elif args.perturb == "-y":
                    force_vec[1] = -args.force
                elif args.perturb == "+y":
                    force_vec[1] = args.force
                    
        # Apply external force to torso CoM
        mj_data.xfrc_applied[trunk_body_id, :3] = force_vec
        
        # Get controller action
        torso_quat = mj_data.qpos[3:7]
        torso_vel = mj_data.qvel[:3]
        
        if args.controller == "rl":
            action = controller.get_action(sim_time, mj_data.qpos, mj_data.qvel, torso_quat, torso_vel)
        else:
            action = controller.get_action(sim_time, mj_data)
            
        mj_data.ctrl[:] = action
        
        # Step physics sub-steps
        for _ in range(sim_decimation):
            mujoco.mj_step(mj_model, mj_data)
            
        # Extract states for benchmark logging
        # [time, x, y, z, vx, vy, vz, ctrl1, ctrl2, ctrl3, external_force]
        x, y, z = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
        vx, vy, vz = mj_data.qvel[0], mj_data.qvel[1], mj_data.qvel[2]
        # Single limb torques / control inputs (FR limb: abduction, hip flexion, knee extension)
        ctrl_abd = action[0]
        ctrl_hip = action[1]
        ctrl_knee = action[2]
        
        logs.append([sim_time, x, y, z, vx, vy, vz, ctrl_abd, ctrl_hip, ctrl_knee, ext_force_mag])
        
        if viewer is not None:
            # ── Track robot: slide lookat to follow CoM every frame ────────
            with viewer.lock():
                viewer.cam.lookat[0] = mj_data.qpos[0]  # x follows robot
                viewer.cam.lookat[1] = mj_data.qpos[1]  # y follows robot
                viewer.cam.lookat[2] = mj_data.qpos[2]  # z follows robot height
            viewer.sync()
            time.sleep(max(0.0, dt_ctrl - 0.005))
            if not viewer.is_running():
                break
                
    total_real_time = time.time() - start_real_time
    total_sim_time = len(logs) * dt_ctrl
    
    # Save log CSV
    log_arr = np.array(logs)
    if args.output_csv is None:
        out_csv = os.path.join(LOG_DIR, f"go1_{args.controller}_{args.perturb}.csv")
    else:
        out_csv = args.output_csv
        
    df = pd.DataFrame(log_arr, columns=[
        "time", "x", "y", "z", "vx", "vy", "vz", "ctrl_abd", "ctrl_hip", "ctrl_knee", "ext_force"
    ])
    df.to_csv(out_csv, index=False)
    
    # Compute run performance summary
    final_x = log_arr[-1, 1]
    mean_vx = np.mean(log_arr[len(log_arr)//4:, 4])
    var_vx = np.var(log_arr[len(log_arr)//4:, 4])
    final_z = log_arr[-1, 3]
    
    print("\n" + "=" * 65)
    print("SIMULATION RUN SUMMARY")
    print("=" * 65)
    print(f"Simulated Duration : {total_sim_time:.2f} s ({len(logs)} steps)")
    print(f"Computation Time   : {total_real_time:.2f} s ({total_sim_time/total_real_time:.1f}x real-time)")
    print(f"Distance Traveled  : {final_x:.2f} m along X-axis")
    print(f"Mean Steady Vel vx : {mean_vx:.3f} m/s (Target: 0.50 m/s)")
    print(f"Velocity Var(vx)   : {var_vx:.6f}")
    print(f"Final CoM Height   : {final_z:.3f} m (Nominal: 0.26 m)")
    print(f"Output CSV Saved   : {out_csv}")
    print("=" * 65)
    
    return out_csv

if __name__ == "__main__":
    args = parse_args()
    run_simulation(args)
