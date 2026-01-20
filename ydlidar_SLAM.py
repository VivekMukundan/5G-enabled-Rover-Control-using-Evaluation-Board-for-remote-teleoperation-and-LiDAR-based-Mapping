#!/usr/bin/env python3
"""
ydlidar_slam.py

2D SLAM using YDLidarX2 as scan source + ICP scan-matching + pose-graph optimization.

- Imports YDLidarX2 from local ydlidar_x2.py if present, else from the uploaded file path
  '/mnt/data/18472f30-7488-4954-985e-2cd7c931baf1.py'.
- Live visualization using matplotlib.
- Saves:
    - 'slam_pointcloud.csv' (global XY points in meters)
    - 'slam_occupancy.png' (occupancy grid)
- Tunable parameters at the top of this file.

Author: assistant (adapted for your YDLidarX2)
"""

import os
import time
import math
import numpy as np
import threading
import csv
import importlib.util
from collections import deque

# Visualization & ICP / optimization
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.optimize import least_squares

# ----------------------------- Config (tweak these) -------------------------
# LiDAR / IO
LIDAR_PORT = 'COM8'               # serial port for your LiDAR
DRIVER_FALLBACK_PATH = '/mnt/data/18472f30-7488-4954-985e-2cd7c931baf1.py'

# SLAM params
ASSUMED_SPEED = 0.18              # m/s (you provided)
ASSUMED_SCAN_HZ = 7.0             # nominal; script measures real scans too
DOWNSAMPLE_ANGLE = 2              # keep every 2nd beam (2 deg if 360 beams) -> reduce density
ICP_MAX_ITER = 50
ICP_DISTANCE_THRESHOLD = 0.2      # meters; max correspondence distance
KEYFRAME_SPACING = 0.5            # meters -> add keyframe when traveled this far
LOOP_CLOSURE_RADIUS = 1.0         # meters -> consider loop closures within this radius
LOOP_CLOSURE_MIN_SEP = 10         # minimum keyframes separation index to consider closure
OCCUPANCY_RESOLUTION = 0.02       # meters per cell for output occupancy grid

# Visualization
LIVE_PLOT_UPDATE_SECONDS = 0.5

# Outputs
OUT_PCD_CSV = 'slam_pointcloud.csv'
OUT_OCCUPANCY_PNG = 'slam_occupancy.png'

# ---------------------------------------------------------------------------

# ------------------------- Import YDLidarX2 driver --------------------------
try:
    from ydlidar_x2 import YDLidarX2
except Exception:
    # fallback to session-uploaded file path
    if os.path.exists(DRIVER_FALLBACK_PATH):
        spec = importlib.util.spec_from_file_location("ydlidar_user_driver", DRIVER_FALLBACK_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        YDLidarX2 = getattr(mod, "YDLidarX2")
    else:
        raise ImportError("Could not import YDLidarX2. Place ydlidar_x2.py next to this script "
                          "or ensure DRIVER_FALLBACK_PATH exists.")

# ----------------------------- Utility functions ----------------------------
def polar_to_cartesian_m(angle_deg, dist_mm):
    """Convert polar (deg, mm) to cartesian (x, y) in meters relative to sensor."""
    if dist_mm <= 0:
        return None
    r = dist_mm / 1000.0
    theta = math.radians(angle_deg)
    x = r * math.sin(theta)
    y = r * math.cos(theta)
    return np.array([x, y], dtype=np.float64)

def scan_to_points(indices_angles, distances_mm, downsample=DOWNSAMPLE_ANGLE, out_of_range=65000):
    """Convert raw 360-array distances (mm) to Nx2 numpy array (meters)."""
    pts = []
    for angle in range(0, len(distances_mm), downsample):
        d = distances_mm[angle]
        if d < out_of_range and d > 0:
            p = polar_to_cartesian_m(angle, d)
            if p is not None:
                pts.append(p)
    if len(pts) == 0:
        return np.zeros((0,2), dtype=np.float64)
    return np.vstack(pts)

# 2D SE(2) helpers: pose = (x, y, yaw)
def pose_to_transform(p):
    x, y, th = p
    c = math.cos(th); s = math.sin(th)
    T = np.array([[c, -s, x],
                  [s,  c, y],
                  [0,  0, 1]], dtype=np.float64)
    return T

def transform_to_pose(T):
    x = T[0,2]; y = T[1,2]; th = math.atan2(T[1,0], T[0,0])
    return np.array([x, y, th], dtype=np.float64)

def compose_pose(pA, pB):
    TA = pose_to_transform(pA)
    TB = pose_to_transform(pB)
    T = TA.dot(TB)
    return transform_to_pose(T)

def invert_pose(p):
    T = pose_to_transform(p)
    Tinv = np.linalg.inv(T)
    return transform_to_pose(Tinv)

def apply_pose_to_points(points, pose):
    """Apply pose (x,y,yaw) to Nx2 points."""
    T = pose_to_transform(pose)
    ones = np.ones((points.shape[0],1))
    hom = np.hstack((points, ones))
    transf = (T.dot(hom.T)).T
    return transf[:,0:2]

# --------------------- ICP wrapper using Open3D (2D adapted) -----------------
def icp_2d(A_pts, B_pts, init_transform=np.eye(3), max_iter=ICP_MAX_ITER, dist_thresh=ICP_DISTANCE_THRESHOLD):
    """
    Run ICP between two Nx2 numpy arrays A (source) and B (target).
    Returns 3x3 transform matrix (T such that T * A = B) and fitness/error.
    """
    if A_pts.shape[0] < 10 or B_pts.shape[0] < 10:
        return np.eye(3), 0.0, float('inf')  # not enough points

    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.hstack((A_pts, np.zeros((A_pts.shape[0],1)))))
    tgt.points = o3d.utility.Vector3dVector(np.hstack((B_pts, np.zeros((B_pts.shape[0],1)))))

    # initial transform: 4x4
    init4 = np.eye(4)
    init4[0:3,0:3] = init_transform[0:3,0:3]
    init4[0:3,3] = init_transform[0:3,2]

    # use ICP from Open3D with point-to-point
    try:
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, dist_thresh,
            init4,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )
    except Exception as e:
        # fallback to identity if ICP fails for any reason
        print("ICP exception:", e)
        res = None

    if res is None:
        return np.eye(3), 0.0, float('inf')

    T4 = res.transformation
    T3 = np.eye(3)
    T3[0:2,0:2] = T4[0:2,0:2]
    T3[0:2,2] = T4[0:2,3]
    fitness = res.fitness
    rmse = res.inlier_rmse
    return T3, fitness, rmse

# ------------------------- Pose graph optimization --------------------------
# Graph represented by:
# nodes: list of poses (x,y,yaw)
# edges: list of (i, j, relative_pose, information_matrix(3x3))
#
# Use least_squares to minimize sum || log( transform_inv( Tij ) * (T_i^-1 * T_j) ) ||_info
def residuals_for_optimization(flat_poses, edges, anchor_idx=0):
    # flat_poses: 3*N array
    N = flat_poses.size // 3
    poses = flat_poses.reshape((N,3))
    res = []
    for (i,j,rel,info) in edges:
        Ti = pose_to_transform(poses[i])
        Tj = pose_to_transform(poses[j])
        Tij_est = np.linalg.inv(Ti).dot(Tj)            # estimated relative
        Trel = pose_to_transform(rel)
        Td = np.linalg.inv(Trel).dot(Tij_est)         # error transform
        # map to minimal vector (dx,dy,dyaw)
        dx = Td[0,2]; dy = Td[1,2]; dyaw = math.atan2(Td[1,0], Td[0,0])
        vec = np.array([dx, dy, dyaw])
        # weigh by sqrt information (assume info is positive-def)
        # flatten: sqrt_info * vec
        sqrt_info = np.linalg.cholesky(info)
        r = sqrt_info.dot(vec)
        res.extend(r.tolist())
    # anchor the first pose to avoid gauge freedom by adding small residuals driving it to zero
    # (or we could fix it by removing its variables; here add tiny penalty)
    anchor_res = 1e-6 * poses[anchor_idx]
    res.extend(anchor_res.tolist())
    return np.array(res, dtype=np.float64)

def optimize_pose_graph(nodes, edges, iterations=50):
    if len(nodes) == 0:
        return nodes
    x0 = np.array(nodes).reshape(-1)
    def fun(x):
        return residuals_for_optimization(x, edges)
    res = least_squares(fun, x0, verbose=0, max_nfev=iterations)
    xopt = res.x.reshape((-1,3))
    return [tuple(x) for x in xopt]

# -------------------------- SLAM main class ---------------------------------
class LidarSLAM:
    def __init__(self, port=LIDAR_PORT):
        self.lidar = YDLidarX2(port)
        self.lidar.scale_factor = 0.15
        self.nodes = []         # poses (x,y,yaw)
        self.edges = []         # edges for pose graph
        self.keyframes = []     # each keyframe: dict{ 'pose_idx', 'points' (Nx2), 'pose' }
        self.global_points = [] # list of all transformed points
        self.lock = threading.Lock()
        self.running = False
        self.last_scan_time = None
        self.last_pose = np.array([0.0, 0.0, 0.0])
        self.cumulative_distance = 0.0
        self.last_keyframe_pose = self.last_pose.copy()
        # visualization
        self.vis_fig, self.vis_ax = plt.subplots(figsize=(8,6))
        plt.ion()

    def start(self):
        if not self.lidar.connect():
            raise RuntimeError("Lidar connect failed")
        if not self.lidar.start_scan():
            print("Warning: start_scan returned False")
        self.running = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def stop(self):
        self.running = False
        try:
            self.lidar.stop_scan()
        except Exception:
            pass
        try:
            self.lidar.disconnect()
        except Exception:
            pass

    def _read_loop(self):
        """Main loop: read scans, do ICP, create keyframes, add edges, optimize periodically."""
        scan_count = 0
        while self.running:
            if not self.lidar.available:
                time.sleep(0.002)
                continue
            distances = self.lidar.get_data()  # expected 360-length array of mm
            t_now = time.time()
            # convert to points (meters)
            pts = scan_to_points(range(360), distances, downsample=DOWNSAMPLE_ANGLE, out_of_range=self.lidar.out_of_range)
            if pts.shape[0] < 10:
                # ignore sparse scans
                continue

            # compute dt and odom-predicted pose change (assumed forward along x)
            if self.last_scan_time is None:
                dt = 1.0 / ASSUMED_SCAN_HZ
            else:
                dt = max(1e-6, t_now - self.last_scan_time)
            self.last_scan_time = t_now

            # simple odom delta: translate forward by speed * dt in the current body frame
            dx = ASSUMED_SPEED * dt
            init_delta = np.array([dx, 0.0, 0.0])   # small forward move (x in body frame)
            # rotate initial delta into world frame via last_pose yaw
            yaw = self.last_pose[2]
            d_world = np.array([
                init_delta[0]*math.cos(yaw) - init_delta[1]*math.sin(yaw),
                init_delta[0]*math.sin(yaw) + init_delta[1]*math.cos(yaw),
                init_delta[2]
            ])
            init_pred_pose = self.last_pose + d_world

            # if we have previous keyframe, use its points as target, else create first keyframe
            if len(self.keyframes) == 0:
                # create initial node and keyframe
                with self.lock:
                    self.nodes.append(tuple(self.last_pose))
                    kf = {'pose_idx': 0, 'points': pts.copy(), 'pose': tuple(self.last_pose)}
                    self.keyframes.append(kf)
                    # add points transformed to global
                    global_pts = apply_pose_to_points(pts, self.last_pose)
                    self.global_points.append(global_pts)
                    self.last_keyframe_pose = self.last_pose.copy()
                scan_count += 1
                continue

            # find most recent keyframe as ICP target
            target_kf = self.keyframes[-1]
            target_pts = target_kf['points']

            # initial transform guess from odometry: build 3x3 transform from relative pose
            rel_guess = np.array([init_pred_pose[0]-self.last_pose[0],
                                  init_pred_pose[1]-self.last_pose[1],
                                  init_pred_pose[2]-self.last_pose[2]])
            # build initial 3x3 transform matrix
            yaw_guess = rel_guess[2]
            Rg = np.array([[math.cos(yaw_guess), -math.sin(yaw_guess)],
                           [math.sin(yaw_guess),  math.cos(yaw_guess)]])
            T_init = np.eye(3)
            T_init[0:2,0:2] = Rg
            T_init[0:2,2] = rel_guess[0:2]

            # Run ICP: source = current scan pts, target = target_kf pts (both in local frames)
            T3, fitness, rmse = icp_2d(pts, target_pts, init_transform=T_init,
                                       max_iter=ICP_MAX_ITER, dist_thresh=ICP_DISTANCE_THRESHOLD)

            # Convert T3 (3x3) to delta pose in body frame
            delta_pose_est = transform_to_pose(T3)
            # apply delta to last node pose to get current pose estimate
            cur_pose = compose_pose(self.keyframes[-1]['pose'], delta_pose_est)

            # record node and edge
            with self.lock:
                node_idx = len(self.nodes)
                self.nodes.append(tuple(cur_pose))
                # relative pose from previous node index-1 to node_idx
                rel = delta_pose_est
                # information matrix (3x3) : approximate from RMSE (simple heuristic)
                info = np.eye(3) * max(1e-3, 1.0 / (rmse + 1e-6))
                self.edges.append((node_idx-1, node_idx, tuple(rel), info))

                # Add global transformed points to global_points list for visualization
                pts_global = apply_pose_to_points(pts, cur_pose)
                self.global_points.append(pts_global)

                # update distance since last keyframe
                dx_k = cur_pose[0] - self.last_keyframe_pose[0]
                dy_k = cur_pose[1] - self.last_keyframe_pose[1]
                self.cumulative_distance += math.hypot(dx_k, dy_k)
                self.last_keyframe_pose = cur_pose

                # Add keyframe if traveled enough
                if self.cumulative_distance >= KEYFRAME_SPACING:
                    kf = {'pose_idx': node_idx, 'points': pts.copy(), 'pose': tuple(cur_pose)}
                    self.keyframes.append(kf)
                    self.cumulative_distance = 0.0

            # detect loop closures against earlier keyframes
            # naive: check euclidean distance to keyframe poses
            with self.lock:
                if len(self.keyframes) > LOOP_CLOSURE_MIN_SEP:
                    cur_xy = np.array(cur_pose[0:2])
                    for ik, kf in enumerate(self.keyframes[:-LOOP_CLOSURE_MIN_SEP]):
                        kf_pose = np.array(kf['pose'][0:2])
                        dist = np.linalg.norm(cur_xy - kf_pose)
                        if dist < LOOP_CLOSURE_RADIUS:
                            # attempt ICP between current pts (in cur frame) and kf points (in kf frame)
                            T3_lc, fit_lc, rmse_lc = icp_2d(pts, kf['points'], init_transform=np.eye(3),
                                                            max_iter=30, dist_thresh=ICP_DISTANCE_THRESHOLD)
                            if fit_lc > 0.2 and rmse_lc < 0.2:
                                # we have a candidate loop constraint: compute relative pose rel_{kf -> cur}
                                rel_pose = transform_to_pose(T3_lc)
                                # add edge between kf.pose_idx and node_idx
                                info_lc = np.eye(3) * max(1e-3, 1.0 / (rmse_lc + 1e-6))
                                self.edges.append((kf['pose_idx'], node_idx, tuple(rel_pose), info_lc))
                                print(f"Loop closure added between {kf['pose_idx']} and {node_idx}; fit={fit_lc:.3f}, rmse={rmse_lc:.3f}")
                                # optimize pose graph immediately (could also be periodic)
                                # build node pose list
                                node_list = [np.array(p) for p in self.nodes]
                                optimized = optimize_pose_graph(node_list, self.edges, iterations=50)
                                # update nodes and recompute global_points according to optimized poses
                                self.nodes = [tuple(p) for p in optimized]
                                # rebuild global_points from keyframes using optimized poses
                                all_pts = []
                                for idx_kf, k in enumerate(self.global_points):
                                    # global_points list aligned with nodes, but easier to recompute:
                                    pass
                                # recompute global_points by transforming each recorded scan using optimized node poses
                                new_global = []
                                # NOTE: we did not store each scan's raw points historically (we stored keyframes and some global points)
                                # For simplicity we will rebuild global_points from keyframes only:
                                for kf in self.keyframes:
                                    pid = kf['pose_idx']
                                    p_est = self.nodes[pid]
                                    pts_t = apply_pose_to_points(kf['points'], p_est)
                                    new_global.append(pts_t)
                                self.global_points = new_global
                                break  # break loop closure search once found one

            self.last_pose = cur_pose.copy()
            scan_count += 1

        print("Read loop terminated")

    def _vis_loop(self):
        """Periodic visualization of accumulated global points and current pose graph."""
        while self.running:
            with self.lock:
                # combine all points
                if len(self.global_points) == 0:
                    time.sleep(LIVE_PLOT_UPDATE_SECONDS)
                    continue
                combined = np.vstack(self.global_points)
                self.vis_ax.clear()
                self.vis_ax.scatter(combined[:,0], combined[:,1], s=1)
                # draw poses
                xs = [p[0] for p in self.nodes]
                ys = [p[1] for p in self.nodes]
                self.vis_ax.plot(xs, ys, '-r')
                self.vis_ax.set_aspect('equal', 'box')
                self.vis_ax.set_title('SLAM Live View (points + poses)')
                plt.pause(0.001)
            time.sleep(LIVE_PLOT_UPDATE_SECONDS)

    def save_outputs(self):
        """Save CSV pointcloud and occupancy grid."""
        with self.lock:
            if len(self.global_points) == 0:
                print("No points to save")
                return
            all_pts = np.vstack(self.global_points)
        # save CSV (meters)
        with open(OUT_PCD_CSV, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x_m','y_m'])
            for x,y in all_pts:
                w.writerow([x,y])
        print(f"Saved point cloud to {OUT_PCD_CSV} ({all_pts.shape[0]} points)")

        # build occupancy grid
        xs = all_pts[:,0]; ys = all_pts[:,1]
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        pad = 0.5
        minx -= pad; miny -= pad; maxx += pad; maxy += pad
        nx = int(np.ceil((maxx-minx) / OCCUPANCY_RESOLUTION))
        ny = int(np.ceil((maxy-miny) / OCCUPANCY_RESOLUTION))
        grid = np.zeros((ny, nx), dtype=np.uint8)
        # convert points to grid cells and mark occupied
        ix = ((xs - minx) / OCCUPANCY_RESOLUTION).astype(int)
        iy = ((ys - miny) / OCCUPANCY_RESOLUTION).astype(int)
        # clamp indices
        valid = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)
        ix = ix[valid]; iy = iy[valid]
        grid[iy, ix] = 255
        # save image (flip vertically so Y increases upward)
        import imageio
        imageio.imwrite(OUT_OCCUPANCY_PNG, np.flipud(grid))
        print(f"Saved occupancy grid to {OUT_OCCUPANCY_PNG} (resolution {OCCUPANCY_RESOLUTION} m)")

# ------------------------------- Main --------------------------------------
def main():
    slam = LidarSLAM(port=LIDAR_PORT)
    try:
        slam.start()
        print("SLAM started. Press Ctrl+C to stop.")
        try:
            # run visualization in main thread (this uses plt.pause internally)
            slam._vis_loop()
        except KeyboardInterrupt:
            pass
    finally:
        slam.stop()
        # save outputs
        slam.save_outputs()
        print("Exiting.")

if __name__ == "__main__":
    main()
