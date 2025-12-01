"""
Multi-Constraint Hand Pose Optimization Solver
Simultaneously optimizes for:
1. 2D reprojection consistency (observation fidelity)
2. Bone length constraints (skeletal structure)
3. Joint angle limits (biomechanical constraints)

Uses scipy.optimize.minimize with SLSQP algorithm.
"""

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from config import (
    FINGER_CHAINS, JOINT_ANGLE_LIMITS, JOINT_CONFIG_MAP,
    OPTIMIZER_WEIGHTS, OPTIMIZER_SIGMA, OPTIMIZER_CONFIG,
    WRIST, INDEX_MCP, PINKY_MCP, WEIGHT_SCHEDULER_CONFIG
)


class MultiConstraintHandSolver:
    """
    Unified optimization solver combining three objectives:
    - E_reproj: 2D reprojection error (minimize distance between projected 3D and observed 2D)
    - E_bone: bone length deviations (keep skeletal structure consistent)
    - E_angle: joint angle constraint violations (enforce biomechanical limits)
    
    Solved via scipy.optimize.minimize with SLSQP algorithm.
    """
    
    def __init__(self, K1, K2, D1, D2, P1, P2, 
                 bone_lengths_ref, joint_limits=None):
        """
        Args:
            K1, K2: Camera intrinsic matrices (3x3)
            D1, D2: Distortion coefficients (1D arrays, typically 5 elements)
            P1, P2: Projection matrices (3x4)
            bone_lengths_ref: Dictionary {(s, e): length_mm, ...}
            joint_limits: Dictionary of joint angle limits (from config)
        """
        self.K1 = K1
        self.K2 = K2
        self.D1 = D1
        self.D2 = D2
        self.P1 = P1
        self.P2 = P2
        self.bone_lengths_ref = bone_lengths_ref
        self.joint_limits = joint_limits if joint_limits is not None else JOINT_ANGLE_LIMITS
        self.finger_chains = FINGER_CHAINS
        
        # ========== Load weights from config ==========
        self.w_reproj = OPTIMIZER_WEIGHTS['w_reproj']
        self.w_bone = OPTIMIZER_WEIGHTS['w_bone']
        self.w_angle = OPTIMIZER_WEIGHTS['w_angle']
        
        # ========== Load normalization sigmas from config ==========
        self.sigma_reproj = OPTIMIZER_SIGMA['sigma_reproj']
        self.sigma_bone = OPTIMIZER_SIGMA['sigma_bone']
        self.sigma_angle = OPTIMIZER_SIGMA['sigma_angle']
        
        # Storage for diagnostic info
        self.last_reproj_error = 0
        self.last_bone_error = 0
        self.last_angle_error = 0
    
    def reproject_points(self, pts3d_flat, pts2d_obs, camera='left'):
        """
        Project 3D points to 2D image and compute reprojection error.
        
        Args:
            pts3d_flat: Flattened 3D points (63,) = 21 joints × 3 coords
            pts2d_obs: Observed 2D points from MediaPipe (21, 2) in pixels
            camera: 'left' or 'right'
        
        Returns:
            Reprojection error (sum of squared pixel distances)
        """
        pts3d = pts3d_flat.reshape(21, 3)
        
        # Select camera parameters
        if camera == 'left':
            P = self.P1
            K = self.K1
            D = self.D1
        else:
            P = self.P2
            K = self.K2
            D = self.D2
        
        # Project to homogeneous coordinates
        ones = np.ones((pts3d.shape[0], 1))
        pts3d_homo = np.hstack([pts3d, ones]) 
        pts_homo = (P @ pts3d_homo.T).T
        
        # Convert to normalized image coordinates
        pts_2d_norm = pts_homo[:, :2] / (pts_homo[:, 2:3] + 1e-8)  # (21, 2)
        
        # Apply distortion
        pts_2d_distorted = self._apply_distortion(pts_2d_norm, K, D)
        
        # Compute error
        error = np.sum((pts_2d_distorted - pts2d_obs) ** 2)
        
        return error
    
    def _apply_distortion(self, pts_norm, K, D):
        """
        Apply OpenCV distortion model to normalized image coordinates.
        
        Args:
            pts_norm: Normalized image coordinates (N, 2)
            K: Camera intrinsic matrix (3, 3)
            D: Distortion coefficients
        
        Returns:
            Distorted pixel coordinates (N, 2)
        """
        D = np.asarray(D).flatten()

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Normalize from pixel to principal point offset
        x = (pts_norm[:, 0] - cx) / fx
        y = (pts_norm[:, 1] - cy) / fy
        
        # Compute radius squared
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2 * r4
        
        # Radial distortion coefficients
        k1 = D[0] if len(D) > 0 else 0
        k2 = D[1] if len(D) > 1 else 0
        k3 = D[4] if len(D) > 4 else 0
        
        # Radial distortion factor
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        
        # Tangential distortion coefficients
        p1 = D[2] if len(D) > 2 else 0
        p2 = D[3] if len(D) > 3 else 0
        
        # Tangential distortion
        x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_distorted = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        
        # Convert back to pixel coordinates
        u = fx * x_distorted + cx
        v = fy * y_distorted + cy
        
        return np.column_stack([u, v])
    
    def compute_bone_length_error(self, pts3d_flat):
        """
        Compute bone length constraint violation.
        
        Measures deviation of actual distances from reference lengths.
        """
        pts3d = pts3d_flat.reshape(21, 3)
        error = 0.0
        
        for chain in self.finger_chains.values():
            for i in range(len(chain) - 1):
                idx_s = chain[i]
                idx_e = chain[i + 1]
                bone_pair = (idx_s, idx_e)
                
                actual_length = np.linalg.norm(pts3d[idx_e] - pts3d[idx_s])
                ref_length = self.bone_lengths_ref.get(bone_pair, actual_length)
                
                error += (actual_length - ref_length) ** 2
        
        return error
    
    def compute_joint_angle_error(self, pts3d_flat):
        """
        Compute joint angle constraint violations.
        
        For each joint, checks if angle exceeds flexion or extension limits.
        Returns penalty for violations.
        """
        pts3d = pts3d_flat.reshape(21, 3)
        
        # Compute stable palm basis
        p_wrist = pts3d[WRIST]
        p_index_mcp = pts3d[INDEX_MCP]
        p_pinky_mcp = pts3d[PINKY_MCP]
        
        vec_hand_dir = self._normalize(p_index_mcp - p_wrist)
        vec_transverse = self._normalize(p_pinky_mcp - p_index_mcp)
        palm_normal = self._normalize(np.cross(vec_hand_dir, vec_transverse))
        
        error = 0.0
        
        for finger_name, chain in self.finger_chains.items():
            # Iterate through joints (pivot joints from index 1 to len-2)
            for i in range(1, len(chain) - 1):
                idx_parent = chain[i - 1]
                idx_pivot = chain[i]
                idx_child = chain[i + 1]
                
                p_parent = pts3d[idx_parent]
                p_pivot = pts3d[idx_pivot]
                p_child = pts3d[idx_child]
                
                # Bone vectors
                u = p_pivot - p_parent
                v = p_child - p_pivot
                
                u_norm = self._normalize(u)
                v_norm = self._normalize(v)
                
                # Compute stable hinge axis
                if finger_name == "thumb":
                    hinge_axis = palm_normal
                else:
                    hinge_axis = self._normalize(np.cross(u_norm, palm_normal))
                
                # Project v onto flexion plane
                proj_v = v_norm - hinge_axis * np.dot(v_norm, hinge_axis)
                proj_v = self._normalize(proj_v)
                
                # Reference (zero) direction
                ref_zero = u_norm
                
                # Compute angle magnitude
                cos_angle = np.clip(np.dot(ref_zero, proj_v), -1.0, 1.0)
                angle_mag = np.degrees(np.arccos(cos_angle))
                
                # Determine sign
                cross_check = np.cross(ref_zero, proj_v)
                sign_check = np.dot(cross_check, hinge_axis)
                signed_angle = angle_mag if sign_check > 0 else -angle_mag
                
                # Get joint limits
                joint_config = self._get_joint_config(idx_pivot)
                if joint_config is None:
                    continue
                
                finger_name_cfg, joint_type = joint_config
                limits = self.joint_limits[finger_name_cfg][joint_type]
                max_flex = limits["flexion"]
                max_ext = limits["extension"]
                
                # Compute constraint violation penalty
                if signed_angle > max_flex:
                    violation = signed_angle - max_flex
                    error += violation ** 2
                elif signed_angle < -max_ext:
                    violation = -max_ext - signed_angle
                    error += violation ** 2
        
        return error
    
    def objective_function(self, pts3d_flat, pts2d_l_obs, pts2d_r_obs):
        """
        Complete objective function combining all three constraints.
        
        Args:
            pts3d_flat: Flattened 3D points (60,)
            pts2d_l_obs: Observed 2D points left camera (20, 2)
            pts2d_r_obs: Observed 2D points right camera (20, 2)
        
        Returns:
            Weighted sum of normalized errors
        """
        # ========== Constraint 1: Reprojection consistency ==========
        reproj_error_l = self.reproject_points(pts3d_flat, pts2d_l_obs, camera='left')
        reproj_error_r = self.reproject_points(pts3d_flat, pts2d_r_obs, camera='right')
        E_reproj = reproj_error_l + reproj_error_r
        
        # ========== Constraint 2: Bone length ==========
        E_bone = self.compute_bone_length_error(pts3d_flat)
        
        # ========== Constraint 3: Joint angle ==========
        E_angle = self.compute_joint_angle_error(pts3d_flat)
        
        # Store for diagnostics
        self.last_reproj_error = E_reproj
        self.last_bone_error = E_bone
        self.last_angle_error = E_angle
        
        # ========== Weighted sum with normalization ==========
        E_total = (
            self.w_reproj * (E_reproj / (self.sigma_reproj ** 2)) +
            self.w_bone * (E_bone / (self.sigma_bone ** 2)) +
            self.w_angle * (E_angle / (self.sigma_angle ** 2))
        )
        
        return E_total
    
    def solve(self, pts3d_initial, pts2d_l_obs, pts2d_r_obs):
        """
        Execute optimization to solve for corrected 3D hand pose.
        
        Args:
            pts3d_initial: Initial 3D points (20, 3)
            pts2d_l_obs: Observed 2D points left camera (20, 2)
            pts2d_r_obs: Observed 2D points right camera (20, 2)
        
        Returns:
            Dictionary with optimized 3D points and convergence info
        """
        pts3d_flat = pts3d_initial.flatten()
        
        # Configure optimizer
        options = {
            'maxiter': OPTIMIZER_CONFIG['max_iterations'],
            'ftol': OPTIMIZER_CONFIG['ftol'],
            'disp': False
        }
        
        # Run optimization
        result = minimize(
            self.objective_function,
            pts3d_flat,
            args=(pts2d_l_obs, pts2d_r_obs),
            method=OPTIMIZER_CONFIG['method'],
            options=options
        )
        
        pts3d_optimized = result.x.reshape(21, 3)
        
        return {
            'pts3d': pts3d_optimized,
            'success': result.success,
            'fun': result.fun,
            'nit': result.nit,
            'msg': result.message,
            'reproj_error': self.last_reproj_error,
            'bone_error': self.last_bone_error,
            'angle_error': self.last_angle_error
        }
    
    def _normalize(self, v):
        """Safely normalize a vector."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 0.0])
        return v / norm
    
    def _get_joint_config(self, idx):
        """Get joint configuration (finger name, joint type) for a given index."""
        return JOINT_CONFIG_MAP.get(idx, None)


class ReprojectionValidator:
    """
    Validates optimized 3D hand against original 2D observations.
    Detects anomalies and provides quality metrics.
    """
    
    def __init__(self, K1, K2, D1, D2, P1, P2):
        """
        Args:
            K1, K2, D1, D2, P1, P2: Camera calibration parameters
        """
        self.K1 = K1
        self.K2 = K2
        self.D1 = D1
        self.D2 = D2
        self.P1 = P1
        self.P2 = P2
        self.max_reproj_error = OPTIMIZER_CONFIG['reproj_error_threshold']
    
    def validate(self, pts3d, pts2d_l_obs, pts2d_r_obs):
        """
        Validate 3D hand pose by computing reprojection errors.
        
        Args:
            pts3d: 3D points (20, 3)
            pts2d_l_obs: Observed 2D left (20, 2)
            pts2d_r_obs: Observed 2D right (20, 2)
        
        Returns:
            Dictionary with validation metrics
        """
        pts2d_l_proj = self._reproject(pts3d, self.P1, self.K1, self.D1)
        pts2d_r_proj = self._reproject(pts3d, self.P2, self.K2, self.D2)
        
        # Compute per-joint errors
        error_l = np.linalg.norm(pts2d_l_proj - pts2d_l_obs, axis=1)
        error_r = np.linalg.norm(pts2d_r_proj - pts2d_r_obs, axis=1)
        
        # Identify outliers
        outlier_indices = np.where(
            (error_l > self.max_reproj_error) | 
            (error_r > self.max_reproj_error)
        )[0]
        
        mean_error_l = np.mean(error_l)
        mean_error_r = np.mean(error_r)
        
        diagnosis = {
            'mean_error_l': mean_error_l,
            'mean_error_r': mean_error_r,
            'max_error_l': np.max(error_l),
            'max_error_r': np.max(error_r),
            'per_joint_error_l': error_l,
            'per_joint_error_r': error_r,
            'outlier_indices': outlier_indices,
            'is_valid': (mean_error_l < OPTIMIZER_CONFIG['reproj_quality_threshold']) and 
                       (mean_error_r < OPTIMIZER_CONFIG['reproj_quality_threshold']),
            'outlier_count': len(outlier_indices)
        }
        
        return diagnosis
    
    def _reproject(self, pts3d, P, K, D):
        """Project 3D points to 2D with distortion."""
        ones = np.ones((pts3d.shape[0], 1))
        pts3d_homo = np.hstack([pts3d, ones])
        
        pts_homo = (P @ pts3d_homo.T).T
        
        pts_norm = pts_homo[:, :2] / (pts_homo[:, 2:3] + 1e-8)
        pts_distorted = self._apply_distortion(pts_norm, K, D)
        return pts_distorted
    
    def _apply_distortion(self, pts_norm, K, D):
        """Apply OpenCV distortion model."""
        D = np.asarray(D).flatten()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x = (pts_norm[:, 0] - cx) / fx
        y = (pts_norm[:, 1] - cy) / fy
        
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2 * r4
        
        k1 = D[0] if len(D) > 0 else 0
        k2 = D[1] if len(D) > 1 else 0
        k3 = D[4] if len(D) > 4 else 0
        
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        
        p1 = D[2] if len(D) > 2 else 0
        p2 = D[3] if len(D) > 3 else 0
        
        x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_distorted = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        
        u = fx * x_distorted + cx
        v = fy * y_distorted + cy
        
        return np.column_stack([u, v])


class AdaptiveWeightScheduler:
    """
    Dynamically adjusts constraint weights during tracking based on
    pose quality and reprojection errors.
    """
    
    def __init__(self):
        """Initialize with default weights from config."""
        self.w_reproj = OPTIMIZER_WEIGHTS['w_reproj']
        self.w_bone = OPTIMIZER_WEIGHTS['w_bone']
        self.w_angle = OPTIMIZER_WEIGHTS['w_angle']
    
    def update_weights(self, frame_idx, diagnosis, solver):
        """
        Update weights based on validation diagnosis.
        
        Strategy:
        1. Early frames: emphasize 2D consistency
        2. Later frames: balance all three constraints
        3. If reprojection error high: increase w_reproj
        4. If many outliers: increase w_bone
        
        Args:
            frame_idx: Current frame number
            diagnosis: Validation diagnostics
            solver: Reference to solver for updating weights
        """
        cfg = WEIGHT_SCHEDULER_CONFIG
        
        # Strategy 1: Warm-up phase
        if frame_idx < cfg['warmup_frames']:
            alpha = frame_idx / cfg['warmup_frames']
            self.w_angle = 0.1 + 0.4 * alpha
        
        # Strategy 2: Adjust based on reprojection quality
        total_error = diagnosis['mean_error_l'] + diagnosis['mean_error_r']
        
        if total_error > cfg['reproj_error_high']:
            # Poor 2D alignment
            self.w_reproj = min(
                self.w_reproj * cfg['w_reproj_increase_factor'],
                3.0
            )
        elif total_error < cfg['reproj_error_low']:
            # Good 2D alignment
            self.w_reproj = max(
                self.w_reproj * cfg['w_reproj_decrease_factor'],
                0.5
            )
        
        # Strategy 3: Adjust based on outliers
        if diagnosis['outlier_count'] > cfg['outlier_threshold']:
            # Many outliers - trust structure more
            self.w_bone = min(
                self.w_bone * cfg['w_bone_increase_factor'],
                2.0
            )
            self.w_reproj = max(
                self.w_reproj * cfg['w_reproj_outlier_decrease_factor'],
                0.5
            )
        elif diagnosis['outlier_count'] == 0:
            # No outliers - can trust observations
            self.w_reproj = min(
                self.w_reproj * cfg['w_reproj_no_outlier_increase_factor'],
                3.0
            )
        
        # Apply updated weights to solver
        solver.w_reproj = self.w_reproj
        solver.w_bone = self.w_bone
        solver.w_angle = self.w_angle
        
        return {
            'w_reproj': self.w_reproj,
            'w_bone': self.w_bone,
            'w_angle': self.w_angle
        }
