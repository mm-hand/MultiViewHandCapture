"""
Bone Length Constraint Module
Handles calibration and correction of hand bone lengths during tracking.
Maintains skeletal structure consistency by enforcing calibrated lengths.
"""

import numpy as np
from collections import defaultdict
from config import FINGER_CHAINS, CALIBRATION_FRAMES


class BoneLengthCalibrator:
    """
    Accumulates and computes average bone lengths from stereo triangulation.
    Used during the first CALIBRATION_FRAMES frames to establish skeletal baseline.
    """
    
    def __init__(self):
        """Initialize the accumulator dictionary."""
        # Create bone pairs from all finger chains
        # Each bone is a tuple (start_idx, end_idx)
        self.length_accumulator = defaultdict(list)
    
    def accumulate_frame(self, pts3d):
        """
        Accumulate bone lengths from a single frame of 3D triangulated points.
        
        Args:
            pts3d: 3D hand landmarks (20, 3) from stereo triangulation
        """
        for chain in FINGER_CHAINS.values():
            for i in range(len(chain) - 1):
                start_idx = chain[i]
                end_idx = chain[i + 1]
                bone_pair = (start_idx, end_idx)
                
                # Compute Euclidean distance
                distance = np.linalg.norm(pts3d[end_idx] - pts3d[start_idx])
                self.length_accumulator[bone_pair].append(distance)
    
    def finalize_calibration(self):
        """
        Compute final calibrated bone lengths by averaging accumulated values.
        
        Returns:
            Dictionary {(start_idx, end_idx): mean_length_mm, ...}
        """
        bone_lengths_final = {}
        for bone_pair, distances in self.length_accumulator.items():
            if len(distances) > 0:
                # Use mean of all accumulated measurements
                bone_lengths_final[bone_pair] = np.mean(distances)
        
        return bone_lengths_final
    
    def get_status(self):
        """Return current calibration status."""
        return {
            'num_bones': len(self.length_accumulator),
            'sample_counts': {k: len(v) for k, v in self.length_accumulator.items()}
        }


class BoneLengthCorrector:
    """
    Applies bone length constraints to correct 3D hand pose after triangulation.
    Uses forward kinematic chain propagation from wrist to fingertips.
    
    NOTE: This maintains the original simple length constraint approach.
    For more sophisticated handling, see constraint_solver.py for integrated optimization.
    """
    
    def __init__(self, bone_lengths_ref):
        """
        Args:
            bone_lengths_ref: Dictionary {(start_idx, end_idx): reference_length_mm, ...}
        """
        self.bone_lengths_ref = bone_lengths_ref
    
    def correct_pose(self, pts3d):
        """
        Apply bone length constraints via forward chain correction.
        Iterates through each finger chain from wrist to fingertip,
        enforcing that each bone maintains its reference length.
        
        Args:
            pts3d: 3D hand landmarks (20, 3)
        
        Returns:
            Corrected 3D landmarks (20, 3) with enforced bone lengths
        """
        corrected = pts3d.copy()
        
        # Process each finger chain independently
        for chain in FINGER_CHAINS.values():
            # Forward propagation: from wrist to fingertip
            for i in range(len(chain) - 1):
                start_idx = chain[i]
                end_idx = chain[i + 1]
                bone_pair = (start_idx, end_idx)
                
                # Get reference length
                if bone_pair not in self.bone_lengths_ref:
                    continue
                ref_length = self.bone_lengths_ref[bone_pair]
                
                # Vector from start to end
                v = corrected[end_idx] - corrected[start_idx]
                current_length = np.linalg.norm(v)
                
                # Avoid division by zero
                if current_length < 1e-8:
                    continue
                
                # Normalize and scale to reference length
                direction = v / current_length
                corrected[end_idx] = corrected[start_idx] + direction * ref_length
        
        return corrected


def create_bone_pairs_from_chains():
    """
    Helper function to generate all bone pairs from finger chains.
    Useful for initialization and debugging.
    
    Returns:
        List of bone pairs [(start_idx, end_idx), ...]
    """
    bone_pairs = []
    for chain in FINGER_CHAINS.values():
        for i in range(len(chain) - 1):
            bone_pairs.append((chain[i], chain[i + 1]))
    return bone_pairs
