'''import numpy as np
d = np.load('extracted/gmr_markers/BMCLab_SUB01_off_walk_1_canonical_markers_m.npz')
print(d.files)                    # 看有哪些字段: marker_names, positions, fps, src_fps, unit, source_file
print(d['marker_names'])
print(d['positions'].shape)
print(d['fps'])'''

import numpy as np

d = np.load("/home/mocap/scripts/data_analysis/extracted/carepd/BMCLab_SUB05_off_walk_1_canonical_keypoints_mm.npz", allow_pickle=True)

joint_names  = d["joint_names"]    # (29,) str, e.g. "pelvis", "left_hip", ...
positions    = d["positions"]      # (1371, 29, 3) float32, world-frame xyz
orientations = d["orientations"]   # (1371, 29, 4) float32, quaternion wxyz
fps          = float(d["fps"])
human_height = float(d["human_height"])
source_file  = str(d["source_file"])

# e.g. pelvis trajectory over time:
pelvis_idx = list(joint_names).index("pelvis")
pelvis_traj = positions[:, pelvis_idx, :]   

print("joint_names (number of frames, number of joints, dimensions):", joint_names)
print("positions.shape:", positions.shape)
print("orientations.shape:", orientations.shape)
print("fps:", fps)
print("human_height:", human_height)
print("source_file:", source_file)

print("pelvis trajectory shape:", pelvis_traj.shape)
print("pelvis trajectory (first 5 frames):", pelvis_traj[:5])
