import numpy as np

d = np.load(
    "extracted/gmr/gmr_BMCLab_SUB01_off_walk_1_canonical_keypoints_m.npz",
    allow_pickle=True,
)

joint_names      = d["joint_names"]        # (19,) str — keypoint names
positions        = d["positions"]          # (133, 19, 3) float32, world-frame xyz
orientations     = d["orientations"]       # (133, 19, 4) float32, quaternion wxyz
robot_body_names = d["robot_body_names"]   # (19,) str — actual robot body each keypoint was read from
dof_pos          = d["dof_pos"]            # (133, 61) float32 — robot joint angles (radians)
fps              = float(d["fps"])
unit             = str(d["unit"])          # "m" here (filename suffix confirms it)
source_file      = str(d["source_file"])

pelvis_idx = list(joint_names).index("pelvis")
pelvis_traj = positions[:, pelvis_idx, :]

print("joint_names (number of frames, number of joints, dimensions):", joint_names)
print("positions.shape:", positions.shape)
print("orientations.shape:", orientations.shape)
print("robot_body_names:", robot_body_names)
print("dof_pos.shape:", dof_pos.shape)
print("fps:", fps)
print("unit:", unit)
print("source_file:", source_file)

print("pelvis trajectory shape:", pelvis_traj.shape)
print("pelvis trajectory (first 5 frames):", pelvis_traj[:5])
