import numpy as np

d = np.load(
    "extracted/mujoco_markers/mujoco_markers_sub05walk1.npz",
    allow_pickle=True,
)

marker_names = d["marker_names"]   # (39,) str — Vicon marker labels, e.g. "LASI", "RKNE", "C7"
positions    = d["positions"]      # (3830, 39, 3) float32, world-frame xyz
t            = d["t"]              # (3830,) float32 — per-frame timestamps (seconds)
fps          = float(d["fps"])
unit         = str(d["unit"])
source_file  = str(d["source_file"])

print("marker_names (number of frames, number of markers, dimensions):", marker_names)
print("positions.shape:", positions.shape)
print("t.shape:", t.shape)
print("fps:", fps)
print("unit:", unit)
print("source_file:", source_file)

pelvis_idx = list(marker_names).index("LASI")
pelvis_traj = positions[:, pelvis_idx, :] 

print("LASI trajectory (first 5 frames):", pelvis_traj[:5])

