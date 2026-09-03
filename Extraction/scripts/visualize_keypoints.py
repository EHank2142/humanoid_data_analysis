"""
visualize_keypoints.py

Renders the *_keypoints.npz produced by carepd_extract_keypoints.py (or
the corresponding GMR/robot version, as long as the npz has joint_names +
positions fields) into a 3D skeleton animation gif, for visually verifying
that the keypoint trajectories extracted by Part A are sane (skeleton
proportions correct, no jitter/clipping, walking posture looks like the
original gait).

Only depends on numpy + matplotlib, no smplx / general_motion_retargeting
needed -- can run in any environment (no need for conda activate gmr).

Bone connections use the SMPL-X body model's parent-child relationships
(the tree of 22 body joints). If the npz only has a subset of keypoints
(e.g. GMR/robot-side extraction, which has no ankle/collar keypoints --
retargeting's IK maps knee straight to foot, and shoulder straight to
spine3), each keypoint's bone is drawn to its nearest *present* ancestor
up the SMPL-X chain instead of requiring its immediate literal parent --
otherwise those keypoints would end up as bone-less floating dots even
though their position data is perfectly fine.

Usage:
    python visualize_keypoints.py \
        --keypoints_npz extracted/carepd/BMCLab_SUB05_off_walk_1_canonical_keypoints.npz \
        --out_gif checks/carepd_sub05walk1_skeleton.gif

    # For clips with many frames, skip frames to speed up rendering / shrink the gif
    python visualize_keypoints.py --keypoints_npz ... --out_gif ... --stride 2
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parent-child relationships of SMPL-X's first 22 body joints, from smplx
# body_model.parents -- the *full* canonical chain, used to find each
# keypoint's nearest present ancestor (see find_bone_parent below), not
# applied as a literal "must have exactly this parent" requirement.
PARENTS = {
    "left_hip": "pelvis", "right_hip": "pelvis", "spine1": "pelvis",
    "left_knee": "left_hip", "right_knee": "right_hip", "spine2": "spine1",
    "left_ankle": "left_knee", "right_ankle": "right_knee", "spine3": "spine2",
    "left_foot": "left_ankle", "right_foot": "right_ankle",
    "neck": "spine3", "left_collar": "spine3", "right_collar": "spine3",
    "head": "neck", "left_shoulder": "left_collar", "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder", "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow", "right_wrist": "right_elbow",
    # Foot landmark points, not on the kinematic chain; attached under
    # their corresponding foot joint for drawing purposes
    "left_heel": "left_foot", "left_big_toe": "left_foot", "left_small_toe": "left_foot",
    "right_heel": "right_foot", "right_big_toe": "right_foot", "right_small_toe": "right_foot",
    # Oli's single-toe-contact naming (GMR/robot-side extraction) -- not an
    # SMPL-X name, added so it still gets a bone instead of floating
    "left_toe": "left_foot", "right_toe": "right_foot",
    # Not a real bone -- sacrum is a stand-in landmark co-located with
    # pelvis on both CARE-PD and GMR sides (see extraction script docstrings)
    "sacrum": "pelvis",
}


def find_bone_parent(joint_name, present, parents_dict):
    """Walk up parents_dict from joint_name until hitting a keypoint that's
    actually present in this npz, so reduced keypoint sets (e.g. GMR's,
    which has no ankle/collar) still get a sensible bone instead of the
    joint ending up disconnected."""
    seen = set()
    current = joint_name
    while current in parents_dict:
        parent = parents_dict[current]
        if parent in present:
            return parent
        if parent in seen:
            return None
        seen.add(parent)
        current = parent
    return None

# Vicon Plug-in Gait Full Body marker set (39 markers) -- the <site>-level
# extraction (gmr_extract_markers.py), stored under the "marker_names" field
# instead of "joint_names". Skin markers, not joint centers, so this is a
# different (denser, more literal) stick figure than PARENTS above -- not
# meant to line up bone-for-bone with the 19 SMPL-X joint keypoints.
MARKER_PARENTS = {
    # Head band
    "RFHD": "LFHD", "LBHD": "LFHD", "RBHD": "RFHD",
    "LFHD": "C7", "RFHD": "C7",
    # Torso loop + back marker
    "CLAV": "C7", "STRN": "CLAV", "T10": "STRN", "RBAK": "C7",
    # Pelvis loop, bridged to torso via T10
    "RASI": "LASI", "LPSI": "LASI", "RPSI": "RASI",
    "LASI": "T10",
    # Legs
    "LTHI": "LASI", "LKNE": "LTHI", "LTIB": "LKNE", "LANK": "LTIB",
    "LHEE": "LANK", "LTOE": "LANK",
    "RTHI": "RASI", "RKNE": "RTHI", "RTIB": "RKNE", "RANK": "RTIB",
    "RHEE": "RANK", "RTOE": "RANK",
    # Arms, attached at the shoulder/clavicle level
    "LSHO": "CLAV", "LUPA": "LSHO", "LELB": "LUPA", "LFRM": "LELB",
    "LWRA": "LFRM", "LWRB": "LWRA", "LFIN": "LWRA",
    "RSHO": "CLAV", "RUPA": "RSHO", "RELB": "RUPA", "RFRM": "RELB",
    "RWRA": "RFRM", "RWRB": "RWRA", "RFIN": "RWRA",
}

COLOR_LEFT = "#2a78d6"   # left limb
COLOR_RIGHT = "#eb6834"  # right limb
COLOR_AXIAL = "#52514e"  # axial (spine/head/pelvis), not left/right specific


def bone_color(joint_name):
    if joint_name.startswith("left_"):
        return COLOR_LEFT
    if joint_name.startswith("right_"):
        return COLOR_RIGHT
    return COLOR_AXIAL


def marker_bone_color(marker_name):
    if marker_name.startswith("L"):
        return COLOR_LEFT
    if marker_name.startswith("R"):
        return COLOR_RIGHT
    return COLOR_AXIAL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints_npz", required=True)
    parser.add_argument("--out_gif", required=True)
    parser.add_argument("--stride", type=int, default=1,
                         help="Take every Nth frame, to speed up rendering for long clips / many keypoints")
    parser.add_argument("--follow", action="store_true", default=True,
                         help="Camera follows the pelvis, zoomed in to see joint/limb detail (on by default)")
    parser.add_argument("--no-follow", dest="follow", action="store_false",
                         help="Camera fixed in world coordinates, for viewing overall displacement/stride (turns follow off)")
    args = parser.parse_args()

    d = np.load(args.keypoints_npz, allow_pickle=True)
    if "marker_names" in d.files:
        # Vicon Plug-in Gait <site>-level extraction (gmr_extract_markers.py)
        joint_names = list(d["marker_names"])
        parents_dict = MARKER_PARENTS
        color_fn = marker_bone_color
        root_name = "C7"  # apex of the torso chain -- has no parent in MARKER_PARENTS
    else:
        # SMPL-X joint-center extraction (gmr_extract_keypoints.py / carepd_extract_keypoints.py)
        joint_names = list(d["joint_names"])
        parents_dict = PARENTS
        color_fn = bone_color
        root_name = "pelvis"
    positions = d["positions"][:: args.stride]  # (T, J, 3)
    fps = float(d["fps"]) / args.stride

    present = set(joint_names)
    bones = []
    for j in joint_names:
        p = find_bone_parent(j, present, parents_dict)
        if p is not None:
            bones.append((j, p))
    bone_idx = [(joint_names.index(j), joint_names.index(p)) for j, p in bones]
    bone_colors = [color_fn(j) for j, p in bones]
    unconnected = [j for j in joint_names
                   if j != root_name and find_bone_parent(j, present, parents_dict) is None]
    if unconnected:
        print(f"[warn] no ancestor found in this keypoint set for: {unconnected} "
              f"(will still show as scatter points, just no bone line)")
    print(f"[info] {len(joint_names)} keypoints, {len(bones)} bone connections, "
          f"{len(positions)} frames @ {fps:.2f}fps, follow={args.follow}")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z (up)")

    if args.follow:
        # Camera re-centers on the pelvis every frame; span is computed from
        # the body's own maximum extent across all frames, so the figure
        # always fills the frame and joint angles/limb motion are clearly visible.
        per_frame_center = positions.mean(axis=1, keepdims=True)  # (T, 1, 3)
        span = (positions - per_frame_center).reshape(-1, 3)
        span = np.abs(span).max() + 0.15
    else:
        # Fixed world-coordinate range, frame doesn't jump around, for
        # viewing the displacement path/stride over the whole clip
        all_pts = positions.reshape(-1, 3)
        world_center = all_pts.mean(axis=0)
        span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 + 0.1

    scatter = ax.scatter([], [], [], s=20, c="#0b0b0b", depthshade=False)
    lines = [ax.plot([], [], [], lw=2, color=c)[0] for c in bone_colors]
    title = ax.set_title("")

    def update(frame_idx):
        pts = positions[frame_idx]
        center = pts.mean(axis=0) if args.follow else world_center
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for line, (j, p) in zip(lines, bone_idx):
            line.set_data([pts[j, 0], pts[p, 0]], [pts[j, 1], pts[p, 1]])
            line.set_3d_properties([pts[j, 2], pts[p, 2]])
        title.set_text(f"frame {frame_idx}/{len(positions)}  t={frame_idx / fps:.2f}s")
        return [scatter, *lines, title]

    anim = FuncAnimation(fig, update, frames=len(positions), interval=1000 / fps, blit=False)

    out_path = Path(args.out_gif)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Finished, written to {out_path}")


if __name__ == "__main__":
    main()
