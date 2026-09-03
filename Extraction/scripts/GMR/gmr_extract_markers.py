"""
gmr_extract_markers.py

[Part A / GMR side, marker-level] Extract world-space trajectories of the
39 Vicon Plug-in Gait Full Body markers from GMR retargeting output
(.pkl: root_pos / root_rot / dof_pos / fps -- the files in
/home/mocap/whole_body_tracking/motions/*.pkl), by forward-kinematics
through the same <site> elements added to HU_D04_01.xml for the
sim-vs-real marker comparison (see mujoco/sim_mujoco.py's marker logger,
which records these same 39 sites live off a running policy rollout).

Why this is a separate script from gmr_extract_keypoints.py
----------------------------------
gmr_extract_keypoints.py extracts *joint centers* (pelvis, hip, knee, ...)
via ik_match_table1's human-keypoint -> robot-body correspondence -- that
answers "did retargeting's IK land the joint where it was supposed to".
This script instead extracts *skin markers* (LKNE, LASI, C7, ...) via the
<site> elements placed on the mesh surface -- these have no correspondence
in ik_match_table1 (retargeting never targeted them), so they measure a
different thing: "if you'd put real Vicon markers on this robot, where
would they end up", directly comparable to a real .c3d marker file
(e.g. the CARE-PD raw captures under /home/mocap/Downloads/C3D_Files/)
without going through any joint-center abstraction. Position only, no
orientation -- physical markers don't carry orientation, matching what a
.c3d file records.

Robot xml note
----------------------------------
GMR's own copy (GMR_Repo/GMR/assets/oli/oli.xml) does not have these
marker sites (only whole_body_tracking's deployment copy does), but both
files describe the identical robot skeleton -- verified the <joint> name
order matches exactly between the two files -- so it's safe to run FK
for a GMR-produced .pkl through the whole_body_tracking HU_D04_01.xml
copy instead; dof_pos lines up the same way in both.

Output .npz fields:
    marker_names  : (39,) str          Vicon Plug-in Gait Full Body marker
                                        names (LFHD, RFHD, ..., RTOE)
    positions     : (T, 39, 3) float32 world-frame marker positions, unit
                                        per the unit field
    fps           : float              fps as stored in the motion file
    src_fps       : float              same as fps (GMR .pkl doesn't retain
                                        a separate pre-resampling rate)
    unit          : str                length unit for positions, m/cm/mm
    source_file   : str                path to the input .pkl

How many markers does the output have?
Fixed at 39 by MARKER_NAMES below (the Vicon Plug-in Gait Full Body set) --
it does NOT depend on which .pkl you point this at. The input motion file
only supplies the trajectory (root_pos/root_rot/dof_pos) driven through
FK; it has no say in which sites get read out. The count only changes if
you pass --markers with a different subset, or if --robot_xml points at a
model missing some of these <site> names (raises ValueError listing the
missing ones -- see resolve_marker_site_ids -- not a silently smaller
count).

Usage (only needs mujoco + numpy, no smplx/torch/GMR environment required):
    python gmr_extract_markers.py \
        --motion_file /home/mocap/whole_body_tracking/motions/BMCLab_SUB01_off_walk_1_canonical.pkl \
        --out_dir extracted/gmr_markers \
        --measurement_unit m

    # Batch process a whole directory
    python gmr_extract_markers.py \
        --motion_file /home/mocap/data/GMR/pkl \
        --out_dir extracted/gmr_markers
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import mujoco as mj

# Vicon Plug-in Gait Full Body marker set (39 markers) -- must match the
# <site> names added to HU_D04_01.xml.
MARKER_NAMES = [
    "LFHD", "RFHD", "LBHD", "RBHD",
    "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI",
    "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
]

# whole_body_tracking's deployment copy of the robot xml -- the one with
# the marker <site> elements (GMR_Repo's own oli.xml copy does not have
# them; see module docstring for why using this file is still correct).
DEFAULT_ROBOT_XML = (
    "/home/mocap/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/HU_D04_description/xml/HU_D04_01.xml"
)

# GMR's native unit is meters (the mujoco scene/robot xml are both built in
# meters); this is the conversion factor applied before export
UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


def resolve_marker_site_ids(model, marker_names):
    site_ids = []
    missing = []
    for name in marker_names:
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            missing.append(name)
        site_ids.append(sid)
    if missing:
        raise ValueError(f"These marker sites are not found in the robot xml: {missing}")
    return site_ids


def extract_markers(motion_file, robot_xml, marker_names, unit="m"):
    with open(motion_file, "rb") as f:
        motion = pickle.load(f)

    fps = float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)       # (T, 3)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)  # (T, 4), GMR pkl stores xyzw
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]                    # mujoco qpos wants wxyz
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)         # (T, ndof)

    model = mj.MjModel.from_xml_path(str(robot_xml))
    data = mj.MjData(model)

    site_ids = resolve_marker_site_ids(model, marker_names)

    num_frames = root_pos.shape[0]
    scale = UNIT_SCALE[unit]

    positions = np.zeros((num_frames, len(marker_names), 3), dtype=np.float32)

    for t in range(num_frames):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz[t]
        data.qpos[7:] = dof_pos[t]
        mj.mj_forward(model, data)

        positions[t] = data.site_xpos[site_ids] * scale

    return {
        "marker_names": np.array(marker_names),
        "positions": positions,
        "fps": np.float32(fps),
        "src_fps": np.float32(fps),
        "unit": unit,
        "source_file": str(motion_file),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True,
                         help="A single GMR output .pkl file, or a directory containing multiple such files")
    parser.add_argument("--robot_xml", default=DEFAULT_ROBOT_XML,
                         help="Robot xml to run FK through (must contain the 39 marker <site> elements)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--markers", default=None,
                         help="Comma-separated marker subset; if omitted, uses all 39 Plug-in Gait markers")
    parser.add_argument("--measurement_unit", choices=["m", "cm", "mm"], default="m",
                         help="Length unit for output coordinates; defaults to m (mujoco's native unit) if omitted")
    args = parser.parse_args()

    marker_names = args.markers.split(",") if args.markers else MARKER_NAMES

    robot_xml = Path(args.robot_xml)
    if not robot_xml.exists():
        raise FileNotFoundError(f"Robot xml not found: {robot_xml}")

    src = Path(args.motion_file)
    files = sorted(src.glob("*.pkl")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"Did not find .pkl files in {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"[info] Extracting {f.name} ...")
        result = extract_markers(f, robot_xml, marker_names, unit=args.measurement_unit)
        out_path = out_dir / f"{f.stem}_markers_{args.measurement_unit}.npz"
        np.savez(out_path, **result)
        print(f"  -> {out_path}  ({result['positions'].shape[0]} frames, "
              f"{len(marker_names)} markers, fps={result['fps']:.2f}, unit={args.measurement_unit})")

    print(f"Finished, Processed {len(files)} files in total. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
