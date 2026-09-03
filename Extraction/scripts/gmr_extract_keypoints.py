"""
gmr_extract_keypoints.py

[Part A / GMR side] Extract world-space trajectories + orientations of key
bodies from GMR retargeting output that's already been run and saved to
disk (.pkl: root_pos / root_rot / dof_pos / fps, e.g. the files in
/home/mocap/whole_body_tracking/motions/*.pkl that get fed into IsaacLab
training). Saves them in exactly the same .npz format as
carepd_extract_keypoints.py, so Part B / Part C can read both with the
same code.

Relationship to carepd_extract_keypoints.py
----------------------------------
The output fields, shapes, and unit conventions of both scripts are
identical (joint_names / positions / orientations / fps / unit /
source_file), so when Part C compares "CARE-PD raw vs GMR output"
fidelity, it can just subtract same-named keypoint arrays directly,
regardless of which side the data came from.

Why read the saved .pkl directly, instead of re-running retargeting
----------------------------------
gmr_high_level_metrics.py currently re-runs GeneralMotionRetargeting every
time (needs the smplx/torch/GMR IK environment). But what we actually want
to measure fidelity against is the motion file that's already been
produced and will actually be used for IsaacLab training/replay -- using
that file matches Part C's definition of "compare motion files across
checkpoints" more directly, and it lets Part A drop the smplx/torch
dependency entirely, needing only mujoco for forward kinematics (FK):
    qpos = [root_pos(3), root_rot_wxyz(4), dof_pos(N)] -> mj_forward
        -> read data.xpos / data.xquat
This qpos layout is exactly the same as what
general_motion_retargeting.RobotMotionViewer.step() / load_robot_motion()
use (they also convert root_rot from xyzw to wxyz before putting it into
qpos[3:7]). It's reimplemented independently here so this script doesn't
need to import general_motion_retargeting -- its __init__ pulls in
motion_retarget.py along with it (the smplx/torch/IK heavy dependencies),
so this extraction script can also run on machines/environments that
don't have those libraries installed.

How keypoints are chosen and mapped to robot bodies
----------------------------------
Rather than picking robot bodies arbitrarily, this reuses the "human
keypoint -> robot body" correspondence the GMR team already defined in the
ik_config (ik_match_table1) -- that mapping is exactly what retargeting IK
aimed for when aligning the two poses, so the robot trajectory read out
via this mapping measures precisely "did IK actually get this body where
it was supposed to go", which is the error definition most faithful to
retargeting's own semantics.

Keypoints not covered by ik_match_table1:
  - sacrum: same as on the CARE-PD side, there's no dedicated landmark, so
    base_link (=pelvis) is used as a stand-in.
  - left/right_heel, left/right_toe: retargeting doesn't treat these as IK
    targets (the foot's ground-contact points are driven indirectly via
    base_link/foot, not controlled individually) -- they're standalone
    foot contact bodies in the robot's urdf/mjcf, read directly by body
    name here. Note: Oli only has a single toe contact point per foot,
    unlike SMPL-X's big_toe/small_toe distinction, so these are named
    left_toe/right_toe -- they don't correspond 1:1 with
    left_big_toe/left_small_toe on the CARE-PD side. If Part C needs to
    align them, take the midpoint of big_toe/small_toe on the CARE-PD side.
  - spine1/spine2/neck/left_collar/right_collar/head: ik_match_table1 has
    no corresponding IK target (Oli only has a single waist_pitch_link
    mapped to spine3, with no independent neck/head joint retargeting
    could aim at) -- the robot structurally cannot express these
    keypoints, so they're excluded from the default keypoint list. This
    itself is fidelity information worth reporting in Part C: a
    "structural absence", not "an error that numerically happens to be
    zero".

Output .npz fields (aligned with carepd_extract_keypoints.py):
    joint_names       : (J,) str          names of the extracted keypoints
    positions         : (T, J, 3) float32 world-frame positions, unit per
                                           the unit field
    orientations      : (T, J, 4) float32 world-frame orientations,
                                           quaternion (w, x, y, z)
    fps               : float             fps as stored in the motion file
    src_fps           : float             same as fps -- the GMR output
                                           .pkl doesn't separately retain a
                                           "pre-resampling" frame rate
    robot_body_names  : (J,) str          the actual robot body name each
                                           keypoint was read from, to make
                                           it easy to check the mapping
    unit              : str               length unit for positions, m/cm/mm
    source_file       : str               path to the input .pkl

Low-level data: dof_pos (robot joint angles, in radians) is already
present in the source .pkl, and is passed through as-is into the output
.npz (field dof_pos) -- no need for a separate script to export it.

How many keypoints (J) does the output have?
J is fixed at 19 by DEFAULT_KEYPOINTS below (14 IK_TARGET_KEYPOINTS + 4
EXTRA_CONTACT_KEYPOINTS + sacrum) -- it does NOT depend on which .pkl you
point this at. The input motion file only supplies the trajectory
(root_pos/root_rot/dof_pos) that gets driven through FK; it has no say in
which keypoints are read out. J only changes if you pass --keypoints with
a different subset, or if a keypoint in that list has no entry in the
given --ik_config's ik_match_table1 (raises ValueError, not a silently
smaller J).

Usage (only needs mujoco + numpy, no smplx/torch/GMR environment required):
    # Single file, using the CARE-PD-specific IK config (ground_height is
    # tuned for CARE-PD data)
    python gmr_extract_keypoints.py \
        --motion_file /home/mocap/whole_body_tracking/motions/BMCLab_SUB01_off_walk_1_canonical.pkl \
        --robot oli_carepd \
        --out_dir extracted/gmr \
        --measurement_unit m

    # Batch process a whole directory
    python gmr_extract_keypoints.py \
        --motion_file /home/mocap/data/GMR/pkl \
        --robot oli_carepd \
        --out_dir extracted/gmr
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import mujoco as mj

# Keypoints that go through retargeting's own human-robot correspondence
# (ik_match_table1), ordered to line up with carepd_extract_keypoints.py's
# DEFAULT_KEYPOINTS as closely as possible for easy comparison.
IK_TARGET_KEYPOINTS = [
    "pelvis", "left_hip", "right_hip", "spine3",
    "left_knee", "right_knee", "left_foot", "right_foot",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

# Foot ground-contact points not in ik_match_table1, read directly by robot
# body name (see docstring above).
EXTRA_CONTACT_KEYPOINTS = {
    "left_heel": "contact_left_heel",
    "left_toe": "contact_left_tip",
    "right_heel": "contact_right_heel",
    "right_toe": "contact_right_tip",
}

# No dedicated landmark, so the pelvis body is used as a stand-in (same
# treatment as on the CARE-PD side).
SACRUM_ROBOT_BODY_ALIAS = "base_link"

DEFAULT_KEYPOINTS = (
    IK_TARGET_KEYPOINTS
    + list(EXTRA_CONTACT_KEYPOINTS.keys())
    + ["sacrum"]
)

# Asset paths relative to --gmr_repo, equivalent to what
# general_motion_retargeting/params.py's ROBOT_XML_DICT /
# IK_CONFIG_DICT["smplx"] define for oli. Copied here as plain path
# strings instead of importing that package (to avoid pulling in the
# smplx/torch dependencies -- see the note at the top of this file).
ROBOT_XML_REL = {
    "oli": "assets/oli/oli.xml",
    "oli_carepd": "assets/oli/oli.xml",
}
ROBOT_IK_CONFIG_REL = {
    "oli": "general_motion_retargeting/ik_configs/smplx_to_oli.json",
    "oli_carepd": "general_motion_retargeting/ik_configs/smplx_to_oli_carepd.json",
}

# GMR's native unit is meters (the mujoco scene/robot xml are both built in
# meters); this is the conversion factor applied before export
UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


def load_ik_target_mapping(ik_config_path, keypoints):
    """Reverse-lookup human_keypoint -> robot_body_name from the ik_config's ik_match_table1."""
    with open(ik_config_path) as f:
        cfg = json.load(f)
    # ik_match_table1: {robot_body_name: [human_body_name, pos_weight, rot_weight, pos_offset, rot_offset]}
    human_to_robot = {entry[0]: robot_body for robot_body, entry in cfg["ik_match_table1"].items()}

    mapping = {}
    missing = []
    for kp in keypoints:
        if kp == "sacrum":
            mapping[kp] = SACRUM_ROBOT_BODY_ALIAS
        elif kp in EXTRA_CONTACT_KEYPOINTS:
            mapping[kp] = EXTRA_CONTACT_KEYPOINTS[kp]
        elif kp in human_to_robot:
            mapping[kp] = human_to_robot[kp]
        else:
            missing.append(kp)
    if missing:
        raise ValueError(
            f"These keypoints have no robot body mapping in {ik_config_path}: {missing}"
        )
    return mapping


def extract_keypoints(motion_file, robot_xml, ik_config, keypoints, unit="m"):
    with open(motion_file, "rb") as f:
        motion = pickle.load(f)

    fps = float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)       # (T, 3)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)  # (T, 4), GMR pkl stores xyzw
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]                    # mujoco qpos wants wxyz
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)         # (T, ndof)

    mapping = load_ik_target_mapping(ik_config, keypoints)

    model = mj.MjModel.from_xml_path(str(robot_xml))
    data = mj.MjData(model)

    body_ids = {}
    missing_bodies = []
    for kp, body_name in mapping.items():
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            missing_bodies.append(body_name)
        body_ids[kp] = bid
    if missing_bodies:
        raise ValueError(f"These robot bodies are not found in {robot_xml}: {missing_bodies}")

    num_frames = root_pos.shape[0]
    scale = UNIT_SCALE[unit]

    positions = np.zeros((num_frames, len(keypoints), 3), dtype=np.float32)
    orientations = np.zeros((num_frames, len(keypoints), 4), dtype=np.float32)

    for t in range(num_frames):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz[t]
        data.qpos[7:] = dof_pos[t]
        mj.mj_forward(model, data)

        for j, kp in enumerate(keypoints):
            bid = body_ids[kp]
            positions[t, j] = data.xpos[bid] * scale
            orientations[t, j] = data.xquat[bid]  # mujoco xquat is already wxyz

    return {
        "joint_names": np.array(keypoints),
        "positions": positions,
        "orientations": orientations,
        "fps": np.float32(fps),
        "src_fps": np.float32(fps),
        "robot_body_names": np.array([mapping[kp] for kp in keypoints]),
        "dof_pos": dof_pos.astype(np.float32),
        "unit": unit,
        "source_file": str(motion_file),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True,
                         help="A single GMR output .pkl file, or a directory containing multiple such files")
    parser.add_argument("--robot", default="oli_carepd", choices=list(ROBOT_XML_REL.keys()),
                         help="Determines which robot xml + which ik_config is used for FK and keypoint mapping")
    parser.add_argument("--gmr_repo", default="/home/mocap/whole_body_tracking/GMR_Repo/GMR",
                         help="Path to the (teammate's customized) GMR repo that has the oli robot config")
    parser.add_argument("--robot_xml", default=None, help="Override the default robot xml path")
    parser.add_argument("--ik_config", default=None, help="Override the default ik_config json path")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--keypoints", default=None,
                         help="Comma-separated keypoint subset; if omitted, uses the default IK targets + foot contact points + sacrum")
    parser.add_argument("--measurement_unit", choices=["m", "cm", "mm"], default="m",
                         help="Length unit for output coordinates; defaults to m (mujoco/GMR's native unit) if omitted")
    args = parser.parse_args()

    keypoints = args.keypoints.split(",") if args.keypoints else DEFAULT_KEYPOINTS

    gmr_repo = Path(args.gmr_repo)
    robot_xml = Path(args.robot_xml) if args.robot_xml else gmr_repo / ROBOT_XML_REL[args.robot]
    ik_config = Path(args.ik_config) if args.ik_config else gmr_repo / ROBOT_IK_CONFIG_REL[args.robot]
    if not robot_xml.exists():
        raise FileNotFoundError(f"Robot xml not found: {robot_xml}")
    if not ik_config.exists():
        raise FileNotFoundError(f"ik_config not found: {ik_config}")

    src = Path(args.motion_file)
    files = sorted(src.glob("*.pkl")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"Did not find .pkl files in {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"[info] Extracting {f.name} ...")
        result = extract_keypoints(f, robot_xml, ik_config, keypoints, unit=args.measurement_unit)
        out_path = out_dir / f"{f.stem}_keypoints_{args.measurement_unit}.npz"
        np.savez(out_path, **result)
        print(f"  -> {out_path}  ({result['positions'].shape[0]} frames, "
              f"{len(keypoints)} keypoints, fps={result['fps']:.2f}, unit={args.measurement_unit})")

    print(f"Finished, Processed {len(files)} files in total. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
