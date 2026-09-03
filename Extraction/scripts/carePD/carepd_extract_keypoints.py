"""
carepd_extract_keypoints.py

[Part A] Extract world-space trajectories + orientations of key body joints
from CARE-PD raw SMPL-X mocap data, and save them as a .npz file that is
fully decoupled from the GMR / IsaacLab / smplx environments, for Part B
(compute metrics) and Part C (compare across checkpoints) to read directly.

Why this step is split out on its own (instead of extract+compute-metric
being combined like carepd_high_level_metrics.py):
  - Part B's algorithm needs to consume both the CARE-PD extraction output
    and the GMR output extraction with the same code path. If it had to
    re-import smplx / general_motion_retargeting every time just to get
    trajectories, Part B/C would be forced to depend on those two libraries'
    environment (the gmr conda env).
  - Once split into a standalone .npz, Part B/C only need numpy to run, no
    need to install smplx/GMR.

Keypoints extracted: the 22 body joints of the SMPL-X body model (excluding
fingers/jaw/eyeballs, which aren't useful for gait/PD analysis), covering
the lower body (gait) + torso (forward-leaning posture) + upper body
(reduced arm swing typical of PD patients):
    pelvis, left_hip, right_hip, spine1, spine2, spine3,
    left_knee, right_knee, left_ankle, right_ankle, left_foot, right_foot,
    neck, left_collar, right_collar, head,
    left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
Plus 6 foot landmark points (not on the kinematic chain, borrowing the
orientation of their parent foot joint):
    left_heel, left_big_toe, left_small_toe, right_heel, right_big_toe, right_small_toe
Zeni heel-strike/toe-off detection is in theory defined using the actual
heel/toe, which is closer to the original definition than the generic
"foot" joint used previously.

There's also a sacrum: SMPL-X has no such landmark, and no mesh vertex
mapping corresponding to LPSI/RPSI either, so the pelvis root joint is used
as a stand-in here (see the comment in smpl.py for details).
This isn't exactly the same as the "LPSI/RPSI midpoint" definition used on
the Vicon side -- it's an internal centroid, not a skin-surface point, so
it will sit somewhat more anterior than the true position -- but it's close
in height, and is the closest approximation currently available.

Use --keypoints to override with a subset.

Output .npz fields:
    joint_names   : (J,) str          names of the extracted keypoints, in
                                       the same order as positions/orientations
    positions     : (T, J, 3) float32 world-frame positions, unit per the
                                       unit field
    orientations  : (T, J, 4) float32 world-frame orientations, quaternion
                                       (w, x, y, z)
    fps           : float             frame rate after resampling
    src_fps       : float             original mocap frame rate
    human_height  : float             height estimated from betas, unit per
                                       the unit field
    unit          : str               length unit for positions/human_height,
                                       m/cm/mm
    source_file   : str               path to the input file

SMPL-X's native unit is meters; --measurement_unit only converts
positions and human_height before export (orientation is a quaternion,
which has no unit and isn't affected). If not passed, it defaults to m,
identical to previous output.

Usage (needs to run in an environment that can import smplx, e.g.
conda activate gmr):
    # Single file
    python carepd_extract_keypoints.py \
        --smplx_file /home/mocap/data/CAREPD/smpl/BMCLab_SUB05_off_walk_1_canonical.npz.zip \
        --smplx_body_model_path /home/mocap/data \
        --out_dir extracted/carepd \
        --tgt_fps 150 \
        --measurement_unit mm

    # Batch process a whole directory
    python carepd_extract_keypoints.py \
        --smplx_file /home/mocap/data/CAREPD/smpl \
        --smplx_body_model_path /home/mocap/data \
        --out_dir extracted/carepd \
        --tgt_fps 100 \
        --measurement_unit m
"""

import argparse
from pathlib import Path

import numpy as np

from smpl import load_smplx_file, get_smplx_data_offline_fast

# The 22 body joints of the SMPL-X body model (excluding fingers/jaw/eyeballs,
# which aren't useful for gait/PD analysis)
DEFAULT_KEYPOINTS = [
    "pelvis", "left_hip", "right_hip",
    "spine1", "spine2", "spine3",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_heel", "left_big_toe", "left_small_toe",
    "right_heel", "right_big_toe", "right_small_toe",
    "sacrum",
]

# SMPL-X's native unit is meters; this is the conversion factor applied before export
UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


def extract_keypoints(smplx_file, smplx_body_model_path, tgt_fps, keypoints, unit="m"):
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        smplx_file, smplx_body_model_path
    )
    src_fps = float(smplx_data["mocap_frame_rate"].item())
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    available = frames[0].keys()
    missing = [k for k in keypoints if k not in available]
    if missing:
        raise ValueError(f"These keypoints are not in the SMPL-X output: {missing}")

    scale = UNIT_SCALE[unit]

    positions = np.stack(
        [[frame[name][0] for name in keypoints] for frame in frames], axis=0
    ).astype(np.float32) * scale  # (T, J, 3)
    orientations = np.stack(
        [[frame[name][1] for name in keypoints] for frame in frames], axis=0
    ).astype(np.float32)  # (T, J, 4), wxyz -- quaternions are unitless, not affected by scale

    return {
        "joint_names": np.array(keypoints),
        "positions": positions,
        "orientations": orientations,
        "fps": np.float32(fps),
        "src_fps": np.float32(src_fps),
        "human_height": np.float32(human_height * scale),
        "unit": unit,
        "source_file": str(smplx_file),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True,
                         help="A single CARE-PD .npz/.npz.zip file, or a directory containing multiple such files")
    parser.add_argument("--smplx_body_model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tgt_fps", type=int, default=30)
    parser.add_argument("--keypoints", default=None,
                         help="Comma-separated keypoint subset; if omitted, uses the default 22 body joints")
    parser.add_argument("--measurement_unit", choices=["m", "cm", "mm"], default="m",
                         help="Length unit for output coordinates; defaults to m (SMPL-X's native unit) if omitted")
    args = parser.parse_args()

    keypoints = args.keypoints.split(",") if args.keypoints else DEFAULT_KEYPOINTS

    src = Path(args.smplx_file)
    if src.is_dir():
        files = sorted(list(src.glob("*.npz.zip")) + list(src.glob("*.npz")))
    else:
        files = [src]

    if not files:
        raise FileNotFoundError(f"Did not find .npz / .npz.zip files in {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"[info] Extracting {f.name} ...")
        result = extract_keypoints(f, args.smplx_body_model_path, args.tgt_fps, keypoints,
                                    unit=args.measurement_unit)
        out_path = out_dir / f"{f.stem.replace('.npz', '')}_keypoints_{args.measurement_unit}.npz"
        np.savez(out_path, **result)
        print(f"  -> {out_path}  ({result['positions'].shape[0]} frames, "
              f"{len(keypoints)} keypoints, fps={result['fps']:.2f}, unit={args.measurement_unit})")

    print(f"Finished, Processed {len(files)} files in total. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
