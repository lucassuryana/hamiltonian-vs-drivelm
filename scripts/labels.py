"""
scripts/labels.py
Single source of truth for DriveLM categorical labelling.

Speed thresholds: empirical, derived from DriveLM ∩ nuScenes v1.0-trainval
                  (train split only -- val excluded, its answer field is
                  withheld by the benchmark) via
                  scripts/04b_drivelm_label_stats_full.py.
                  n ≈ 3,857 labelled records (vs. n=38 in the original
                  mini-based version).

Direction thresholds: empirical for the right-side boundaries; the left-side
                  boundaries are NOT independently derivable because DriveLM's
                  behavior QA never uses the "slightly steering to the left"
                  label at all (0 samples, confirmed at full scale, not a
                  data-loading issue). Left-side boundaries are therefore set
                  by ASSUMING LEFT/RIGHT SYMMETRY and mirroring the empirical
                  right-side values. This is a stated assumption, not an
                  independent empirical result for the left side -- flag this
                  explicitly in the methods section if used for the paper.

Full-data source stats: config/drivelm_label_stats_full.json
Original mini-based stats (for comparison): config/drivelm_label_stats.json
"""
import math

# ── Speed thresholds on avg_speed_mps (m/s) ──────────────────────────
# Empirical class boundaries from DriveLM ∩ nuScenes trainval, n≈3,857.
# Each value is the boundary BELOW which the class applies.
#
# Previous (mini, n=38):  [1.21, 3.78, 6.23, 9.78]
SPEED_THRESHOLDS_MPS = [1.073, 4.196, 6.152, 8.834]
SPEED_LABELS = [
    'not moving',
    'driving slowly',
    'driving with normal speed',
    'driving fast',
    'driving very fast',
]

# ── Direction thresholds on angle_overall_deg ────────────────────────
# Empirical (right side only, full data):
#   going straight / slightly-right boundary:  -2.031 deg
#   slightly-right / full-right boundary:      -8.495 deg
# Left side has zero samples for "slightly steering to the left" in DriveLM's
# behavior QA (confirmed at full scale) -- no empirical boundary derivable.
# Mirrored from the right side under an assumed left/right symmetry:
#
# Previous (mini): DIR_SLIGHT_DEG ~= 3.81, DIR_FULL_DEG ~= 14.93
#                   (geometric approximation from DriveLM's published y3
#                   thresholds, not empirical -- mini had only 2/5 direction
#                   classes populated)
DIR_SLIGHT_DEG = 2.031   # empirical (right side), mirrored for left
DIR_FULL_DEG   = 8.495   # empirical (right side), mirrored for left

DIRECTION_LABELS = [
    'steering to the left',
    'slightly steering to the left',
    'going straight',
    'slightly steering to the right',
    'steering to the right',
]


def continuous_to_drivelm_label(speed_mps, angle_deg, horizon_s=1.5):
    """
    Map continuous physics fields to DriveLM categorical labels.
    Applies thresholds directly to speed and angle — no x3/y3 reconstruction.

    Returns (direction, speed) as DriveLM-vocabulary strings.
    """
    # Direction: thresholded directly on angle_overall_deg
    if   angle_deg >  DIR_FULL_DEG:   direction = 'steering to the left'
    elif angle_deg >  DIR_SLIGHT_DEG: direction = 'slightly steering to the left'
    elif angle_deg < -DIR_FULL_DEG:   direction = 'steering to the right'
    elif angle_deg < -DIR_SLIGHT_DEG: direction = 'slightly steering to the right'
    else:                             direction = 'going straight'

    # Speed: thresholded directly on avg_speed_mps
    if   speed_mps < SPEED_THRESHOLDS_MPS[0]: speed = 'not moving'
    elif speed_mps < SPEED_THRESHOLDS_MPS[1]: speed = 'driving slowly'
    elif speed_mps < SPEED_THRESHOLDS_MPS[2]: speed = 'driving with normal speed'
    elif speed_mps < SPEED_THRESHOLDS_MPS[3]: speed = 'driving fast'
    else:                                    speed = 'driving very fast'

    return direction, speed


def format_drivelm_label(direction, speed):
    """Combine direction + speed into the canonical DriveLM sentence."""
    return f"The ego vehicle is {direction}. The ego vehicle is {speed}."
