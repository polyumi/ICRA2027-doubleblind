#!/usr/bin/env bash
# Collect the whole water-level dataset: 30 trials at each of three levels, pausing to refill.
#
#   ./collect_water_dataset.sh              # 30 trials x empty, some, half, interleaved
#   BLOCK=10 ./collect_water_dataset.sh 30 plastic black_metal silver_metal
#   BLOCK=30 ./collect_water_dataset.sh     # one block per class (NOT recommended -- see below)
#   ./collect_water_dataset.sh 30 half some empty
#   TRIALS=5 ./collect_water_dataset.sh     # short rehearsal before committing to the full set
#   ./collect_water_dataset.sh 25 plastic black_metal silver_metal   # any class labels
#
# DURATION_S overrides the derived recording window. Set it whenever the shake is CLAMPED by the
# controller -- a request over max_pos_speed is stretched, not truncated, so the motion runs
# longer than n_shakes*period_s and the derived window would cut it off mid-trial.
#   AMPLITUDE_M=0.12 PERIOD_S=0.5 ./collect_water_dataset.sh   # a faster, larger shake
#
# SHAKE GEOMETRY is set by AMPLITUDE_M (default 0.08 m), PERIOD_S (default 1.0 s) and N_SHAKES
# (default 5), and passed through to every trial. Change them for a whole dataset, never between
# levels: the classifier assumes water level is the only thing that differs, so a dataset shaken
# two ways has a second, unlabelled variable in it. The recording window is derived from them, so
# a slower shake is not silently truncated.
#
# Speed scales as AMPLITUDE_M/PERIOD_S and acceleration as AMPLITUDE_M/PERIOD_S^2, so shortening
# the period bites much harder than raising the amplitude. At the defaults the peak is 0.25 m/s
# and 1.6 m/s^2, about a quarter of the controller's own max_pos_speed and ~5% of its 20 N force
# ceiling against a 0.7 kg payload -- there is a lot of headroom for a more vigorous shake.
#
# INTERLEAVED IN BLOCKS. The classes are NOT collected one after another. The run is split into
# rounds: BLOCK trials of each class in turn, then round two, and so on, stopping for a swap each
# time -- 3 classes x 30 trials at BLOCK=10 is nine blocks and nine swaps.
#
# This is the one structural defence against a nuisance variable that drifts over a session. The
# finger camera settles at a slightly different level and colour balance each time the rig is
# disturbed -- about 1-2 grey levels out of 255, invisible, but stable to a fifth of a grey level
# within a block. Collected one class per block, that offset IS the label: a classifier scored 93%
# on it, and separated the first fifteen trials of a single class from its own last fifteen at
# 100%. Spread each class across three separated blocks and no such offset can line up with the
# class any more. Randomising the start position masks the nuisance; only interleaving removes it.
#
# It also means the validation split (the last third of each class) comes from a different block
# than the training trials, which is the honest generalisation test rather than a formality.
#
# It STOPS between blocks so the object can be swapped -- it will not roll on unattended, because
# that would silently label a block with the previous object.
#
# RESUMING. Trial numbers are per level and the recorder refuses to overwrite, so a run that dies
# partway is resumed by starting the level again with a higher first trial:
#     ./collect_water_trials.sh some 18 13
#
# The last 5 trials of each level become the validation split (water_dataset.py --val-from 26),
# so collect them in the same session and the same way as the first 25 -- they are the held-out
# test of whether the classifier generalises, and a change of setup between 25 and 26 would show
# up as a generalisation failure that is really a procedure change.
set -euo pipefail

TRIALS="${TRIALS:-${1:-30}}"
shift || true
# Trials per block before swapping to the next class. The default gives three rounds over a
# 30-trial set. Smaller blocks interleave harder at the cost of more swaps.
BLOCK="${BLOCK:-10}"
AMPLITUDE_M="${AMPLITUDE_M:-0.08}"
PERIOD_S="${PERIOD_S:-1.0}"
N_SHAKES="${N_SHAKES:-5}"
WAYPOINT_DT="${WAYPOINT_DT:-0.05}"
SHAKE_AXIS="${SHAKE_AXIS:-[0.0, 0.0, 1.0]}"
SYMMETRIC="${SYMMETRIC:-false}"
DURATION_S="${DURATION_S:-}"
export AMPLITUDE_M PERIOD_S N_SHAKES
LEVELS=("$@")
# half -> some -> empty: you can always POUR WATER OUT between levels, but refilling to an
# exact level mid-run is fiddly and drifts. Starting full and emptying keeps the bottle, the
# grip and the setup identical down the sequence, so the only thing changing is the water.
[ ${#LEVELS[@]} -eq 0 ] && LEVELS=(half some empty)
export WAYPOINT_DT SHAKE_AXIS SYMMETRIC
# One grip width for the whole dataset, every object, every run. collect_water_trials.sh reads it
# from ${ROOT}/grip_width_m when it is not set here, and the first uncalibrated trial writes that
# file -- so exporting nothing is the normal case and still yields one shared width. Set
# GRIP_WIDTH_M to reuse a width measured on some earlier day, or GRIP_SQUEEZE_M to change how hard
# the calibrating trial squeezes.
export GRIP_WIDTH_M="${GRIP_WIDTH_M:-}"
export GRIP_SQUEEZE_M="${GRIP_SQUEEZE_M:-0.002}"
# Randomised start, shared the same way: the first trial records the reference position into
# ${ROOT}/home_xyz and every class collected afterwards jitters about that same point.
export HOME_XYZ="${HOME_XYZ:-}"
export JITTER_XYZ_M="${JITTER_XYZ_M:-[0.03, 0.03, 0.02]}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PER_LEVEL="${HERE}/collect_water_trials.sh"
ROOT="${EVAL_ROOT:-/data/eval_results_9_6}/water"

ROUNDS=$(( (TRIALS + BLOCK - 1) / BLOCK ))
echo "=============================================================="
echo " water dataset: ${TRIALS} trials x ${#LEVELS[@]} classes (${LEVELS[*]})"
echo " interleaved: ${ROUNDS} round(s) of ${BLOCK} trials each, $(( ROUNDS * ${#LEVELS[@]} )) blocks, $(( ROUNDS * ${#LEVELS[@]} )) swaps"
echo " shake: ${N_SHAKES} x ${PERIOD_S}s at ${AMPLITUDE_M}m  (identical for every trial)"
echo " start: jittered +/- ${JITTER_XYZ_M} m about one recorded reference"
echo " bags -> ${ROOT}/<class>/trial_<n>"
echo "=============================================================="
echo
echo "Before starting, confirm ALL of these -- a wrong one costs the whole level:"
echo "  * the NUC is up with execute_arm:=true"
echo "  * the Pi is streaming (tactile + audio) and the GoPro is up"
echo "  * the bottle is gripped, and the arm is somewhere roomy"
echo "  * you have dry-run the motion once:  ros2 run polyumi_ros2 water_shake"
echo
read -r -p "Ready? [Enter to begin, Ctrl-C to abort] " _

BLOCK_NO=0
for ((r = 1; r <= ROUNDS; r++)); do
    for i in "${!LEVELS[@]}"; do
        LEVEL="${LEVELS[$i]}"
        BLOCK_NO=$((BLOCK_NO + 1))

        # Where this class got to, read off disk rather than counted in a variable, so an
        # interrupted run resumes correctly simply by being re-run.
        START=1
        if compgen -G "${ROOT}/${LEVEL}/trial_*" > /dev/null; then
            LAST=$(ls -d "${ROOT}/${LEVEL}"/trial_* | sed 's/.*trial_//' | sort -n | tail -1)
            START=$((LAST + 1))
        fi
        # This round's target for this class, capped at TRIALS so the last round takes the
        # remainder when TRIALS is not a multiple of BLOCK.
        TARGET=$((r * BLOCK))
        [ "${TARGET}" -gt "${TRIALS}" ] && TARGET="${TRIALS}"
        COUNT=$((TARGET - START + 1))

        echo
        echo "=============================================================="
        echo " ROUND ${r}/${ROUNDS}  ---  BLOCK ${BLOCK_NO}/$((ROUNDS * ${#LEVELS[@]}))  ---  ${LEVEL}"
        echo "=============================================================="
        if [ "${COUNT}" -le 0 ]; then
            echo "  ${LEVEL} already has ${TARGET} trials -- nothing to do this round."
            continue
        fi
        echo
        echo "  >>> SWAP TO: ${LEVEL}"
        echo "      Trials ${START}-${TARGET} (${COUNT} of them) go in this block."
        echo "      Change ONLY the object. Leave the arm, the lighting and the camera alone --"
        echo "      anything else you change becomes a second unlabelled variable."
        echo
        read -r -p "  Press ENTER when '${LEVEL}' is in the gripper... " _

        "${PER_LEVEL}" "${LEVEL}" "${START}" "${COUNT}" ${DURATION_S}
        echo "  ${LEVEL}: $(ls -d "${ROOT}/${LEVEL}"/trial_* 2>/dev/null | wc -l) trial(s) on disk"
    done
done

echo
echo "=============================================================="
echo " done. per-level counts:"
for LEVEL in "${LEVELS[@]}"; do
    printf '   %-8s %s\n' "${LEVEL}" "$(ls -d "${ROOT}/${LEVEL}"/trial_* 2>/dev/null | wc -l)"
done
echo
echo " next, offline:"
echo "   python3 analysis/water_dataset.py --root ${ROOT} --out water_tensors.npz \\"
echo "       --levels ${LEVELS[*]} --val-from $(( TRIALS - BLOCK + 1 ))"
echo "   python3 analysis/train_water_cnn.py --tensors water_tensors.npz"
echo
echo " the validation split is the LAST round, so it is a different block from the training"
echo " trials -- a real generalisation test rather than a reshuffle of the same recording."
echo "=============================================================="
