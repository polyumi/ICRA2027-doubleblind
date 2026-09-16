#!/usr/bin/env bash
# Collect labelled water-bottle shake trials: record a bag while the arm does one fixed shake.
#
#   ./collect_water_trials.sh <level> <first_trial> [count] [duration_s]
#   ./collect_water_trials.sh empty 1 20        # 20 trials, empty bottle
#   ./collect_water_trials.sh half  1 20
#
# WHY A WRAPPER. The classifier's only premise is that the water level is the sole thing that
# differs between trials, so the recording window and the motion must line up the same way every
# time. Doing that by hand across 60 trials is how a label gets attached to a bag that caught half
# a shake. Here the recorder is started first, given time to subscribe, and stopped by its own
# duration; the shake is fired once in between.
#
# THE LABEL IS THE PATH. Bags land at $EVAL_ROOT/water/<level>/trial_<n>, reusing
# record_eval_trial.sh's <policy> slot as the class label -- so the training set is just a
# directory listing, with no separate manifest to drift out of sync.
#
# BETWEEN LEVELS, refill the bottle and start a new level; trial numbers are per level, so each
# run of this script can start again at 1. The shake returns the arm to where it started, so
# trials within a level need no reset.
set -euo pipefail

LEVEL="${1:?usage: collect_water_trials.sh <class-label> <first_trial> [count] [duration_s]}"
FIRST="${2:?missing first trial number}"
COUNT="${3:-1}"

# Shake geometry. These MUST be identical across every trial in a dataset -- the classifier's
# whole premise is that water level is the only thing that differs -- so they are env vars set
# once for a whole collection run rather than per-trial arguments, and they are echoed below so
# the value used is visible in the session log next to the bags it produced.
AMPLITUDE_M="${AMPLITUDE_M:-0.08}"
PERIOD_S="${PERIOD_S:-1.0}"
N_SHAKES="${N_SHAKES:-5}"
WAYPOINT_DT="${WAYPOINT_DT:-0.05}"
SHAKE_AXIS="${SHAKE_AXIS:-[0.0, 0.0, 1.0]}"
SYMMETRIC="${SYMMETRIC:-false}"

# ONE jaw width, in metres, for every object and every trial in the whole dataset.
#
# It is calibrated once -- the first trial that runs with no width on record measures where the
# jaws rest, takes GRIP_SQUEEZE_M off that, and writes the result to GRIP_FILE -- and every trial
# after that, of every class, is handed the same absolute number.
#
# Two separate mistakes this rules out. Re-deriving the width from the current jaw position each
# trial ratchets, because the jaws stay closed between trials and so the next measurement reads
# what the last trial squeezed to (76.9 mm to 73.7 mm over eight trials, observed). And
# calibrating per object gives each class its own width, which makes the grip a perfect predictor
# of the label -- exactly the confound a per-class calibration looks like it is avoiding. Only a
# width shared across all classes takes the gripper out of the experiment.
#
# It follows that the objects must be close enough in size for one width to hold them all. If they
# are not, the grip cannot be held constant and the confound has to be handled another way --
# interleaving the classes so it is at least not aligned with collection order.
BAG_ROOT="${EVAL_ROOT:-/data/eval_results_9_6}/water"
GRIP_FILE="${GRIP_FILE:-${BAG_ROOT}/grip_width_m}"
GRIP_SQUEEZE_M="${GRIP_SQUEEZE_M:-0.002}"
GRIP_WIDTH_M="${GRIP_WIDTH_M:-}"
if [ -z "${GRIP_WIDTH_M}" ] && [ -r "${GRIP_FILE}" ]; then
    GRIP_WIDTH_M="$(cat "${GRIP_FILE}")"
    echo "==> grip width ${GRIP_WIDTH_M}m read from ${GRIP_FILE}"
fi

# Randomised start position. Every trial begins at the recorded reference position plus a fresh
# uniform offset bounded by JITTER_XYZ_M; the orientation is levelled identically as before and is
# never randomised.
#
# The reference is recorded ONCE, on the first trial with no position on record, and written to
# HOME_FILE alongside the grip width -- for the same reason the grip has to be absolute. Jittering
# about wherever the arm currently sits would make each trial start from the last one's offset, so
# the start would random-walk across the session and reintroduce exactly the drift this is meant
# to remove.
#
# What it buys: with a fixed start every trial photographs the same scene, so the only thing that
# varies is a per-session offset in the camera's level and colour balance, and a classifier reads
# that instead of the object -- it scored 93% on the finger camera that way, and separated two
# halves of a SINGLE class at 100%. Moving the start makes the view vary far more than any such
# offset. It MASKS the nuisance rather than removing it, though, because the offset belongs to the
# session and not to the pose, so it is worth interleaving the classes as well rather than
# collecting all of one and then all of the next.
HOME_FILE="${HOME_FILE:-${BAG_ROOT}/home_xyz}"
JITTER_XYZ_M="${JITTER_XYZ_M:-[0.03, 0.03, 0.02]}"
HOME_XYZ="${HOME_XYZ:-}"
if [ -z "${HOME_XYZ}" ] && [ -r "${HOME_FILE}" ]; then
    HOME_XYZ="$(cat "${HOME_FILE}")"
    echo "==> start reference ${HOME_XYZ} read from ${HOME_FILE}"
fi

# Seconds the recorder is given to subscribe before the arm moves. Inside the recording window,
# so the derived duration below has to include it.
SUBSCRIBE_S=3

# Default the recording window to the motion it has to contain, rather than a fixed number that
# silently truncates a slower shake: the subscribe wait, 2 s levelling, 0.5 s settle, the shakes
# themselves, and 2 s margin for planning and for the arm to come to rest.
if [ -z "${4:-}" ]; then
    DURATION=$(awk -v s="${SUBSCRIBE_S}" -v n="${N_SHAKES}" -v p="${PERIOD_S}" \
               'BEGIN{printf "%d", s + 2 + 0.5 + n*p + 2 + 0.999}')
else
    DURATION="${4}"
fi

# The label is free text -- it names a class, and the same machinery serves water levels or
# materials or anything else. Only the character set is checked, because the label becomes a
# directory name and a stray slash or space would scatter a level across two places.
case "${LEVEL}" in
    *[!a-zA-Z0-9_-]*|'')
        echo "error: class label must be non-empty and [a-zA-Z0-9_-] only (got '${LEVEL}')" >&2
        exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDER="${HERE}/record_eval_trial.sh"
SHAKE_ARGS=(--ros-args -p execute:=true
            -p "amplitude_m:=${AMPLITUDE_M}"
            -p "period_s:=${PERIOD_S}"
            -p "n_shakes:=${N_SHAKES}"
            -p "waypoint_dt:=${WAYPOINT_DT}"
            -p "shake_axis:=${SHAKE_AXIS}"
            -p "symmetric:=${SYMMETRIC}"
            -p "grip_squeeze_m:=${GRIP_SQUEEZE_M}"
            -p "jitter_xyz_m:=${JITTER_XYZ_M}")

echo "==> ${COUNT} trial(s) at level '${LEVEL}', starting at ${FIRST}, ${DURATION}s each"
echo "    shake: ${N_SHAKES} x ${PERIOD_S}s at ${AMPLITUDE_M}m along ${SHAKE_AXIS} (symmetric=${SYMMETRIC})"
echo "    start: ${HOME_XYZ:-<measured on the first trial>} +/- ${JITTER_XYZ_M} m, orientation levelled"
echo "    bags -> ${BAG_ROOT}/${LEVEL}/"
if [ -n "${GRIP_WIDTH_M}" ]; then
    echo "    grip:  ${GRIP_WIDTH_M}m, the same width every class in this dataset is held at"
else
    echo "    grip:  UNCALIBRATED -- trial ${FIRST} measures (rest width minus ${GRIP_SQUEEZE_M}m)"
    echo "           and every class collected afterwards is held at whatever it finds"
fi
echo

for i in $(seq 0 $((COUNT - 1))); do
    TRIAL=$((FIRST + i))
    echo "--- trial ${TRIAL} ($((i + 1))/${COUNT}) ---"

    # Recorder first, in the background, so it is subscribed before the arm moves. Its own
    # `timeout --signal=INT` ends it; we just wait for the process.
    TASK=water "${RECORDER}" "${LEVEL}" "${TRIAL}" "${DURATION}" &
    REC_PID=$!

    # ros2 bag needs a moment to discover publishers and subscribe. Starting the shake before it
    # has is the failure this sleep exists to prevent -- the bag would open, the arm would move,
    # and the first shake would be missing from the data with nothing to show it.
    sleep "${SUBSCRIBE_S}"

    # A width already in hand is commanded outright; without one this trial calibrates and the
    # width it reports is captured below, so only the first trial of a run ever measures.
    SHAKE_RUN=("${SHAKE_ARGS[@]}")
    if [ -n "${GRIP_WIDTH_M}" ]; then
        SHAKE_RUN+=(-p "grip_width_m:=${GRIP_WIDTH_M}")
    fi
    if [ -n "${HOME_XYZ}" ]; then
        SHAKE_RUN+=(-p "home_xyz:=${HOME_XYZ}")
    fi

    # Not a bare mktemp: an exported TMPDIR pointing at a directory that does not exist makes it
    # fail, and under `set -e` that aborts the run between the recorder starting and the shake --
    # so the bag is opened, nothing moves, and the trial is lost. Create TMPDIR if it is named,
    # then fall back to /tmp, which is the one directory that is always there.
    mkdir -p "${TMPDIR:-/tmp}" 2>/dev/null || true
    SHAKE_LOG="$(mktemp 2>/dev/null || mktemp -p /tmp)"
    ros2 run polyumi_ros2 water_shake "${SHAKE_RUN[@]}" 2>&1 | tee "${SHAKE_LOG}" || {
        echo "WARNING: shake failed on trial ${TRIAL}; the bag will be short." >&2
    }

    if [ -z "${GRIP_WIDTH_M}" ]; then
        GRIP_WIDTH_M="$(sed -n 's/.*GRIP_TARGET_M=\([0-9.]*\).*/\1/p' "${SHAKE_LOG}" | tail -1)"
        if [ -n "${GRIP_WIDTH_M}" ]; then
            # Recorded at the dataset root, not under the class, because it belongs to the whole
            # dataset: every class collected after this reads the same file and grips the same.
            mkdir -p "${BAG_ROOT}"
            printf '%s\n' "${GRIP_WIDTH_M}" > "${GRIP_FILE}"
            echo "    calibrated grip: ${GRIP_WIDTH_M}m -> ${GRIP_FILE}"
            echo "    every trial of every class from here on is held at this width."
        else
            echo "WARNING: no grip width reported on trial ${TRIAL}; the next trial will measure" >&2
            echo "         again, which lets the grip ratchet across the session. Check the" >&2
            echo "         gripper driver is up before collecting anything you intend to keep." >&2
        fi
    fi
    if [ -z "${HOME_XYZ}" ]; then
        CAPTURED="$(sed -n 's/.*HOME_XYZ=\([0-9eE.,+-]*\).*/\1/p' "${SHAKE_LOG}" | tail -1)"
        if [ -n "${CAPTURED}" ]; then
            HOME_XYZ="[${CAPTURED}]"
            mkdir -p "${BAG_ROOT}"
            printf '%s\n' "${HOME_XYZ}" > "${HOME_FILE}"
            echo "    start reference: ${HOME_XYZ} -> ${HOME_FILE}"
            echo "    every trial of every class from here on jitters about this point."
        else
            echo "WARNING: no start reference reported on trial ${TRIAL}; the next trial will use" >&2
            echo "         wherever the arm ends up, which lets the start random-walk." >&2
        fi
    fi
    rm -f "${SHAKE_LOG}"

    # The recorder stops itself at DURATION. Waiting on it (rather than sleeping) keeps the loop
    # in step with the artefact, and surfaces its non-usable-trial check per trial rather than at
    # the end of a 20-trial run.
    if wait "${REC_PID}"; then
        echo "    ok"
    else
        echo "ERROR: trial ${TRIAL} was not usable -- see the recorder's message above." >&2
        echo "       Fix before continuing; the remaining trials would fail the same way." >&2
        exit 1
    fi
    echo
done

echo "==> done. ${COUNT} trial(s) at level '${LEVEL}', all gripped at ${GRIP_WIDTH_M:-unknown}m."
echo "    count them:  ls -d ${BAG_ROOT}/${LEVEL}/trial_* | wc -l"
