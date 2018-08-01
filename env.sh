# Hacky script to add/remove this directory to PYTHONPATH.

_TRACK_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
if [[ -n "${PYTHONPATH}" ]] ; then
    export PYTHONPATH="${_TRACK_PROJECT_DIR}:${PYTHONPATH}"
else
    export PYTHONPATH="${_TRACK_PROJECT_DIR}"
fi
unset _TRACK_PROJECT_DIR

deactivate_track() {
    _TRACK_PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    export PYTHONPATH=${PYTHONPATH//:${_TRACK_PROJECT_DIR}/}
    export PYTHONPATH=${PYTHONPATH//${_TRACK_PROJECT_DIR}/}
    unset -f deactivate_track
    unset _TRACK_PROJECT_DIR
}
