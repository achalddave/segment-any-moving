# Usage: . ./track_mot_train.sh <output_dir>
# Note that you should source this script instead of running it so that the
# current conda environment, PYTHONPATH, etc. are maintained.

if [[ "$#" != 1 ]] ; then
    echo "Usage: . $0 <output_dir>"
    return
fi
output="$1"
mkdir -p ${output}

# Make a copy of track.py so that the whole loop uses the same code. Ideally,
# we would make this a python script so that changes to track.py after the
# script is launched don't affect the code. In the mean time, we use this as a
# workaround.
track_py_backup=$(mktemp '.track.py.track_mot_train.XXXXX')
# Overwrite ${track_py_backup}. Use /usr/bin/cp to avoid 'cp -i' alias.
/usr/bin/cp track.py ${track_py_backup}
for seq_path in /ssd1/achald/MOT17/mask_roi_feat_features/train/MOT17-* ; do
    sequence="${seq_path##*/}"
    echo "Processing ${sequence}"
    python ${track_py_backup} \
        --detectron-dir ${seq_path} \
        --images-dir /ssd1/achald/MOT17/images/train/${sequence}/ \
        --extension '.jpg' \
        --output-track-file ${output}/${sequence}-FRCNN.txt # \
        # --output-video ${output}/${sequence}.mp4
    break
done

echo "Output results to:"
readlink -e ${output}
rm ${track_py_backup}
