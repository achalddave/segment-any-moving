CURRENT=../
OBJECTNESS=/data/achald/track/detectron-pytorch/e2e_mask_rcnn_R-50-FPN_1x/objectness-Jul14-16-38-46_compute-1-16.local_step/step89999/infer-simple-fbms/tracker/TestSet/iou-only_fix-skip-frames/

for seq_dir in ${CURRENT}/*.mp4 ; do
    seq=${seq_dir##*/}
    vid hstack ${OBJECTNESS}/${seq} ${seq_dir} ${seq} --verbose
done
