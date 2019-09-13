# Towards Segmenting Anything That Moves

[<img src="http://www.achaldave.com/projects/anything-that-moves/videos/ZXN6A-tracked-with-objectness-trimmed.gif" width="32%" />](http://www.achaldave.com/projects/anything-that-moves/videos/ZXN6A-tracked-with-objectness-trimmed.mp4)[<img src="http://www.achaldave.com/projects/anything-that-moves/videos/c95cd17749.gif" width="32%" />](http://www.achaldave.com/projects/anything-that-moves/videos/c95cd17749.mp4)[<img src="http://www.achaldave.com/projects/anything-that-moves/videos/e0bdb5dfae.gif" width="32%" />](http://www.achaldave.com/projects/anything-that-moves/videos/e0bdb5dfae.mp4)

[[Pre-print](https://arxiv.org/abs/1902.03715)] [[Website](http://www.achaldave.com/projects/anything-that-moves/)]

[Achal Dave](http://www.achaldave.com/), [Pavel Tokmakov](http://thoth.inrialpes.fr/people/tokmakov/), [Deva Ramanan](http://www.cs.cmu.edu/~deva/)

## Setup

1. Download
   [models](https://drive.google.com/file/d/1qckICZRzX_GBTJSRhn2NDMoJuVppgWUS/view?usp=sharing)
   and extract them to release/models
1. Install pytorch 0.4.0.
1. Run `git submodule update --init`.
1. Setup [detectron-pytorch](./detectron_pytorch).
1. Setup [flownet2](https://github.com/lmb-freiburg/flownet2). If you just
want to use the appearance stream, you can skip this step.
1. Install requirements with `pip install -r requirements.txt`<sup>[1](#footnote1)</sup>.
1. Copy [`./release/example_config.yaml`](./release/example_config.yaml) to
   `./release/config.yaml`, and edit fields marked with `***EDIT THIS***`.
1. Add root directory to `PYTHONPATH`: `source ./env.sh activate`.

## Running models

All scripts needed for running our models on standard datasets, as well as on
new videos, are provided in the [`./release`](./release) directory. Outside
of the `release` directory, this repository contains a number of scripts
which are not used for the final results. They can be safely ignored, but are
provided in case anyone finds them useful.

## Run on your own video

1. **Extract frames**: To run the model on your own video, first dump the frames from your video.
For a single video, you can just use

    ```ffmpeg -i video.mp4 %04d.jpg```

    Alternatively, you can use [this
script](https://github.com/achalddave/video-tools/blob/master/dump_frames.py)
to extract frames in parallel on multiple videos.

1. **Run joint model**: To run the joint model, run the following commands:
    ```bash
    # Inputs
    FRAMES_DIR=/path/to/frames/dir
    # Outputs
    OUTPUT_DIR=/path/to/output/dir

    python release/custom/run.py \
    --model joint \
    --frames-dir ${FRAMES_DIR} \
    --output-dir ${OUTPUT_DIR}
    ```

1. **Run appearance only model**: To run only the appearance model, you don't
    need to compute optical flow, or set up flownet2:
    ```bash
    python release/custom/run.py \
    --model appearance \
    --frames-dir ${FRAMES_DIR} \
    --output-dir ${OUTPUT_DIR}
    ```


## FBMS, DAVIS 2016/2017, YTVOS

The instructions for FBMS, DAVIS 2016/2017 and YTVOS datasets are roughly the
same. Once you have downloaded the dataset and edited the paths in
`./release/config.yaml`, run the following scripts:

```bash
# or davis16, davis17, ytvos
dataset=fbms
python release/${dataset}/compute_flow.py
python release/${dataset}/infer.py
python release/${dataset}/track.py
# For evaluation:
python release/${dataset}/evaluate.py
```

Note that by default, we use our final model trained on COCO, FlyingThings3D,
DAVIS, and YTVOS. For YTVOS, we provide the option to run using a model that
was trained without YTVOS, to evaluate generalization. To activate this, pass
`--without-ytvos-train` to `release/ytvos/infer.py` and
`release/ytvos/track.py`.

---
<a name="footnote1">1</a>: This should contain all the requirements, but this was created manually so I may be missing some pip modules. If you run into an import error, try pip installing the module, and/or [file an issue](https://github.com/achalddave/segment-any-moving/issues).

