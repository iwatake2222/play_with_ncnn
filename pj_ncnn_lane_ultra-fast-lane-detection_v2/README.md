# Ultra-Fast-Lane-Detection-V2 with ncnn in C++

![00_doc/demo.jpg](00_doc/demo.jpg)

## How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_ncnn/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh
        - Convert `ufldv2_culane_res18_320x1600.onnx` using `onnx2ncnn`
        - Place the generated ncnn model to `resource/model/ufldv2_culane_res18_320x1600.param` and `.bin`
    - Build `pj_ncnn_lane_ultra-fast-lane-detection_v2` project (this directory)


## Acknowledgements
- https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2
- https://github.com/PINTO0309/PINTO_model_zoo
- Video
    - Drive Video by Dashcam Roadshow
    - 4K Tokyo Drive thru Ikebukuro, Shinjuku, Shibuya.mp4
    - https://www.youtube.com/watch?v=tTuUjnISt9s
