# Play with ncnn
Sample projects to use ncnn (https://github.com/Tencent/ncnn )

## Target Environment
- Platform
    - Linux (x64)
        - Tested in Xubuntu 18 in VirtualBox in Windows 10
    - Linux (armv7)
        - Tested in Raspberry Pi4 (Raspbian 32-bit)
    - Linux (aarch64)
        - Tested in Jetson Nano (JetPack 4.3) and Jetson NX (JetPack 4.4)
    - Android (aarch64)
        - Tested in Pixel 4a
    - Windows (x64). Visual Studio 2017, 2019
        - Tested in Windows10 64-bit

## Usage
```
./main [input]

 - input = blank: use the default image file set in source code (main.cpp)
    - e.g. ./main
 - input = *.mp4, *.avi, *.webm: use video file
    - e.g. ./main test.mp4
 - input = *.jpg, *.png, *.bmp: use image file
    - e.g. ./main test.jpg
 - input = number (e.g. 0, 1, 2, ...): use camera
    - e.g. ./main 0
```

## How to build application code
### Requirements
- OpenCV 4.x

### Common 
- Get source code
    ```sh
    git clone https://github.com/iwatake2222/play_with_ncnn.git
    cd play_with_ncnn
    git submodule update --init
    ```

- Download prebuilt libraries
    - Download prebuilt libraries (ThirdParty.zip) from https://github.com/iwatake2222/InferenceHelper/releases/ 
    - Extract it to `InferenceHelper/ThirdParty/`
- Download models
    - Download models (resource.zip) from https://github.com/iwatake2222/play_with_ncnn/releases
    - Extract it to `resource/`

### Linux
```
cd pj_ncnn_cls_mobilenet_v2   # for example
mkdir build && cd build
cmake ..
make
./main
```

### Option (Camera input)
```sh
cmake .. -DSPEED_TEST_ONLY=off
```

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
    - `Where is the source code` : path-to-play_with_tflite/pj_tflite_cls_mobilenet_v2	(for example)
    - `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!

### Android project
If you want to run Android project, please select `ViewAndroid` directory in Android Studio.

You will need the following settings at first.

- Configure NDK
    - File -> Project Structure -> SDK Location -> Android NDK location
        - C:\Users\abc\AppData\Local\Android\Sdk\ndk\21.3.6528147
- Import OpenCV
    - Download and extract OpenCV android-sdk (https://github.com/opencv/opencv/releases )
    - File -> New -> Import Module
        - path-to-opencv\opencv-4.3.0-android-sdk\OpenCV-android-sdk\sdk
    - FIle -> Project Structure -> Dependencies -> app -> Declared Dependencies -> + -> Module Dependencies
        - select `sdk`
    - In case you cannot import OpenCV module, remove `sdk` module and dependency of `app` to `sdk` in Project Structure

### How to create pre-built ncnn library
pre-built ncnn library is stored in InferenceHelper/ThirdParty/ncnn_prebuilt .
Please follow the instruction (https://github.com/Tencent/ncnn/wiki/how-to-build ), if you want to build them by yourself.

# License
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0
    - [LICENSE](LICENSE)

# Acknowledgements
- This project utilizes OSS (Open Source Software)
    - [NOTICE.md](NOTICE.md)
