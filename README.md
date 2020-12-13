# Play with ncnn
Sample projects to use ncnn (https://github.com/Tencent/ncnn )

## Target environment
 - Windows(MSVC2017) (x64)
 - Linux (x64)
 - Linux (armv7) e.g. Raspberry Pi 3,4
 - Linux (aarch64) e.g. Jetson Nano
 - *Native build only (Cross build is not supported)


## How to build application code
### Preparation
- Please download the following files from [Releases](https://github.com/iwatake2222/play_with_ncnn/releases ) in GitHub, and extract them to the same name directory
	- third_party.zip
	- resource.zip

### Linux
```
cd pj_ncnn_cls_mobilenet_v2   # for example
mkdir build && cd build
cmake ..
# cmake -DSPEED_TEST_ONLY=off ..	# for camera input
make

./main
```

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
	- `Where is the source code` : path-to-play_with_ncnn/project_cls_mobilenet_v2	(for example)
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
pre-built ncnn library is stored in third_party/ncnn_prebuilt (please download from [Releases](https://github.com/iwatake2222/play_with_ncnn/releases )). 
Please follow the instruction (https://github.com/Tencent/ncnn/wiki/how-to-build ), if you want to build them by yourself.

# License
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0
	- [LICENSE](LICENSE)

# Acknowledgements
- This project utilizes OSS (Open Source Software)
	- [NOTICE.md](NOTICE.md)
