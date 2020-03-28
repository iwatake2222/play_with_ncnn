# Play with ncnn
## How to build
### Common (Get source code)
```sh
git clone https://github.com/iwatake2222/play_with_ncnn.git
cd play_with_ncnn

# if needed
git submodule init
git submodule update
```

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
	- `Where is the source code` : path-to-play_with_ncnn/project_cls_mobilenet_v2	(for example)
	- `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!


### Linux (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
```
cd project_cls_mobilenet_v2   # for example
mkdir build && cd build

cmake ..	# For PC Linux
cmake .. -DARCH_TYPE=armv7	# For Raspberry Pi 3, 4 (Raspbian(32-bit))
cmake .. -DARCH_TYPE=aarch64	# For Jetson Nano (AArch64(ARMv8))

make
./main
```

## Acknowledgement
- This project includes external library:
	- ncnn (https://github.com/Tencent/ncnn)
- This project includes models:
	- MobileNet v2 (https://github.com/onnx/models)


