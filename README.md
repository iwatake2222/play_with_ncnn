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
	- `Where is the source code` : path-to-play_with_ncnn/mobilenet_v2	(for example)
	- `Where to build the binaries` : path-to-build	(any)
- Open `play_with_ncnn_mobilenet_v2.sln`
- Set `play_with_ncnn_mobilenet_v2` project as a startup project, then build and run!

You may see `Protobuf not found, caffe model convert tool won't be built` warning, but you can ignore it. Or, you can disable `NCNN_BUILD_TOOLS`

### Linux (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
```
cd mobilenet_v2   # for example
mkdir build && cd build

# For PC Linux
cmake ..

# For Raspberry Pi
cmake -DCMAKE_TOOLCHAIN_FILE=../../ncnn/toolchains/pi3.toolchain.cmake -DPI3=ON  ..

# For Jetson Nano (without vulkan)
cmake -DCMAKE_TOOLCHAIN_FILE=../../ncnn/toolchains/jetson.toolchain.cmake -DNCNN_VULKAN=OFF ..

# For Jetson Nano (in case the above command fails)
cmake ..

make
./play_with_ncnn_mobilenet_v2
```

## Note
- I recommend you generate library (.a, .lib) and keep them. So that you don't need to build ncnn library every time.
	- this project structure and cmake may be useful: https://github.com/iwatake2222/NcnnMultiPlatformProject

## Acknowledge
- This project includes external library:
	- ncnn (https://github.com/Tencent/ncnn)
- This project includes models:
	- MobileNet v2 (https://github.com/onnx/models)


