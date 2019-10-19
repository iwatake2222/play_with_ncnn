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

### Windows (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
```
cd mobilenet_v2   # for example
mkdir build && cd build
cmake ..
make
./play_with_ncnn_mobilenet_v2
```

## Acknowledge
- This project includes external library:
	- ncnn (https://github.com/Tencent/ncnn)
- This project includes models:
	- MobileNet v2 (https://github.com/onnx/models)


