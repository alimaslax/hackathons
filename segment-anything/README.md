# SAM.cpp

Inference of Meta's [Segment Anything Model](https://github.com/facebookresearch/segment-anything/) in pure C/C++
https://github.com/YavorGIvanov/sam.cpp/assets/1991296/a69be66f-8e27-43a0-8a4d-6cfe3b1d9335

## Requirements
Note: you need to download the model checkpoint below (`sam_vit_b_01ec64.pth`) first from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it in the `checkpoints` folder
```bash
# Convert PTH model to ggml. Requires python3, torch and numpy
python convert-pth-to-ggml.py checkpoints/sam_vit_b_01ec64.pth . 1
```

### brew libraries
```bash
brew install sdl2
```

### ggml and imgui cpp

clone these two commits
```bash
segment-anything/ggml @ dd92cfd
segment-anything/imgui @ 7b5fb33
```
```bash
git clone https://github.com/ggerganov/ggml.git
git checkout dd92cfd
cd examples/third-party/imgui && git clone https://github.com/ocornut/imgui.git
git checkout 7b5fb33
```

## Building

```bash
# Build sam.cpp.
mkdir build && cd build
cmake .. && make -j4
```

## Running the Model

```bash
# run inference
./bin/sam -t 16 -i ../examples/img.jpg -m ../checkpoints/ggml-model-f16.bin
```