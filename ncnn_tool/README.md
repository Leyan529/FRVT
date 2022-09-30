## ncnn-20220701-ubuntu-1804
### Usage
* **onnx2ncnn**: ./onnx2ncnn [onnxpb] [ncnnparam] [ncnnbin]
* **ncnnoptimize**: ./ncnnoptimize [inparam] [inbin] [outparam] [outbin] [flag] [cutstart] [cutend]

### How to use
  ```
  ./onnx2ncnn scrfd_10g_bnkps_shape256x256.onnx output/scrfd_10g.param output/scrfd_10g.bin
  ./ncnnoptimize output/scrfd_10g.param output/scrfd_10g.bin opt-output/scrfd_10g-opt.param opt-output/scrfd_10g-opt.bin 1

  ./onnx2ncnn arcface_CurricularFace_ResNeXt200.onnx output/fr.param output/fr.bin
  ./ncnnoptimize output/fr.param output/fr.bin opt-output/fr-opt.param opt-output/fr-opt.bin 1

  ./onnx2ncnn model.onnx output/fr.param output/fr.bin
  ./ncnnoptimize output/fr.param output/fr.bin opt-output/fr-opt.param opt-output/fr-opt.bin 1
  ```

## ncnn-20220701-ubuntu-1804-shared

1. usage
- Shared library(.so) for build cmake project
- UnShared library(.a): for create transform ncnn format model

https://github.com/Tencent/ncnn/releases
