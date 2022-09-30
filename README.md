# FRVT

## arcface_torch:
  * Insightface Face Distributed Arcface Training in Pytorch 
  
## arcface_ada:
  * Insightface Face Distributed Arcface Training in Pytorch + AdaFace (With AdaFace margin version)
  * Reference: [AdaFace: Quality Adaptive Margin for Face Recognition](https://github.com/mk-minchul/AdaFace)

## [Tencent/ncnn/20220701](https://github.com/Tencent/ncnn/releases/tag/20220701):
  * Prebuild-tools and library 
    - Shared library(.so) for build cmake project
    - UnShared library(.a): for create transform ncnn format model

## FRVT submission:
  1:1 test project(11)
  Include: /11/src/nullImpl
   * FaceDetection
   * FaceRecognize
   ```
   mkdir -p /11/src/nullImpl/build 
   cd /11/src/nullImpl/build
   cmake .. && make clean && make && ./nullImpl
   ```
