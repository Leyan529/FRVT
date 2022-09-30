# FRVT

## Arcface_torch:
  * Insightface Face Distributed Arcface Training in Pytorch 
  
## Arcface_ada:
  * Insightface Face Distributed Arcface Training in Pytorch + AdaFace (With AdaFace margin version)
  * Reference: [AdaFace: Quality Adaptive Margin for Face Recognition](https://github.com/mk-minchul/AdaFace)

## ncnn_tool:
  * ncnn is a high-performance neural network forward computing framework optimized for mobile phones
  * Using [20220701](https://github.com/Tencent/ncnn/releases/tag/20220701) version
  * Prebuild-tools and library 
    - Shared library(.so) for build cmake project
    - UnShared library(.a): for create transform ncnn format model

## FRVT submission:
  * Reference Repository for the Face Recognition Vendor Test ([FRVT](https://github.com/usnistgov/frvt))
  * And only using FRVT 1:1 validation package 
  * Include task for FaceDetection & FaceRecognize

### How to use:
   ```
   mkdir -p /11/src/nullImpl/build 
   cd /11/src/nullImpl/build
   cmake .. && make clean && make && ./nullImpl
   ```
