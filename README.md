# FRVT

## arc_torch:
  Insightface FR training code

## ncnn_tool:
  include two zip files to transform model & share libs for submission project

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
