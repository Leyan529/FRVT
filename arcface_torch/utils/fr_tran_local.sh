
model_dir=resnest/resnest152_8x14d

model_type=resnest152_8x14d

cp /media/leyan/E/Git/adaface_torch/work_dirs/ms1m-retinaface-t1_${model_type}_local/model.onnx /media/leyan/E/Git/ncnn-20220701-ubuntu-1804/bin

cd /media/leyan/E/Git/ncnn-20220701-ubuntu-1804/bin
./onnx2ncnn model.onnx output/fr.param output/fr.bin
./ncnnoptimize output/fr.param output/fr.bin opt-output/fr-opt.param opt-output/fr-opt.bin 1

mv ./output/*.param /home/leyan/workspace/11/src/nullImpl/modeldir/$model_dir
mv ./output/*.bin /home/leyan/workspace/11/src/nullImpl/modeldir/$model_dir
mv ./opt-output/*.param /home/leyan/workspace/11/src/nullImpl/modeldir/$model_dir
mv ./opt-output/*.bin /home/leyan/workspace/11/src/nullImpl/modeldir/$model_dir

cd /home/leyan/workspace/11/src/nullImpl/build
cmake .. && make clean && make && ./nullImpl