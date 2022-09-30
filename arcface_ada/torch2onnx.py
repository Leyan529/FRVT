import numpy as np
import onnx
import torch
import onnxruntime as ort
import cv2
import datetime
# import utils.checkpoint as cu

def convert_onnx(net, path_module, output, opset=10, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    net = net.to("cpu")
    net.eval()  

    dict_checkpoint = torch.load(path_module, map_location="cpu")
    net.load_state_dict(dict_checkpoint["state_dict_backbone"])

    torch.onnx.export(net, img, output, 
                        input_names=["actual_input_1"], 
                        output_names = [ "output%d" % i for i in range(1) ], 
                        verbose=True, opset_version=opset)

    model = onnx.load(output)
    # graph = model.graph
    # graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        # model, check = simplify(model, dynamic_input_shape=True)
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    # Check that the model is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    graph = onnx.helper.printable_graph(model.graph)
    onnx_graph_path = output.replace(".onnx", ".txt")
    with open(onnx_graph_path, "w", encoding="utf-8") as f:
        f.write(graph)

    # Test forward with onnx session (test image) 
    # ort_session = ort.InferenceSession(output, providers=['CUDAExecutionProvider'])     
    ort_session = ort.InferenceSession(output, providers=['CPUExecutionProvider'])  

    # Test image
    sample_image = os.path.join("tests/data/00000001.jpg") 
    image = cv2.imread(sample_image)   
    image_shape = np.array(np.shape(image)[0:2])

    new_image       = cv2.resize(image, (args.input_shape[0], args.input_shape[1]), interpolation=cv2.INTER_CUBIC)
    new_image       = np.expand_dims(np.transpose(np.array(new_image, dtype=np.float32)/255, (2, 0, 1)), 0) # (1, 3, 112, 112)

    time_l = []

    for i in range(0, 1000):
        ta = datetime.datetime.now()
        outputs = ort_session.run(
            None,
            {"actual_input_1": new_image}, # (1, 512)
        )
        tb = datetime.datetime.now()
        time_l.append((tb-ta).total_seconds()*1000)
        print('all cost:', (tb-ta).total_seconds()*1000, " ms")

        print(outputs[0].shape)

    print("average time: ", sum(time_l)/len(time_l), " ms")

    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('--input', type=str,       
        # default="work_dirs/ms1m-retinaface-t1_resnest200_8x14d_2022_7_20/model_epoch_0000_step_000300.pt",
        # default="work_dirs/ms1m-retinaface-t1_resnext200_8x14d_2022_7_19/model_epoch_0000_step_260000.pt",
        default="work_dirs/WebFace42M_resnet_269/epoch_20/epoch_gpu_1.pt",
        # default="work_dirs/WebFace42M_resnext152_8x14d_2022_7_20/model_epoch_0000_step_080000.pt",
        # default="work_dirs/WebFace42M_resnest152_1x64d_2022_7_21/model_epoch_0000_step_096000.pt",
        # default="work_dirs/WebFace42M_resnest152_8x14d_2022_7_27/model_epoch_0000_step_118000.pt",
        # default="work_dirs/WebFace42M_resnet_269_2022_7_26/model_epoch_0000_step_142000.pt",
        help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    # parser.add_argument('--network', type=str, default="bottlenet_resnet_269", help='backbone network')
    # parser.add_argument('--network', type=str, default="resnext200_8x14d", help='backbone network')
    parser.add_argument('--network', type=str, default="resnet_269", help='backbone network')
    parser.add_argument('--simplify', type=bool, default=True, help='onnx simplify')
    parser.add_argument("--input_shape", type=int, nargs="+", default=(112, 112, 3))
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
    assert args.network is not None
    print(args)
    # network = args.network.replace("ms1mv1_arcface_", "")
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    convert_onnx(backbone_onnx, input_file, args.output, simplify=args.simplify)
