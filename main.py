
import timm
import torch
import torchvision
from timm.data import resolve_model_data_config, create_transform,create_loader,ImageDataset
from timm import create_model
from module import OnnxStaticQuantization
import torch, onnx, io, onnxsim, argparse, os
from utils import AverageMeter, accuracy
from utils import timetaken
from tqdm import tqdm 
import onnxruntime as ort


#----------------------------------------------------
artifacts_root_dir = "inception_v3"
fp32_onnx_path = f"{artifacts_root_dir}/model.fp32.onnx" 
fp16_onnx_path = f"{artifacts_root_dir}/model.fp16.onnx" 
int8_onnx_path = f"{artifacts_root_dir}/model.int8.onnx" 

os.makedirs(artifacts_root_dir, exist_ok=True)
#----------------------------------------------------

@timetaken
def export(model, dummy_input, device):
    model.to(device)
    dummy_input.to(device)
    file_handler = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        file_handler,
        export_params=True, 
        opset_version=13
    )
    onnx_model = onnx.load_from_string(file_handler.getvalue())
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.save(onnx_model, fp32_onnx_path)

def onnx_validation(onnx_path, val_loader, EP_list=['CPUExecutionProvider']):
    val_top1 = AverageMeter("Top@1", ":6.2f")
    val_top5 = AverageMeter("Top@5", ":6.2f")

    session = ort.InferenceSession(onnx_path, providers=EP_list)
    pbar = tqdm(val_loader, total=len(val_loader), desc="Onnx Validation")
    for inputs, labels in pbar:

        ort_outputs = session.run([], { 
            session.get_inputs()[0].name: inputs.numpy() 
        })[0]

        output = torch.from_numpy(ort_outputs)

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        val_top1.update(acc1.item(), labels.size(0))
        val_top5.update(acc5.item(), labels.size(0))
        pbar.set_postfix_str("Top1: {:0.2f} Top5: {:0.2f}".format(val_top1.avg, val_top5.avg))
    return

def torch_validation(model, val_loader, device):
    val_top1 = AverageMeter("Top@1", ":6.2f")
    val_top5 = AverageMeter("Top@5", ":6.2f")

    model.eval()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="PyTorch Validation")
    for idx, (images, targets) in pbar:
        # if idx == 10: break
        images, targets = images.to(device), targets.to(device)

        output = model(images)
        
        torch.cuda.synchronize()

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        val_top1.update(acc1.item(), targets.size(0))
        val_top5.update(acc5.item(), targets.size(0))
        pbar.set_postfix_str("Top1: {:0.2f} Top5: {:0.2f}".format(val_top1.avg, val_top5.avg))
    return 

def get_dataloader(args, model, onnx_val=False):
    _dir = os.path.join(args.dataset_dir, "val")
    data_config = resolve_model_data_config(model)
    dataset = ImageDataset(root='/kaggle/input/imagenet-1k-validation/imagenet_dataset', transform=create_transform(**data_config, is_training=False))
    return (
        dataset, 
        timm.data.create_loader(dataset, batch_size=1, num_workers=1, pin_memory=True)
    )

def get_model(args):
    return create_model(
        args.model_name,
        pretrained=True, 
        num_classes=1000
    )

def float_conversion():
    onnx_model = onnx.load(fp32_onnx_path)
    from onnxconverter_common import float16
    onnx_model = float16.convert_float_to_float16(
        onnx_model, keep_io_types=True, min_positive_val=1e-7, max_finite_val=1e4,
    )
    onnx.save(onnx_model, fp16_onnx_path)

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = get_model(args).to(device)
    val_dataset, val_dataloader = get_dataloader(args, model)
    dummy_input = torch.randn(1,3,299,299).to(device)

    # FP32 onnx validation
    torch_validation(model, val_dataloader, device)
    
    # Exporting Fp32 Model 
    export(model, dummy_input, device)
    
    # FP32 OnnxRuntime
    onnx_validation(fp32_onnx_path, val_dataloader)

    # FP32 to FP16 Conversion
    float_conversion()

    # FP16 OnnxRuntime
    onnx_validation(fp16_onnx_path, val_dataloader)
    
    # Quantizing the FP32 Model to INT8 Model Using QDQ
    module = OnnxStaticQuantization()

    module.fp32_onnx_path = fp32_onnx_path
    module.quantization(
        fp32_onnx_path=fp32_onnx_path,
        future_int8_onnx_path=int8_onnx_path,
        calib_method="Percentile",
        calibration_loader=val_dataloader,
        sample=100
    )

    # INT8 OnnxRuntime
    onnx_validation(int8_onnx_path, val_dataloader)

    return

def get_argparse():
    parser = argparse.ArgumentParser(description='Create project')
    parser.add_argument('--model_name', required=True, type=str, help="")
    parser.add_argument('--dataset_dir', required=True, type=str, help="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":    
    args = get_argparse()
    main(args)