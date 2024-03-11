import torch
import tvm
import numpy as np
from tvm import relay
import onnx
import time
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import timm
from timm.data import create_transform,create_loader
model = timm.create_model('inception_v3', pretrained=True).to(device)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
imagenet_val_dataset = timm.data.ImageDataset('/kaggle/input/imagenet-1k-validation',transform=transforms)
data_loader1 = timm.data.create_loader(imagenet_val_dataset, (1,3,299,299), 1,device = device)
model_path = '/kaggle/input/quantised/onnx/quantisation/1/inception_v3_quant.onnx'
onnx_model = onnx.load(model_path)
input_shape = (1, 3, 299, 299)
input_name = "input.1"
shape_dict = {input_name: input_shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, params
    ).evaluate()
x,y = next(iter(data_loader1))
ndarray = x.numpy()
input_data = tvm.nd.array(ndarray.astype("float32"))
start_time = time.time()
output = executor(input_data).numpy()
end_time = time.time()
inference_time = end_time - start_time
print("Inference Time:", inference_time, "seconds")
top1_correct = 0
top5_correct = 0
total_samples = 0
#acc check
for idx, (images, labels) in tqdm(enumerate(data_loader1), total=1000, desc="Processing images"):
    # Set the input data
    numpy_images = images.numpy()
    input_data = tvm.nd.array(numpy_images.astype("float32"))
    tvm_output = executor(input_data).numpy()
    predicted_labels = np.argmax(tvm_output, axis=1)
    top1_correct += np.sum(predicted_labels == labels.numpy())
    # Calculate top-5 accuracy
    top5_predicted_labels = np.argsort(tvm_output, axis=1)[:, -5:]
    for i in range(labels.size(0)):
        if labels.numpy()[i] in top5_predicted_labels[i]:
            top5_correct += 1

    total_samples += labels.size(0)
    if idx >= 1000:
        break

# Calculate accuracy
top1_accuracy = top1_correct / total_samples
top5_accuracy = top5_correct / total_samples

print(f"Top-1 Accuracy: {top1_accuracy * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
