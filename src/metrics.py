import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from sklearn.metrics import accuracy_score
# from fvcore.nn import FlopCountAnalysis
# from ptflops import get_model_complexity_info
from torchprofile import profile_macs


# 1. Total Parameter Count (Pre-Pruning)
def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 2. Effective Parameter Count (Post-Pruning)
def calculate_pruned_parameter_count(model):
    total_params = 0
    nonzero_params = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = module.weight.data
            total_params += weight.numel()
            nonzero_params += torch.count_nonzero(weight).item()  # 统计非零参数
    return nonzero_params

# 3. FLOPs Calculation
# def calculate_flops(model, input_size=(1, 3, 224, 224)):
#     input_tensor = torch.randn(input_size).to(device)
#     try:
#         flops = FlopCountAnalysis(model, input_tensor)
#         return flops.total() / 1e9  # GFLOPs
#     except Exception as e:
#         print(f"Error calculating FLOPs: {e}")
#         return None

# # Alternative FLOPs with ptflops (optional)
# def calculate_flops_with_ptflops(model, input_res=(3, 224, 224)):
#     try:
#         flops, _ = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
#         return flops / 1e9  # GFLOPs
#     except Exception as e:
#         print(f"Error with ptflops: {e}")
#         return None

def calculate_macs(device, model, input_size=(1, 3, 224, 224)):
    macs = profile_macs(model, torch.rand(input_size).to(device))
    return macs / 1e6  # MMACs

# 4. Inference Latency
import torch
import time

def calculate_inference_latency(device, model, input_size=(1, 3, 224, 224), num_runs=100):
    model.eval()
    input_data = torch.randn(input_size).to(device)

    # 预热模型
    with torch.no_grad():
        model(input_data)
    
    # 开始测量推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(input_data)
            # GPU 上执行时确保完成同步
            if device == "cuda":
                torch.cuda.synchronize()
    
    avg_latency = (time.time() - start_time) / num_runs * 1000  # 毫秒
    return avg_latency

# 5. Memory Footprint
def calculate_memory_footprint(model, input_size=(1, 3, 224, 224), device='cuda'):
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()
    
    # 创建输入数据并移动到指定设备
    input_data = torch.randn(input_size).to(device)
    
    # 清空显存缓存并重置峰值内存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # 计算内存占用
    with torch.no_grad():
        output = model(input_data)  # 前向传播计算输出
    
    # 计算模型输出的内存占用
    output_memory = sum([o.element_size() * o.nelement() for o in output]) / (1024 ** 2) if isinstance(output, (tuple, list)) else output.element_size() * output.nelement() / (1024 ** 2)
    
    # 获取峰值内存使用量
    max_memory = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # 转为 MB
    
    # 清空缓存
    torch.cuda.empty_cache()
    
    return max_memory + output_memory  # 返回总内存占用，包括输出的内存

#6. Accuracy Calculation
def calculate_accuracy(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy()) * 100
    return accuracy

# Gather all metrics
def evaluate_model(model, test_loader, device):
    print('--Metrics--')
    metrics = {}
    metrics["Total Parameters (M)"] = count_total_parameters(model) / 1e6  # in millions
    # metrics["Positive Parameters (M)"] = calculate_pruned_parameter_count(model) / 1e6  # in millions
    # metrics["FLOPs (GFLOPs)"] = calculate_flops(model)
    # Optional alternative for FLOPs
    # metrics["MACs (M)"] = calculate_macs(device, model)
    metrics["Inference Latency (ms)"] = calculate_inference_latency(device, model)
    # metrics["Memory Footprint (MB)"] = calculate_memory_footprint(model)
    metrics["Top-1 Accuracy (%)"] = calculate_accuracy(model, test_loader, device)
    for metric, value in metrics.items():
        print(f"{metric}: {value if value is not None else 'Error'}")
    return metrics

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import models, datasets, transforms

    root = '/home/usrs/wang.changlong.s8/datasets'
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = torch.nn.Linear(1280, 10)
    # model = models.resnet18(weights=None)
    # model.fc = nn.Linear(512, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluate_model(model, test_loader, device)

