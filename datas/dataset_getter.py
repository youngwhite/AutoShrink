from torch.utils.data import Dataset
from torchvision import transforms

class DatasetAdapter(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        if 'img' in example.keys():
            image = example['img']
        elif 'image' in example.keys():
            image = example['image']
        else:
            raise ValueError("Unknown image key")
        
        if 'label' in example:
            label = example['label']
        elif 'fine_label' in example:
            label = example['fine_label']
        else:
            raise ValueError("Unknown label key")

        if self.transform:
            image = self.transform(image)
        return image, label

def get_datasets(dataset_name: str, fold: int=1):
    from datasets import load_dataset # hugging face

    if dataset_name == 'mnist':
        num_classes = 10
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

        dataset = load_dataset(dataset_name)
        trainset, valset = dataset['train'], dataset['test']

    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        num_classes = 10 if dataset_name == 'cifar10' else 100
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]

        dataset = load_dataset(dataset_name)
        trainset, valset = dataset['train'], dataset['test']

    elif dataset_name == 'tiny-imagenet':
        num_classes = 200
        
        dataset = load_dataset("zh-plus/tiny-imagenet")
        trainset, valset = dataset['train'], dataset['valid']    

        # # PIL.Image.Image -> torch.Tensor
        # trainset.set_format(type='torch', columns=['image', 'label'])
        # valset.set_format(type='torch', columns=['image', 'label'])

        # # ->增加 id
        # trainset = trainset.map(lambda example, idx: {**example, 'id': idx}, with_indices=True)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整亮度、对比度、饱和度
        transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为 3 通道
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为 3 通道
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])        

    return {
        'trainset': DatasetAdapter(trainset,  transform=train_transform),
        'valset': DatasetAdapter(valset, transform=val_transform),
        'num_classes': num_classes
    }

if __name__ == '__main__':
    for name in ['mnist', 'cifar10', 'cifar100', 'tiny-imagenet']:
        d = get_datasets(name)
        trainset, valset, num_classes = d['trainset'], d['valset'], d['num_classes']
        X, y = trainset[0]
        print(f"--{name}, X.shape:{X.shape}, label:{y}, num_classes:{num_classes}")
