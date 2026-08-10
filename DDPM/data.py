import os
from dataclasses import dataclass
from torchvision import transforms

@dataclass
class TrainingConfig:
    image_size = 64
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"
    output_dir = "ddpm-animefaces-64"
    overwrite_output_dir = True

config = TrainingConfig()

def get_transforms():
    preprocess = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    def transform(sample):
        image = [preprocess(img.convert("RGB")) for img in sample["image"]]
        return dict(image=image)
    
    return transform

