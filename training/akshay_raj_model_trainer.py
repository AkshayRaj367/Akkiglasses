# USAGE
# python akshay_raj_model_trainer.py --config config/training_config.yaml --mode train
# python akshay_raj_model_trainer.py --config config/training_config.yaml --mode finetune
# python akshay_raj_model_trainer.py --config config/training_config.yaml --mode evaluate

import os
import yaml
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import json
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomObjectDetectionDataset(Dataset):
    """Custom dataset class for object detection training"""
    
    def __init__(self, data_dir, annotations_file, image_size=416, transforms=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transforms = transforms
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_files = list(self.annotations.keys())
        logger.info(f"Loaded {len(self.image_files)} images for training")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.annotations[img_name]
        
        # Convert annotations to required format
        boxes = []
        labels = []
        
        for ann in annotations:
            boxes.append([ann['x'], ann['y'], ann['width'], ann['height']])
            labels.append(ann['class_id'])
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx
        }

class YOLOTrainer:
    """YOLO model trainer with fine-tuning capabilities"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model based on config
        self.model = self._load_model()
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_loss()
        self.scheduler = self._setup_scheduler()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
    def _load_model(self):
        """Load YOLO model or create custom model"""
        if self.config['model']['type'] == 'yolo':
            # Load YOLOv5 or implement custom YOLO
            try:
                import ultralytics
                model = ultralytics.YOLO(self.config['model']['weights'])
                return model
            except ImportError:
                logger.warning("Ultralytics not installed, using custom implementation")
                return self._create_custom_yolo()
        else:
            raise ValueError(f"Model type {self.config['model']['type']} not supported")
    
    def _create_custom_yolo(self):
        """Create a simplified YOLO-like model for training"""
        import torchvision.models as models
        
        # Use ResNet backbone for feature extraction
        backbone = models.resnet50(pretrained=True)
        
        # Remove the final layer
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Add detection head
        class DetectionHead(nn.Module):
            def __init__(self, in_channels, num_classes, num_anchors=3):
                super().__init__()
                self.num_classes = num_classes
                self.num_anchors = num_anchors
                
                self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)
                
            def forward(self, x):
                return self.conv(x)
        
        class SimpleYOLO(nn.Module):
            def __init__(self, backbone, detection_head):
                super().__init__()
                self.backbone = backbone
                self.detection_head = detection_head
                
            def forward(self, x):
                features = self.backbone(x)
                detections = self.detection_head(features)
                return detections
        
        detection_head = DetectionHead(2048, self.config['model']['num_classes'])
        model = SimpleYOLO(backbone, detection_head)
        
        return model.to(self.device)
    
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        if self.config['training']['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(), 
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Optimizer {self.config['training']['optimizer']} not supported")
    
    def _setup_loss(self):
        """Setup loss function"""
        return nn.MSELoss()  # Simplified for demonstration
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['training']['lr_step'],
            gamma=self.config['training']['lr_gamma']
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != 'image'}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss (simplified implementation)
            loss = self.criterion(outputs, images)  # Placeholder loss calculation
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log batch results
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return epoch_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != 'image'}
                
                outputs = self.model(images)
                loss = self.criterion(outputs, images)  # Placeholder
                val_loss += loss.item()
        
        return val_loss / len(dataloader)
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            else:
                logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config['training']['save_frequency'] == 0:
                self.save_checkpoint(epoch)
        
        logger.info("Training completed!")
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = "checkpoints/best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(f'{plots_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training curves saved to plots/training_curves.png")

def get_transforms(config, is_training=True):
    """Get data augmentation transforms"""
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(p=0.2),
            A.Resize(config['data']['image_size'], config['data']['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(config['data']['image_size'], config['data']['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Object Detection Model Training')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--mode', choices=['train', 'finetune', 'evaluate'], 
                       default='train', help='Training mode')
    parser.add_argument('--checkpoint', help='Path to checkpoint for resume training')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create data transforms
    train_transforms = get_transforms(config, is_training=True)
    val_transforms = get_transforms(config, is_training=False)
    
    # Create datasets
    train_dataset = CustomObjectDetectionDataset(
        data_dir=config['data']['train_data_dir'],
        annotations_file=config['data']['train_annotations'],
        image_size=config['data']['image_size'],
        transforms=train_transforms
    )
    
    if config['data']['val_data_dir']:
        val_dataset = CustomObjectDetectionDataset(
            data_dir=config['data']['val_data_dir'],
            annotations_file=config['data']['val_annotations'],
            image_size=config['data']['image_size'],
            transforms=val_transforms
        )
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    ) if val_dataset else None
    
    # Initialize trainer
    trainer = YOLOTrainer(config)
    
    # Load checkpoint if resuming training
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    if args.mode in ['train', 'finetune']:
        trainer.train(train_loader, val_loader)
    elif args.mode == 'evaluate':
        # Implement evaluation logic
        logger.info("Evaluation mode - implement evaluation logic here")

if __name__ == "__main__":
    main()