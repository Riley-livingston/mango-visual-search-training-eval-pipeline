import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

class ModelTrainer:
    def __init__(self, data_dir, model_save_dir="models", batch_size=8, num_workers=8):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create model save directory if it doesn't exist
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize transforms, datasets, and model
        self.transform = self._get_transforms()
        self.train_loader, self.val_loader = self._setup_data_loaders()
        self.model = self._setup_model()
        
    def _get_transforms(self):
        """Define image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _setup_data_loaders(self):
        """Set up training and validation data loaders"""
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, val_loader

    def _setup_model(self):
        """Initialize and configure the model"""
        # Load pre-trained ResNet34
        model = models.resnet34(pretrained=True)
        
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Modify final fully connected layer
        num_classes = len(datasets.ImageFolder(self.data_dir).classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Unfreeze specific layers for fine-tuning
        for layer in [model.layer3, model.layer4, model.fc]:
            for param in layer.parameters():
                param.requires_grad = True
        
        return model.to(self.device)

    def train(self, num_epochs=40, learning_rate=0.0001, patience=3):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

        # Training metrics
        best_val_loss = float('inf')
        early_stop_counter = 0
        metrics = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': []
        }

        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(criterion, optimizer)
            
            # Validation phase
            val_metrics = self._validate_epoch(criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Store metrics
            for key, value in {**train_metrics, **val_metrics}.items():
                metrics[key].append(value)
            
            # Print epoch results
            self._print_epoch_results(epoch, num_epochs, train_metrics, val_metrics)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['val_loss'], best_val_loss, patience):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            best_val_loss = min(best_val_loss, val_metrics['val_loss'])

        # Save the final model
        self._save_model()
        
        return metrics

    def _train_epoch(self, criterion, optimizer):
        """Run one training epoch"""
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(self.train_loader)} processed")

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct_preds.double() / total_preds
        
        return {'train_loss': epoch_loss, 'train_acc': epoch_acc.item()}

    def _validate_epoch(self, criterion):
        """Run one validation epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data)
                total_preds += labels.size(0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = correct_preds.double() / total_preds
        
        return {'val_loss': epoch_loss, 'val_acc': epoch_acc.item()}

    def _check_early_stopping(self, current_loss, best_loss, patience):
        """Check if training should stop early"""
        if current_loss < best_loss:
            self.early_stop_counter = 0
            return False
        
        self.early_stop_counter += 1
        return self.early_stop_counter >= patience

    def _print_epoch_results(self, epoch, num_epochs, train_metrics, val_metrics):
        """Print the results for the epoch"""
        print(
            f"Epoch {epoch}/{num_epochs - 1}, "
            f"Training Loss: {train_metrics['train_loss']:.4f}, "
            f"Training Accuracy: {train_metrics['train_acc']:.4f}, "
            f"Validation Loss: {val_metrics['val_loss']:.4f}, "
            f"Validation Accuracy: {val_metrics['val_acc']:.4f}"
        )

    def _save_model(self):
        """Save the trained model"""
        model_path = self.model_save_dir / "resnet34_model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

def main():
    # Example usage
    trainer = ModelTrainer(
        data_dir="data/training_images",
        model_save_dir="models",
        batch_size=8,
        num_workers=8
    )
    
    metrics = trainer.train(
        num_epochs=40,
        learning_rate=0.0001,
        patience=3
    )

if __name__ == "__main__":
    main()