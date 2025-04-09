import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import optuna
from loss import EdgeAwareRegressionLoss, TotalVariationRegressionLoss, StructuralEdgeRegressionLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, 
                 scheduler=None, num_epochs=100, early_stopping_patience=10,
                 loss_type="edge_aware", edge_weight=1.0, smoothness_weight=0.1, 
                 tv_weight=0.1, ssim_weight=0.5):
        """
        Initialize trainer with various edge-preserving loss options for regression tasks.
        
        Args:
            loss_type: One of ["mse", "edge_aware", "total_variation", "structural"]
            edge_weight: Weight of edge preservation term
            smoothness_weight: Weight of smoothness constraint (for edge_aware loss)
            tv_weight: Weight of total variation term (for total_variation loss)
            ssim_weight: Weight of structural similarity term (for structural loss)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.optimizer = optimizer
        
        # Set up loss function based on type
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "edge_aware":
            self.criterion = EdgeAwareRegressionLoss(
                edge_weight=edge_weight, 
                smoothness_weight=smoothness_weight
            )
        elif loss_type == "total_variation":
            self.criterion = TotalVariationRegressionLoss(
                tv_weight=tv_weight
            )
        elif loss_type == "structural":
            self.criterion = StructuralEdgeRegressionLoss(
                edge_weight=edge_weight,
                ssim_weight=ssim_weight
            )
        else:
            print(f"Unknown loss type: {loss_type}, defaulting to MSE")
            self.criterion = nn.MSELoss()
            
        self.loss_type = loss_type
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, targets in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_model.pth')
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.plot_losses()
        return self.best_val_loss

    def save_model(self, filename):
        model_name = self.model.get_model_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{model_name}_{timestamp}_{filename}"
        self.model.save_model(save_path)
        print(f"Model saved to {save_path}")

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Losses for {self.model.get_model_name()} with {self.loss_type} loss')
        plt.legend()
        
        plot_path = f"{self.model.get_model_name()}_{self.loss_type}_loss_plot.png"
        plt.savefig(plot_path)
        print(f"Loss plot saved to {plot_path}")

def objective(trial, model_class, train_loader, val_loader, device):
    # Sample hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    loss_type = trial.suggest_categorical('loss_type', 
                                         ['edge_aware', 'total_variation', 'structural'])
    
    # Loss-specific hyperparameters
    if loss_type == 'edge_aware':
        edge_weight = trial.suggest_float('edge_weight', 0.1, 2.0)
        smoothness_weight = trial.suggest_float('smoothness_weight', 0.01, 0.5)
        tv_weight = 0.1  # Default, not used
        ssim_weight = 0.5  # Default, not used
    elif loss_type == 'total_variation':
        tv_weight = trial.suggest_float('tv_weight', 0.01, 0.5)
        edge_weight = 1.0  # Default, not used
        smoothness_weight = 0.1  # Default, not used
        ssim_weight = 0.5  # Default, not used
    elif loss_type == 'structural':
        edge_weight = trial.suggest_float('edge_weight', 0.1, 2.0)
        ssim_weight = trial.suggest_float('ssim_weight', 0.1, 1.0)
        smoothness_weight = 0.1  # Default, not used
        tv_weight = 0.1  # Default, not used
    else:
        edge_weight = 1.0
        smoothness_weight = 0.1
        tv_weight = 0.1
        ssim_weight = 0.5
    
    # Create model
    model = model_class(input_channels=7, output_channels=1).to(device)
    
    # Set up optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # Set up scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Create trainer with selected loss
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=50,  # Reduced for quicker trials
        early_stopping_patience=10,
        loss_type=loss_type,
        edge_weight=edge_weight,
        smoothness_weight=smoothness_weight,
        tv_weight=tv_weight,
        ssim_weight=ssim_weight
    )
    
    return trainer.train()

def train_model(model_class, train_loader, val_loader, in_channel=7, out_channel=1, epochs=100, 
                n_trials=50, use_optuna=True, loss_type="edge_aware", edge_weight=1.0,
                smoothness_weight=0.1, tv_weight=0.1, ssim_weight=0.5):
    """
    Train a model with edge-preserving loss for regression tasks.
    
    Args:
        loss_type: One of ["mse", "edge_aware", "total_variation", "structural"]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, model_class, train_loader, val_loader, device), n_trials=n_trials)
        
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        # Train final model with best hyperparameters
        best_lr = trial.params['lr']
        best_optimizer = getattr(optim, trial.params['optimizer'])
        best_loss_type = trial.params.get('loss_type', loss_type)
        
        # Get loss-specific parameters if they exist
        if best_loss_type == 'edge_aware':
            best_edge_weight = trial.params.get('edge_weight', edge_weight)
            best_smoothness_weight = trial.params.get('smoothness_weight', smoothness_weight)
            best_tv_weight = tv_weight
            best_ssim_weight = ssim_weight
        elif best_loss_type == 'total_variation':
            best_tv_weight = trial.params.get('tv_weight', tv_weight)
            best_edge_weight = edge_weight
            best_smoothness_weight = smoothness_weight
            best_ssim_weight = ssim_weight
        elif best_loss_type == 'structural':
            best_edge_weight = trial.params.get('edge_weight', edge_weight)
            best_ssim_weight = trial.params.get('ssim_weight', ssim_weight)
            best_smoothness_weight = smoothness_weight
            best_tv_weight = tv_weight
        else:
            best_edge_weight = edge_weight
            best_smoothness_weight = smoothness_weight
            best_tv_weight = tv_weight
            best_ssim_weight = ssim_weight
    else:
        best_lr = 0.001
        best_optimizer = optim.Adam
        best_loss_type = loss_type
        best_edge_weight = edge_weight
        best_smoothness_weight = smoothness_weight
        best_tv_weight = tv_weight
        best_ssim_weight = ssim_weight
    
    final_model = model_class(input_channels=in_channel, output_channels=out_channel).to(device)
    final_optimizer = best_optimizer(final_model.parameters(), lr=best_lr)
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.1, patience=5)
    
    final_trainer = Trainer(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=final_optimizer,
        device=device,
        scheduler=final_scheduler,
        num_epochs=epochs,
        early_stopping_patience=10,
        loss_type=best_loss_type,
        edge_weight=best_edge_weight,
        smoothness_weight=best_smoothness_weight,
        tv_weight=best_tv_weight,
        ssim_weight=best_ssim_weight
    )
    
    final_trainer.train()
    return final_model