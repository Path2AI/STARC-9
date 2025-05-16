import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import SAVE_DIR, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CLASS_NAMES, METRICS

import warnings
warnings.filterwarnings("ignore")


def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, 
                num_epochs=NUM_EPOCHS, device=None, save_dir=SAVE_DIR, model_name="model"):
    """
    Train the model and save the best model based on test set performance.
    Each epoch's training is wrapped in try/except to prevent interruption on warnings/errors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save paths
    best_model_path = os.path.join(model_dir, f"best_{model_name}.pth")
    final_model_path = os.path.join(model_dir, f"final_{model_name}.pth")
    history_path = os.path.join(model_dir, f"history_{model_name}.csv")
    epoch_metrics_path = os.path.join(model_dir, f"epoch_metrics_{model_name}.csv")
    
    # Print model details
    print("\n" + "="*80)
    print(f"MODEL: {model_name}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Save directory: {model_dir}")
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*80 + "\n")
    
    # Initialize best metrics and history dictionary
    best_test_f1 = 0.0
    history = {
        'epoch': [], 'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [], 
        'train_f1': [], 'test_f1': [],
        'train_precision': [], 'test_precision': [],
        'train_recall': [], 'test_recall': [],
        'train_f1_micro': [], 'test_f1_micro': [],
        'learning_rate': [], 
        'epoch_time': [], 'train_time': [], 'test_time': []
    }
    
    # Write CSV header for per-epoch metrics
    with open(epoch_metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'train_f1', 'train_precision', 'train_recall', 'train_f1_micro',
            'test_loss', 'test_acc', 'test_f1', 'test_precision', 'test_recall', 'test_f1_micro',
            'learning_rate', 'epoch_time', 'train_time', 'test_time'
        ])
    
    start_time = time.time()
    epoch_pbar = tqdm(range(num_epochs), desc=f"Training {model_name}", position=0)
    
    # Training loop wrapped in try/except per epoch to catch errors/warnings
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch+1)
        history['learning_rate'].append(current_lr)
        
        # Training phase inside try/except
        try:
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            train_start_time = time.time()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", 
                              position=1, leave=False)
            
            for inputs, labels in train_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)): # # if a model returns (features, logits), use logits for loss
                    logits = outputs[-1]
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average='macro')
            train_f1_micro = f1_score(all_labels, all_preds, average='micro')
            train_precision = precision_score(all_labels, all_preds, average='macro')
            train_recall = recall_score(all_labels, all_preds, average='macro')
            train_time = time.time() - train_start_time
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['train_f1_micro'].append(train_f1_micro)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_time'].append(train_time)
        except Exception as e:
            print(f"Error during training epoch {epoch+1}: {e}")
            # Optionally log error and set dummy values
            train_loss, train_acc, train_f1, train_f1_micro = float('nan'), float('nan'), float('nan'), float('nan')
            train_precision, train_recall = float('nan'), float('nan')
            train_time = 0.0
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['train_f1_micro'].append(train_f1_micro)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_time'].append(train_time)
        
        # Testing phase inside try/except
        try:
            model.eval()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            test_start_time = time.time()
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Test)", 
                             position=1, leave=False)
            
            with torch.no_grad():
                for inputs, labels in test_pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    if isinstance(outputs, (tuple, list)):   # if a model returns (features, logits), use logits for loss
                        logits = outputs[-1]
                    else:
                        logits = outputs
                    loss = criterion(logits, labels)
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(logits, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    test_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            test_loss = running_loss / len(test_loader.dataset)
            test_acc = accuracy_score(all_labels, all_preds)
            test_f1 = f1_score(all_labels, all_preds, average='macro')
            test_f1_micro = f1_score(all_labels, all_preds, average='micro')
            test_precision = precision_score(all_labels, all_preds, average='macro')
            test_recall = recall_score(all_labels, all_preds, average='macro')
            test_time = time.time() - test_start_time
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_f1'].append(test_f1)
            history['test_f1_micro'].append(test_f1_micro)
            history['test_precision'].append(test_precision)
            history['test_recall'].append(test_recall)
            history['test_time'].append(test_time)
        except Exception as e:
            print(f"Error during testing epoch {epoch+1}: {e}")
            test_loss, test_acc, test_f1, test_f1_micro = float('nan'), float('nan'), float('nan'), float('nan')
            test_precision, test_recall = float('nan'), float('nan')
            test_time = 0.0
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_f1'].append(test_f1)
            history['test_f1_micro'].append(test_f1_micro)
            history['test_precision'].append(test_precision)
            history['test_recall'].append(test_recall)
            history['test_time'].append(test_time)
        
        epoch_time = time.time() - epoch_start_time
        history['epoch_time'].append(epoch_time)
        
        # Step the scheduler (without arguments; can change if needed)
        try:
            scheduler.step()
        except Exception as e:
            print(f"Scheduler step error at epoch {epoch+1}: {e}")
        
        # Save model if test F1 improves
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1_micro': test_f1_micro,
            }, best_model_path)
            epoch_pbar.set_postfix({
                "Best F1": f"{best_test_f1:.4f}",
                "Saved": "Yes"
            })
        else:
            epoch_pbar.set_postfix({
                "Best F1": f"{best_test_f1:.4f}",
                "Current F1": f"{test_f1:.4f}"
            })
        
        # Append epoch metrics to CSV
        with open(epoch_metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, train_loss, train_acc, train_f1, train_precision, train_recall, train_f1_micro,
                test_loss, test_acc, test_f1, test_precision, test_recall, test_f1_micro,
                current_lr, epoch_time, train_time, test_time
            ])
        
        # Print epoch summary with expanded metrics
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Time: {train_time:.2f}s")
        print(f"       Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"       F1-Macro: {train_f1:.4f}, F1-Micro: {train_f1_micro:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Time: {test_time:.2f}s")
        print(f"       Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        print(f"       F1-Macro: {test_f1:.4f}, F1-Micro: {test_f1_micro:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s, Learning Rate: {current_lr:.6f}")
        print(f"Best Test F1: {best_test_f1:.4f}")
        print("-" * 80)
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, final_model_path)
    
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Best test F1: {best_test_f1:.4f}")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    
    # Load best model for later evaluation if needed
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history_df, best_test_f1

def evaluate_model(model, test_loader, device=None, class_names=CLASS_NAMES):
    """
    Evaluate model on test dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            if isinstance(outputs, (tuple, list)):
                logits = outputs[-1]
            else:
                logits = outputs
                
            probabilities = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_per_class': {class_names[i]: precision_per_class[i] for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: recall_per_class[i] for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: f1_per_class[i] for i in range(len(class_names))},
        'confusion_matrix': cm.tolist(),
    }
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (Macro): {precision_macro:.4f}')
    print(f'Recall (Macro): {recall_macro:.4f}')
    print(f'F1 Score (Macro): {f1_macro:.4f}')
    print(f'F1 Score (Micro): {f1_micro:.4f}')
    
    return results, all_preds, all_labels, all_probs

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_training_history(history_df, save_path=None):
    """Plot training history"""
    metrics_df = pd.DataFrame(history_df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    if 'train_loss' in metrics_df.columns:
        ax1.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
    if 'test_loss' in metrics_df.columns:
        ax1.plot(metrics_df['epoch'], metrics_df['test_loss'], label='Test Loss')
    if 'val_loss' in metrics_df.columns:
        ax1.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    if 'train_acc' in metrics_df.columns:
        ax2.plot(metrics_df['epoch'], metrics_df['train_acc'], label='Train Accuracy')
    if 'test_acc' in metrics_df.columns:
        ax2.plot(metrics_df['epoch'], metrics_df['test_acc'], label='Test Accuracy')
    if 'train_f1' in metrics_df.columns:
        ax2.plot(metrics_df['epoch'], metrics_df['train_f1'], label='Train F1', linestyle='--')
    if 'test_f1' in metrics_df.columns:
        ax2.plot(metrics_df['epoch'], metrics_df['test_f1'], label='Test F1', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metrics')
    ax2.set_title('Training and Test Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    plt.close()

def visualize_predictions(model, test_loader, device=None, class_names=CLASS_NAMES, num_samples=5, save_path=None):
    """Visualize some sample predictions"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    with torch.no_grad():
        outputs = model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    plt.figure(figsize=(12, num_samples*3))
    for i in range(min(num_samples, len(images))):
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        pred_idx = preds[i].item()
        true_idx = labels[i].item()
        pred_class = class_names[pred_idx]
        true_class = class_names[true_idx]
        pred_prob = probabilities[i][pred_idx].item()
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img)
        title_color = 'green' if pred_idx == true_idx else 'red'
        plt.title(f'True: {true_class}, Pred: {pred_class} ({pred_prob:.2f})', color=title_color)
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    plt.show()

def apply_grad_cam(model, image_tensor, class_idx, device=None):
    """Apply Grad-CAM visualization technique"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    if hasattr(model, 'layer4'):
        target_layer = model.layer4
    elif hasattr(model, 'blocks') and len(model.blocks) > 0:
        target_layer = model.blocks[-1].norm1
    else:
        print("Could not determine target layer for Grad-CAM")
        return None
    image_tensor.requires_grad = True
    outputs = model(image_tensor)
    score = outputs[0, class_idx]
    model.zero_grad()
    score.backward()
    gradients = image_tensor.grad
    heatmap = torch.abs(gradients[0]).mean(dim=0).cpu().detach().numpy()
    return heatmap

def visualize_feature_maps(model, image_tensor, device=None, save_path=None):
    """Visualize feature maps from early, middle, and late layers"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    features = {}
    def hook_fn(module, input, output):
        features[module] = output
    if hasattr(model, 'layer1'):
        model.layer1.register_forward_hook(hook_fn)
        model.layer2.register_forward_hook(hook_fn)
        model.layer4.register_forward_hook(hook_fn)
    elif hasattr(model, 'blocks') and len(model.blocks) > 0:
        model.blocks[0].register_forward_hook(hook_fn)
        model.blocks[len(model.blocks)//2].register_forward_hook(hook_fn)
        model.blocks[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(image_tensor)
    plt.figure(figsize=(15, 5))
    i = 1
    for layer_name, feature in features.items():
        if len(feature.shape) == 4:
            num_channels = min(4, feature.shape[1])
            for c in range(num_channels):
                plt.subplot(len(features), num_channels, i)
                plt.imshow(feature[0, c].cpu().numpy(), cmap='viridis')
                plt.title(f"{layer_name} - Ch {c}")
                plt.axis('off')
                i += 1
        elif len(feature.shape) == 3:
            n_tokens = feature.shape[1]
            d_embed = feature.shape[2]
            size = int(np.sqrt(n_tokens))
            if size * size != n_tokens:
                continue
            num_dims = min(4, d_embed)
            for d in range(num_dims):
                plt.subplot(len(features), num_dims, i)
                plt.imshow(feature[0, :, d].cpu().numpy().reshape(size, size), cmap='viridis')
                plt.title(f"{layer_name} - Dim {d}")
                plt.axis('off')
                i += 1
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map visualization saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # Example usage of the training pipeline can be placed here if needed.
    # This code is intended to be imported and used by your main training script.
    pass
