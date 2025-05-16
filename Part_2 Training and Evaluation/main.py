import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import time
import glob
import json
from tqdm import tqdm

from custom_models import get_custom_cnn, get_custom_histovit
from kimianet import get_kimianet

from config import (
    TRAIN_FOLDER_PATH, SAVE_DIR, NUM_CLASSES, 
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE
)
from dataset import get_data_loaders
from models import (
    get_resnet50, get_vit_base, 
    get_swin_base, get_convnext, get_efficientnet, get_deit
)
from foundation_models import (
    get_hipt_model, get_transpath_model, get_pathdino_model, get_vim4path_model,
    get_conch_model, get_virchow_model, get_uni_model,
    get_prov_gigapath_model
)
from trainer import (
    train_model, evaluate_model, plot_confusion_matrix, 
    plot_training_history, visualize_predictions, visualize_feature_maps
)


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_model(model_name, num_classes):
    """Get model by name"""
    print(f"Loading model: {model_name}")
    # Base models
    if model_name == 'resnet50':
        return get_resnet50(num_classes=num_classes)
    elif model_name == 'resnet101':
        return get_resnet101(num_classes=num_classes)
    elif model_name == 'vit_base':
        return get_vit_base(num_classes=num_classes)
    elif model_name == 'vit_small':
        return get_vit_small(num_classes=num_classes)
    elif model_name == 'vit_large':
        return get_vit_large(num_classes=num_classes)
    elif model_name == 'swin_base':
        return get_swin_base(num_classes=num_classes)
    elif model_name == 'swin_v2':
        return get_swin_v2(num_classes=num_classes)
    elif model_name == 'convnext':
        return get_convnext(num_classes=num_classes)
    elif model_name == 'efficientnet':
        return get_efficientnet(num_classes=num_classes)
    elif model_name == 'deit':
        return get_deit(num_classes=num_classes)
    elif model_name == 'mvit':
        return get_mvit(num_classes=num_classes)
    elif model_name == 'beit':
        return get_beit(num_classes=num_classes)
    # Foundation models
    elif model_name == 'hipt':
        return get_hipt_model(num_classes=num_classes)
    elif model_name == 'transpath':
        return get_transpath_model(num_classes=num_classes)
    elif model_name == 'pathdino':
        return get_pathdino_model(num_classes=num_classes)
    elif model_name == 'vim4path':
        return get_vim4path_model(num_classes=num_classes)
    elif model_name == 'conch':
        return get_conch_model(num_classes=num_classes)
    elif model_name == 'virchow':
        return get_virchow_model(num_classes=num_classes)
    elif model_name == 'uni':
        return get_uni_model(num_classes=num_classes)
    elif model_name == 'prov_gigapath':
        return get_prov_gigapath_model(num_classes=num_classes)
    # Custom models
    elif model_name == 'custom_cnn':
        return get_custom_cnn(num_classes=num_classes)
    elif model_name == 'custom_histovit':
        return get_custom_histovit(num_classes=num_classes)
    elif model_name == 'kimianet':
        return get_kimianet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate models for CRC classification')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture to use')
    parser.add_argument('--data_path', type=str, default=TRAIN_FOLDER_PATH, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Directory to save outputs')
    parser.add_argument('--no_train', action='store_true', help='Skip training phase')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load model from checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs for training')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Report GPU information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        folder_path=args.data_path, 
        batch_size=args.batch_size
    )
    print(f"Dataset loaded. Train: {len(train_loader.dataset)} samples, "
          f"Test: {len(test_loader.dataset)} samples")
    
    # Get model
    model = get_model(args.model, NUM_CLASSES)
    
# Save model parameter information
    params_info = {
        'model': args.model,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    # Create model directory if it doesn't exist
    model_dir = os.path.join(args.save_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save parameter info
    params_path = os.path.join(model_dir, f"{args.model}_params.json")
    with open(params_path, 'w') as f:
        json.dump(params_info, f, indent=4)
    
    # Multi-GPU support
    
    if args.multi_gpu and torch.cuda.device_count() > 1:    # args.multi_gpu and
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  #device_ids=list(range(torch.cuda.device_count()))
    model = model.to(device)
    
    print(f"Model {args.model} created with {NUM_CLASSES} output classes")
    print(f"Total parameters: {params_info['total_params']:,}")
    print(f"Trainable parameters: {params_info['trainable_params']:,}")
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {args.load_checkpoint}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.lr/100
    )
    
    # Train model if not skipping training
    if not args.no_train:
        model, history_df, best_test_f1 = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=args.epochs,
            device=device,
            save_dir=args.save_dir,
            model_name=args.model
        )
        
        # Plot training history
        history_plot_path = os.path.join(args.save_dir, f"{args.model}_history.png")
        plot_training_history(history_df, save_path=history_plot_path)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    results, all_preds, all_labels, all_probs = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    # Save evaluation results
    results_path = os.path.join(args.save_dir, f"{args.model}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.save_dir, f"{args.model}_confusion_matrix.png")
    plot_confusion_matrix(
        cm=np.array(results['confusion_matrix']),
        class_names=list(results['precision_per_class'].keys()),
        save_path=cm_path
    )
    
    # Generate visualizations if requested
    if args.visualize:
        # Visualize predictions
        pred_path = os.path.join(args.save_dir, f"{args.model}_predictions.png")
        visualize_predictions(
            model=model,
            test_loader=test_loader,
            device=device,
            save_path=pred_path
        )
        
        # Visualize feature maps for a sample image
        sample_images, _ = next(iter(test_loader))
        feature_path = os.path.join(args.save_dir, f"{args.model}_feature_maps.png")
        visualize_feature_maps(
            model=model,
            image_tensor=sample_images[0],
            device=device,
            save_path=feature_path
        )
    
    print("Done!")

if __name__ == "__main__":
    main()