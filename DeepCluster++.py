import os
import csv
import warnings
import argparse
import math
warnings.filterwarnings("ignore")
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms
import torch.multiprocessing as mp

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

import Auto_encoder

class Config:
    def __init__(self, 
                 input_folder: str,
                 output_folder: str,
                 feature_dir: str = 'features',
                 cluster_dir: str = 'clusters',
                 plot_dir: str = 'plots',
                 samples_per_cluster: int = 400,  # New parameter
                 min_clusters: int = 3,           # Minimum number of clusters
                 max_clusters: int = 50,          # Maximum number of clusters
                 batch_size: int = 256,
                 sample_percentage: float = 0.20,
                 num_distance_groups: int = 5):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.feature_dir = os.path.join(output_folder, feature_dir)
        self.cluster_dir = os.path.join(output_folder, cluster_dir)
        self.plot_dir = os.path.join(output_folder, plot_dir)
        self.samples_per_cluster = samples_per_cluster
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.batch_size = batch_size  
        self.sample_percentage = sample_percentage
        self.num_distance_groups = num_distance_groups
        
        # Create directories
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.cluster_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

class TileDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = []
        
        # Collect all image files
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))
        return img, img_path


def visualize_clusters(features_2d, clusters, file_paths, output_prefix: str, palette):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Plot with legend
    plt.figure(figsize=(12, 8))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[palette[cluster]],
            label=f'Cluster {cluster}',
            alpha=0.6
        )
        
        # Draw convex hull
        if np.sum(mask) >= 3:  # Need at least 3 points for a hull
            try:
                hull = ConvexHull(features_2d[mask])
                for simplex in hull.simplices:
                    plt.plot(features_2d[mask][simplex, 0], 
                            features_2d[mask][simplex, 1], 
                            c=palette[cluster])
            except Exception as e:
                print(f"Could not create convex hull for cluster {cluster}: {e}")
    
    plt.title("t-SNE Visualization with K-Means Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_with_legend.png", dpi=400, bbox_inches='tight') 
    plt.close()
    
    # Plot with cluster numbers
    plt.figure(figsize=(16, 11))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[palette[cluster]],
            alpha=0.6
        )
        
        # Add cluster number in center
        if np.sum(mask) > 0:  # Only if cluster has points
            center = features_2d[mask].mean(axis=0)
            plt.text(center[0], center[1], str(cluster), 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontweight='bold')
        
        # Draw convex hull
        if np.sum(mask) >= 3:
            try:
                hull = ConvexHull(features_2d[mask])
                for simplex in hull.simplices:
                    plt.plot(features_2d[mask][simplex, 0], 
                            features_2d[mask][simplex, 1], 
                            c=palette[cluster])
            except Exception as e:
                print(f"Could not create convex hull for cluster {cluster}: {e}")
    
    plt.title("t-SNE Visualization with Cluster Numbers")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_with_numbers.png", dpi=400, bbox_inches='tight') 
    plt.close()

def get_subfolders(folder_path):
    """Get a list of immediate subfolders in the given folder path."""
    return [os.path.join(folder_path, name) for name in os.listdir(folder_path) 
            if os.path.isdir(os.path.join(folder_path, name))]

def count_images_in_folder(folder_path):
    """Count the number of image files in a folder and its subfolders."""
    count = 0
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                count += 1
    
    return count

def calculate_optimal_clusters(num_samples, samples_per_cluster, min_clusters, max_clusters):
    """Calculate the optimal number of clusters based on sample count and target samples per cluster."""
    # Calculate raw number of clusters
    if num_samples == 0:
        return min_clusters
    
    # Calculate number of clusters based on samples per cluster
    n_clusters = max(1, num_samples // samples_per_cluster)
    
    # Apply constraints
    n_clusters = max(min_clusters, min(max_clusters, n_clusters))
    
    # Ensure we don't have more clusters than samples
    n_clusters = min(n_clusters, num_samples - 1) if num_samples > 1 else 1
    
    return n_clusters

def extract_features(subfolder_path: str, subfolder_name: str, config: Config, device: torch.device, model: torch.nn.Module):
    print(f"\nExtracting features for subfolder: {subfolder_name}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = TileDataset(folder_path=subfolder_path, transform=transform)
    
    if len(dataset) == 0:
        print(f"No images found in {subfolder_path}. Skipping.")
        return None, 0
        
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create subfolder-specific feature directory
    subfolder_feature_dir = os.path.join(config.feature_dir, subfolder_name)
    os.makedirs(subfolder_feature_dir, exist_ok=True)
    
    feature_file = os.path.join(subfolder_feature_dir, "features.csv")
    
    feature_dim = 512
    
    with open(feature_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"Feature_{i}" for i in range(1, feature_dim + 1)] + ["File_Path"]
        writer.writerow(header)
        
        with torch.no_grad():
            for batch, file_paths in tqdm(data_loader, desc=f"Extracting features for {subfolder_name}"):
                batch = batch.to(device)
                latent = model.encoder(batch)
                latent_gap = F.adaptive_avg_pool2d(latent, (1, 1))
                latent_flattened = latent_gap.view(latent_gap.size(0), -1).cpu().numpy()
                
                for feature_vector, path in zip(latent_flattened, file_paths):
                    writer.writerow(list(feature_vector) + [path])
    
    print(f"Features extracted and saved to {feature_file}")
    return feature_file, len(dataset)

def process_clusters(subfolder_name: str, feature_file: str, n_samples: int, config: Config):
    print(f"\nClustering images for subfolder: {subfolder_name}")
    
    data = pd.read_csv(feature_file)
    features = data.drop('File_Path', axis=1).values
    file_paths = data['File_Path'].values
    
    # Calculate optimal number of clusters based on sample count
    n_clusters = calculate_optimal_clusters(
        num_samples=n_samples,
        samples_per_cluster=config.samples_per_cluster,
        min_clusters=config.min_clusters,
        max_clusters=config.max_clusters
    )
    
    print(f"Using {n_clusters} clusters for {n_samples} samples (target: {config.samples_per_cluster} samples/cluster)")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print("Applying t-SNE...")
    # Calculate safe perplexity value (must be less than n_samples)
    perplexity = min(30, max(5, min(n_samples - 1, n_samples // 10)))
    
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features_scaled)
    except ValueError as e:
        print(f"t-SNE error: {e}. Using PCA instead.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
    
    print(f"Applying K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_2d)
    
    palette = sns.color_palette("husl", n_colors=n_clusters)
    
    # Create subfolder-specific plot directory
    plot_dir = os.path.join(config.plot_dir, subfolder_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    visualize_clusters(features_2d, clusters, file_paths,
                      os.path.join(plot_dir, "tsne"), palette)
    
    # Save cluster assignments
    feature_dir = os.path.dirname(feature_file)
    assignments = pd.DataFrame({
        'File_Path': file_paths,
        'Cluster': clusters
    })
    assignments.to_csv(os.path.join(feature_dir, "cluster_assignments.csv"),
                      index=False)
    
    # Create subfolder-specific cluster directory
    cluster_base = os.path.join(config.cluster_dir, subfolder_name)
    os.makedirs(cluster_base, exist_ok=True)
    
    for cluster in tqdm(range(n_clusters), desc=f"Creating cluster folders for {subfolder_name}"):
        cluster_dir = os.path.join(cluster_base, f'Cluster_{cluster}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        cluster_files = file_paths[clusters == cluster]
        for file_path in cluster_files:
            try:
                img = Image.open(file_path)
                img.save(os.path.join(cluster_dir, os.path.basename(file_path)))
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
    
    # Print the actual samples per cluster distribution
    cluster_counts = np.bincount(clusters)
    avg_samples_per_cluster = n_samples / n_clusters if n_clusters > 0 else 0
    print(f"Actual distribution: Avg {avg_samples_per_cluster:.1f} samples/cluster")
    print(f"Min: {cluster_counts.min()} samples, Max: {cluster_counts.max()} samples")
    
    return assignments, n_clusters

def sample_clustered_images(subfolder_name: str, n_clusters: int, config: Config):  
    print(f"\nSelecting representative samples for subfolder: {subfolder_name}")
    
    # Read cluster assignments
    subfolder_feature_dir = os.path.join(config.feature_dir, subfolder_name)
    assignment_file = os.path.join(subfolder_feature_dir, "cluster_assignments.csv")
    feature_file = os.path.join(subfolder_feature_dir, "features.csv")
    
    if not os.path.exists(assignment_file) or not os.path.exists(feature_file):
        print(f"Missing required files for sample selection in {subfolder_name}")
        return
    
    # Load assignments and features
    assignments = pd.read_csv(assignment_file)
    features_df = pd.read_csv(feature_file)
    
    # Prepare output directory
    samples_base = os.path.join(config.output_folder, 'samples', subfolder_name)
    os.makedirs(samples_base, exist_ok=True)
    
    # Process each cluster
    cluster_ids = assignments['Cluster'].unique()
    
    for cluster_id in tqdm(cluster_ids, desc=f"Processing clusters for sampling in {subfolder_name}"):
        # Get files and features for this cluster
        cluster_mask = assignments['Cluster'] == cluster_id
        cluster_files = assignments.loc[cluster_mask, 'File_Path'].values
        
        # Skip if cluster is empty
        if len(cluster_files) == 0:
            continue
        
        # Create output directory for this cluster
        cluster_sample_dir = os.path.join(samples_base, f'Cluster_{cluster_id}')
        os.makedirs(cluster_sample_dir, exist_ok=True)
        
        # Match files with their features
        cluster_features = []
        cluster_file_paths = []
        
        for file_path in cluster_files:
            feature_row = features_df[features_df['File_Path'] == file_path].drop('File_Path', axis=1).values
            if len(feature_row) > 0:
                cluster_features.append(feature_row[0])
                cluster_file_paths.append(file_path)
        
        if not cluster_features:
            continue
        
        # Convert to numpy array for calculations
        cluster_features = np.array(cluster_features)
        
        # Calculate centroid
        centroid = np.mean(cluster_features, axis=0)
        
        # Calculate distances from centroid to each sample
        distances = []
        for i, feature in enumerate(cluster_features):
            # Use Euclidean distance
            dist = np.linalg.norm(feature - centroid)
            distances.append((cluster_file_paths[i], dist))
        
        # Normalize distances to [0, 1]
        if len(distances) > 1:  # Only normalize if more than one sample
            max_dist = max(d[1] for d in distances)
            min_dist = min(d[1] for d in distances)
            
            # Handle case where all points are the same distance
            if max_dist == min_dist:
                normalized_distances = [(file_path, 0.0) for file_path, _ in distances]
            else:
                normalized_distances = [
                    (file_path, (dist - min_dist) / (max_dist - min_dist))
                    for file_path, dist in distances
                ]
        else:
            # If only one sample, its normalized distance is 0
            normalized_distances = [(distances[0][0], 0.0)]
        
        # Implementation of equal-frequency binning
        sorted_distances = sorted(normalized_distances, key=lambda x: x[1])
        total_samples = len(sorted_distances)
        
        # Calculate base samples per group and remainder
        samples_per_group = total_samples // config.num_distance_groups
        remainder = total_samples % config.num_distance_groups
        
        # Initialize empty groups
        distance_groups = [[] for _ in range(config.num_distance_groups)]
        current_idx = 0
        
        # Distribute samples using equal-frequency binning
        for group_idx in range(config.num_distance_groups):
            # Add extra sample to earlier groups if there's a remainder
            extra = 1 if group_idx < remainder else 0
            group_size = samples_per_group + extra
            
            # Last group gets any remaining samples
            if group_idx == config.num_distance_groups - 1:
                group_size = total_samples - current_idx
            
            # Add samples to this group
            for i in range(group_size):
                if current_idx < total_samples:
                    file_path, _ = sorted_distances[current_idx]
                    distance_groups[group_idx].append(file_path)
                    current_idx += 1
        
        # Sample from each distance group
        for group_idx, group_files in enumerate(distance_groups):
            if not group_files:
                continue
                
            # Calculate number of samples to select
            num_samples = max(1, int(len(group_files) * config.sample_percentage))
            
            # Randomly select samples
            selected_files = np.random.choice(
                group_files, 
                size=min(num_samples, len(group_files)), 
                replace=False
            )
            
            # Create group directory
            group_dir = os.path.join(cluster_sample_dir, f'Group_{group_idx}')
            os.makedirs(group_dir, exist_ok=True)
            
            # Copy selected files to output directory
            for file_path in selected_files:
                try:
                    img = Image.open(file_path)
                    img.save(os.path.join(group_dir, os.path.basename(file_path)))
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
    
    print(f"Sample selection completed for {subfolder_name}")

def process_subfolder(subfolder_path: str, config: Config, device: torch.device, model: torch.nn.Module):
    subfolder_name = os.path.basename(subfolder_path)
    
    try:
        # First count the number of images to estimate the number of clusters
        n_images = count_images_in_folder(subfolder_path)
        
        if n_images == 0:
            print(f"No images found in subfolder: {subfolder_name}. Skipping.")
            print(f"----------------------------------------------------\n")
            return False
            
        # Pre-calculate and display the expected number of clusters
        n_clusters = calculate_optimal_clusters(
            num_samples=n_images,
            samples_per_cluster=config.samples_per_cluster,
            min_clusters=config.min_clusters,
            max_clusters=config.max_clusters
        )
        
        print(f"\n--- Processing subfolder: {subfolder_name} ---")
        print(f"Found {n_images} images, targeting {n_clusters} clusters")
        print(f"Average of ~{n_images / n_clusters:.1f} samples per cluster expected")
        
        # Extract features
        feature_file, actual_samples = extract_features(subfolder_path, subfolder_name, config, device, model)
        
        if feature_file is not None:
            # Process clusters
            assignments, n_clusters = process_clusters(subfolder_name, feature_file, actual_samples, config)
            
            # Sample representative images
            sample_clustered_images(subfolder_name, n_clusters, config)
            
            print(f"\nProcessing completed for subfolder: {subfolder_name}")
            print(f"\n---------------------------------------------------\n")
            return True
        else:
            print(f"Feature extraction failed for subfolder: {subfolder_name}. Skipping.")
            return False
    except Exception as e:
        print(f"\nError processing subfolder {subfolder_name}: {str(e)}")
        return False

def process_all_subfolders(config: Config, device: torch.device, model_path: str):
    try:
        # Load model
        model = Auto_encoder.AutoEncoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Get all subfolders from the input folder
        subfolders = get_subfolders(config.input_folder)
        
        if not subfolders:
            print(f"No subfolders found in {config.input_folder}")
            return
        
        print(f"Found {len(subfolders)} subfolders to process:")
        for subfolder in subfolders:
            print(f"  - {os.path.basename(subfolder)}")
        
        # Process each subfolder
        successful = 0
        failed = 0
        
        for subfolder_path in subfolders:
            success = process_subfolder(subfolder_path, config, device, model)
            if success:
                successful += 1
            else:
                failed += 1
        
        print("\nAll processing completed!")
        print(f"Successfully processed: {successful} subfolders")
        if failed > 0:
            print(f"Failed to process: {failed} subfolders")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Cluster and sample WSI images from subfolders')
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Path to the input folder containing subfolders with images')
    parser.add_argument('--output', required=True, help='Path to the output folder')
    
    # Optional arguments
    parser.add_argument('--samples-per-cluster', type=int, default=400, 
                        help='Target number of samples per cluster (default: 400)')
    parser.add_argument('--min-clusters', type=int, default=3,
                        help='Minimum number of clusters regardless of samples (default: 3)')
    parser.add_argument('--max-clusters', type=int, default=50,
                        help='Maximum number of clusters regardless of samples (default: 50)')
    parser.add_argument('--sample-percentage', type=float, default=0.20, 
                        help='Percentage of images to sample from each group (default: 0.20)')
    parser.add_argument('--distance-groups', type=int, default=5,
                        help='Number of distance groups for sampling (default: 5)')
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='Batch size for feature extraction (default: 256)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (auto, cpu, cuda) (default: auto)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use if CUDA is selected (default: 0)')
    parser.add_argument('--model', type=str, 
                        default="AE_CRC.pth",
                        help='Path to the model file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create config object
    config = Config(
        input_folder=args.input,
        output_folder=args.output,
        samples_per_cluster=args.samples_per_cluster,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        batch_size=args.batch_size,
        sample_percentage=args.sample_percentage,
        num_distance_groups=args.distance_groups
    )
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:  # args.device == 'cuda'
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Using CPU instead.")
            device = torch.device('cpu')
        else:
            if args.gpu_id >= torch.cuda.device_count():
                print(f"GPU {args.gpu_id} requested but not available. Using GPU 0 instead.")
                device = torch.device('cuda:0')
            else:
                device = torch.device(f'cuda:{args.gpu_id}')
    
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available() and 'cuda' in str(device):
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    
    # Process the subfolders
    process_all_subfolders(config, device, args.model)

if __name__ == "__main__":
    main()