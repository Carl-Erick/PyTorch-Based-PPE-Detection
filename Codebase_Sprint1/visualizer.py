"""
Visualization module for PPE Detection
Displays images and creates analysis graphs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List
from pathlib import Path


class PPEVisualizer:
    """Visualize PPE detection images and statistics"""
    
    @staticmethod
    def show_image(image: np.ndarray, title: str = "PPE Image"):
        """
        Display a single image
        
        Args:
            image: Image as numpy array
            title: Title for the plot
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def show_grid(images: List[np.ndarray], titles: List[str] = None, grid_size: tuple = (2, 3)):
        """
        Display multiple images in a grid
        
        Args:
            images: List of images
            titles: List of titles for each image
            grid_size: Grid dimensions (rows, cols)
        """
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, image in enumerate(images[:len(axes)]):
            if idx < len(axes):
                axes[idx].imshow(image)
                if titles and idx < len(titles):
                    axes[idx].set_title(titles[idx])
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(images), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_image_statistics(data: dict):
        """
        Plot image statistics
        
        Args:
            data: Dictionary with statistics
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total images bar chart
        total = data.get('total_images', 0)
        axes[0].bar(['Total Images'], [total], color='skyblue', width=0.5)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Total PPE Dataset Images', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, total * 1.2)
        for i, v in enumerate([total]):
            axes[0].text(i, v + total*0.05, str(v), ha='center', fontsize=12, fontweight='bold')
        
        # Example distribution (simulated)
        ppe_classes = ['Helmet', 'Vest', 'No PPE']
        counts = [int(total * 0.4), int(total * 0.35), int(total * 0.25)]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        axes[1].bar(ppe_classes, counts, color=colors)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('PPE Classes Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, max(counts) * 1.2)
        for i, v in enumerate(counts):
            axes[1].text(i, v + max(counts)*0.05, str(v), ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_image_sizes(images: List[np.ndarray]):
        """
        Plot image size statistics
        
        Args:
            images: List of images
        """
        sizes = [img.shape[:2] for img in images]
        heights = [s[0] for s in sizes]
        widths = [s[1] for s in sizes]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Height distribution
        axes[0].hist(heights, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Height (pixels)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Image Height Distribution', fontsize=14, fontweight='bold')
        axes[0].axvline(np.mean(heights), color='red', linestyle='--', linewidth=2, label=f'Mean: {int(np.mean(heights))}')
        axes[0].legend()
        
        # Width distribution
        axes[1].hist(widths, bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Width (pixels)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Image Width Distribution', fontsize=14, fontweight='bold')
        axes[1].axvline(np.mean(widths), color='blue', linestyle='--', linewidth=2, label=f'Mean: {int(np.mean(widths))}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_dataset_summary(data: dict):
        """
        Plot summary statistics
        
        Args:
            data: Dictionary with dataset information
        """
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2)
        
        # Title
        fig.suptitle('PPE Detection Dataset Summary', fontsize=16, fontweight='bold', y=0.98)
        
        # Total images
        ax1 = fig.add_subplot(gs[0, 0])
        total = data.get('total_images', 0)
        ax1.text(0.5, 0.5, str(total), ha='center', va='center', fontsize=48, fontweight='bold', color='blue')
        ax1.text(0.5, 0.1, 'Total Images', ha='center', va='center', fontsize=14)
        ax1.axis('off')
        
        # PPE Classes
        ax2 = fig.add_subplot(gs[0, 1])
        classes = ['Helmet', 'Vest', 'No PPE']
        ax2.text(0.5, 0.7, 'PPE Classes:', ha='center', va='top', fontsize=12, fontweight='bold')
        for i, cls in enumerate(classes):
            ax2.text(0.5, 0.5 - i*0.15, f'• {cls}', ha='center', va='top', fontsize=11)
        ax2.axis('off')
        
        # Dataset Info
        ax3 = fig.add_subplot(gs[1, :])
        info_text = f"""
        Dataset Information:
        • Total Images: {total}
        • Format: YOLO (.txt annotations)
        • Image Formats: JPEG, PNG
        • Status: Ready for analysis
        """
        ax3.text(0.1, 0.5, info_text, ha='left', va='center', fontsize=11, 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
