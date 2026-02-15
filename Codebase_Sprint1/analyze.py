"""
Main script to display PPE images and analysis
Saves plots to files instead of displaying them
"""

import sys
sys.path.append('/workspaces/PyTorch-Based-PPE-Detection/Codebase_Sprint1')

from data_loader import PPEDataLoader
from visualizer import PPEVisualizer
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def main():
    """Main execution"""
    
    # Set up paths - use relative path to Dataset_Sample in parent directory
    dataset_dir = Path('../Dataset_Sample')
    images_dir = dataset_dir / 'images'
    output_dir = Path('./output_analysis')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PPE Detection - Image Analysis")
    print("=" * 60)
    
    # Create sample images if they don't exist
    if not images_dir.exists():
        print("\n⚠️  Dataset not found. Generating sample images...")
        import generate_dataset
        # Generate in the correct location
        generate_dataset.create_sample_dataset(str(dataset_dir), num_samples=20)
        print("✓ Sample dataset created!")
    
    # Load data
    print("\n📁 Loading images from:", images_dir)
    loader = PPEDataLoader(str(images_dir))
    stats = loader.get_image_stats()
    
    print(f"✓ Loaded {stats['total_images']} images\n")
    
    # Display dataset summary
    print("Generating dataset summary...")
    PPEVisualizer.plot_dataset_summary(stats)
    plt.savefig(output_dir / 'dataset_summary.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: output_analysis/dataset_summary.png")
    
    # Load images
    images = loader.get_all_images()
    
    if len(images) > 0:
        # Display image grid
        print("\nGenerating image grid...")
        titles = [f"Image {i+1}" for i in range(len(images))]
        PPEVisualizer.show_grid(images[:6], titles[:6], grid_size=(2, 3))
        plt.savefig(output_dir / 'image_grid.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: output_analysis/image_grid.png")
        
        # Display statistics
        print("\nGenerating image statistics...")
        PPEVisualizer.plot_image_statistics(stats)
        plt.savefig(output_dir / 'statistics.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: output_analysis/statistics.png")
        
        # Display size distribution
        print("\nGenerating size distribution...")
        PPEVisualizer.plot_image_sizes(images)
        plt.savefig(output_dir / 'image_sizes.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: output_analysis/image_sizes.png")
        
        # Display first image
        print("\nGenerating detailed image view...")
        PPEVisualizer.show_image(images[0], title="PPE Detection - Sample Image")
        plt.savefig(output_dir / 'sample_image.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: output_analysis/sample_image.png")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print(f"📊 All plots saved to: output_analysis/")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  • {f.name}")
    print("=" * 60)


if __name__ == '__main__':
    main()
