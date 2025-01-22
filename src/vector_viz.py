"""# visualizing the vectors in a 3D space"""

import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

class VectorVisualizer:
    def __init__(self, vector_dict: Dict[str, List[float]]):
        """
        Initialize the vector visualizer
        
        Args:
            vector_dict: Dictionary containing vector embeddings
        """
        self.vector_dict = vector_dict
        self.embeddings = []
        self.colors = []
        self._process_vectors()

    def _process_vectors(self):
        """Extract embeddings and class names from the vector dictionary"""
        for key, value in self.vector_dict.items():
            parts = key.split('_')
            classname = parts[0]  # Get classname from the key
            self.embeddings.append(value)
            self.colors.append(classname)

    def visualize_3d(
        self,
        selected_classes: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        width: int = 700,
        marker_size: int = 6,
        marker_opacity: float = 0.8
    ):
        """
        Create a 3D visualization of the vector embeddings
        
        Args:
            selected_classes: List of classes to visualize (if None, uses all classes)
            output_path: Path to save the HTML plot (if None, doesn't save)
            width: Width of the plot
            marker_size: Size of the markers in the plot
            marker_opacity: Opacity of the markers
        """
        # Filter embeddings for selected classes if specified
        if selected_classes is None:
            selected_classes = list(set(self.colors))

        # Filter the embeddings and colors for the selected classes
        selected_embeddings = [
            embedding for embedding, color in zip(self.embeddings, self.colors)
            if color in selected_classes
        ]
        selected_colors = [
            color for color in self.colors
            if color in selected_classes
        ]

        # Create DataFrame and apply PCA
        df = pd.DataFrame(selected_embeddings)
        df['color'] = selected_colors

        # Apply PCA
        pca = PCA(n_components=3)
        embeddings_pca = pca.fit_transform(df.iloc[:,:-1])

        # Separate embeddings by class
        class_embeddings = {
            classname: embeddings_pca[df['color'] == classname]
            for classname in selected_classes
        }

        # Create the 3D plot
        fig = go.Figure()
        
        # Add traces for each class
        for classname, class_embedding in class_embeddings.items():
            fig.add_trace(go.Scatter3d(
                x=class_embedding[:, 0],
                y=class_embedding[:, 1],
                z=class_embedding[:, 2],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    opacity=marker_opacity
                ),
                name=classname
            ))

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=width,
            margin=dict(r=20, b=10, l=10, t=10)
        )

        # Save the plot if output path is specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            print(f"Plot saved to {output_path}")

        return fig

    def get_explained_variance(self, n_components: int = 3) -> Dict[str, float]:
        """
        Get the explained variance ratios for PCA components
        
        Args:
            n_components: Number of PCA components to analyze
            
        Returns:
            Dict containing explained variance ratios
        """
        pca = PCA(n_components=n_components)
        pca.fit(self.embeddings)
        
        return {
            f'PC{i+1}': ratio
            for i, ratio in enumerate(pca.explained_variance_ratio_)
        }

def main():
    """Example usage of the VectorVisualizer"""
    import pickle
    
    # Load vector dictionary
    with open('data/vector_dict.pkl', 'rb') as f:
        vector_dict = pickle.load(f)
    
    # Initialize visualizer
    visualizer = VectorVisualizer(vector_dict)
    
    # Example classes to visualize
    selected_classes = [
        "base1-7", "col1-27", "cel25-6",
        "dp1-90", "dp2-105", "ecard2-98", "ecard1-136"
    ]
    
    # Create and save visualization
    fig = visualizer.visualize_3d(
        selected_classes=selected_classes,
        output_path="visualizations/3d_scatter_plot.html"
    )
    
    # Print explained variance
    variance_ratios = visualizer.get_explained_variance()
    print("\nExplained Variance Ratios:")
    for pc, ratio in variance_ratios.items():
        print(f"{pc}: {ratio:.4f}")

if __name__ == "__main__":
    main()