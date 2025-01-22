import yaml
from pathlib import Path
from typing import Dict, Any
import torch
from src.model_training import ModelTrainer
from src.vectorize_ground_truth import GroundTruthVectorizer
from src.vector_viz import VectorVisualizer
from src.model_eval import ModelEvaluator

class PipelineConfig:
    def __init__(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_training_params(self) -> Dict[str, Any]:
        return self.config.get('training', {})
    
    def get_vectorization_params(self) -> Dict[str, Any]:
        return self.config.get('vectorization', {})
    
    def get_visualization_params(self) -> Dict[str, Any]:
        return self.config.get('visualization', {})
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        return self.config.get('evaluation', {})

class Pipeline:
    def __init__(self, config_path: str):
        """Initialize the pipeline with configuration"""
        self.config = PipelineConfig(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def run_training(self) -> str:
        """Run model training phase"""
        print("\n=== Starting Model Training ===")
        params = self.config.get_training_params()
        
        trainer = ModelTrainer(
            data_dir=params['data_dir'],
            model_save_dir=params['model_save_dir'],
            batch_size=params.get('batch_size', 8),
            num_workers=params.get('num_workers', 8)
        )
        
        trainer.train(
            num_epochs=params.get('num_epochs', 40),
            learning_rate=params.get('learning_rate', 0.0001),
            patience=params.get('patience', 3)
        )
        
        return str(Path(params['model_save_dir']) / "resnet34_model.pth")

    def run_vectorization(self, model_path: str) -> str:
        """Run vectorization phase"""
        print("\n=== Starting Vectorization ===")
        params = self.config.get_vectorization_params()
        
        vectorizer = GroundTruthVectorizer(
            model_path=model_path,
            input_dir=params['input_dir'],
            output_path=params['output_path'],
            batch_size=params.get('batch_size', 500),
            save_interval=params.get('save_interval', 3000)
        )
        
        vectorizer.vectorize_images()
        stats = vectorizer.analyze_vectors()
        print(f"Vectorization complete! Generated {stats['total_vectors']} vectors")
        
        return params['output_path']

    def run_visualization(self, vector_path: str):
        """Run visualization phase"""
        print("\n=== Starting Visualization ===")
        params = self.config.get_visualization_params()
        
        with open(vector_path, 'rb') as f:
            vector_dict = pickle.load(f)
        
        visualizer = VectorVisualizer(vector_dict)
        visualizer.visualize_3d(
            selected_classes=params.get('selected_classes'),
            output_path=params['output_path'],
            width=params.get('width', 700),
            marker_size=params.get('marker_size', 6),
            marker_opacity=params.get('marker_opacity', 0.8)
        )

    def run_evaluation(self, model_path: str, vector_path: str):
        """Run evaluation phase"""
        print("\n=== Starting Evaluation ===")
        params = self.config.get_evaluation_params()
        
        evaluator = ModelEvaluator(
            model_path=model_path,
            vector_path=vector_path,
            test_dirs=params['test_dirs'],
            reference_dirs=params['reference_dirs']
        )
        
        metrics = evaluator.evaluate_all()
        print("\nFinal Evaluation Metrics:")
        print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        
    def run_pipeline(self, skip_training: bool = False):
        """Run the complete pipeline"""
        if skip_training:
            model_path = self.config.get_training_params()['model_save_dir']
        else:
            model_path = self.run_training()
            
        vector_path = self.run_vectorization(model_path)
        self.run_visualization(vector_path)
        self.run_evaluation(model_path, vector_path)

def main():
    # configuration file
    config_example = """
    training:
      data_dir: "data/training_images"
      model_save_dir: "models"
      batch_size: 8
      num_workers: 8
      num_epochs: 40
      learning_rate: 0.0001
      patience: 3

    vectorization:
      input_dir: "data/ground_truth_images"
      output_path: "data/vectors/vector_dict.pkl"
      batch_size: 500
      save_interval: 3000

    visualization:
      selected_classes: 
        - "base1-7"
        - "col1-27"
        - "cel25-6"
      output_path: "visualizations/3d_scatter.html"
      width: 700
      marker_size: 6
      marker_opacity: 0.8

    evaluation:
      test_dirs:
        - "data/test_sets/normal"
        - "data/test_sets/holo"
        - "data/test_sets/full_art"
      reference_dirs:
        - "data/ground_truth_images"
        - "data/reference_images"
    """
    
    # Save example config
    config_path = "pipeline_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_example)
    
    # Run pipeline
    pipeline = Pipeline(config_path)
    pipeline.run_pipeline(skip_training=False)

if __name__ == "__main__":
    main() 