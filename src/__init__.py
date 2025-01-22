"""
Mango Visual Search Training & Evaluation Pipeline
===============================================

A modular pipeline for training, vectorizing, visualizing, 
and evaluating image classification models.

Modules
-------
model_training
    Handles model training and validation
vectorize_ground_truth
    Generates vector embeddings for ground truth images
model_eval
    Evaluates model performance on test sets
run_pipeline
    Orchestrates the complete pipeline execution
"""

__version__ = '0.1.0'
__author__ = 'Riley Livingston'

from . import model_training
from . import vectorize_ground_truth
from . import model_eval
from . import run_pipeline

__all__ = [
    'model_training',
    'vectorize_ground_truth',
    'model_eval',
    'run_pipeline'
] 