"""
Metrics calculation module for model evaluation
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Union


class MetricsCalculator:
    """
    Calculate various metrics for PPE detection model evaluation.
    """
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize MetricsCalculator.
        
        Args:
            num_classes: Number of PPE classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.tn = np.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        preds = predictions.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        for c in range(self.num_classes):
            tp = np.sum((preds == c) & (targets == c))
            fp = np.sum((preds == c) & (targets != c))
            fn = np.sum((preds != c) & (targets == c))
            tn = np.sum((preds != c) & (targets != c))
            
            self.tp[c] += tp
            self.fp[c] += fp
            self.fn[c] += fn
            self.tn[c] += tn
    
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        correct = np.sum(self.tp)
        total = np.sum(self.tp + self.fp + self.fn + self.tn)
        return correct / total if total > 0 else 0.0
    
    def precision(self, class_id: int = None) -> Union[float, np.ndarray]:
        """
        Calculate precision.
        
        Args:
            class_id: Class ID (None for all classes)
            
        Returns:
            Precision value(s)
        """
        if class_id is not None:
            denom = self.tp[class_id] + self.fp[class_id]
            return self.tp[class_id] / denom if denom > 0 else 0.0
        else:
            precisions = []
            for c in range(self.num_classes):
                denom = self.tp[c] + self.fp[c]
                precisions.append(self.tp[c] / denom if denom > 0 else 0.0)
            return np.array(precisions)
    
    def recall(self, class_id: int = None) -> Union[float, np.ndarray]:
        """
        Calculate recall (sensitivity).
        
        Args:
            class_id: Class ID (None for all classes)
            
        Returns:
            Recall value(s)
        """
        if class_id is not None:
            denom = self.tp[class_id] + self.fn[class_id]
            return self.tp[class_id] / denom if denom > 0 else 0.0
        else:
            recalls = []
            for c in range(self.num_classes):
                denom = self.tp[c] + self.fn[c]
                recalls.append(self.tp[c] / denom if denom > 0 else 0.0)
            return np.array(recalls)
    
    def f1_score(self, class_id: int = None) -> Union[float, np.ndarray]:
        """
        Calculate F1 score.
        
        Args:
            class_id: Class ID (None for all classes)
            
        Returns:
            F1 score(s)
        """
        if class_id is not None:
            p = self.precision(class_id)
            r = self.recall(class_id)
            return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        else:
            f1_scores = []
            for c in range(self.num_classes):
                p = self.precision(c)
                r = self.recall(c)
                f1_scores.append(2 * (p * r) / (p + r) if (p + r) > 0 else 0.0)
            return np.array(f1_scores)
    
    def specificity(self, class_id: int = None) -> Union[float, np.ndarray]:
        """
        Calculate specificity.
        
        Args:
            class_id: Class ID (None for all classes)
            
        Returns:
            Specificity value(s)
        """
        if class_id is not None:
            denom = self.tn[class_id] + self.fp[class_id]
            return self.tn[class_id] / denom if denom > 0 else 0.0
        else:
            specificities = []
            for c in range(self.num_classes):
                denom = self.tn[c] + self.fp[c]
                specificities.append(self.tn[c] / denom if denom > 0 else 0.0)
            return np.array(specificities)
    
    def get_metrics_dict(self) -> Dict:
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary with all metric values
        """
        return {
            'accuracy': self.accuracy(),
            'precision_per_class': self.precision(),
            'recall_per_class': self.recall(),
            'f1_per_class': self.f1_score(),
            'specificity_per_class': self.specificity(),
            'macro_f1': np.mean(self.f1_score()),
            'macro_precision': np.mean(self.precision()),
            'macro_recall': np.mean(self.recall()),
        }


class BBoxMetrics:
    """
    Calculate metrics for bounding box predictions (IoU, mAP, etc.)
    """
    
    @staticmethod
    def iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: First box [x_min, y_min, x_max, y_max]
            box2: Second box [x_min, y_min, x_max, y_max]
            
        Returns:
            IoU value
        """
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2
        
        # Intersection
        x_min_inter = max(x_min1, x_min2)
        y_min_inter = max(y_min1, y_min2)
        x_max_inter = min(x_max1, x_max2)
        y_max_inter = min(y_max1, y_max2)
        
        if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
            return 0.0
        
        inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
        
        # Union
        box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
        box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def mean_iou(pred_boxes: List, gt_boxes: List, iou_threshold: float = 0.5) -> float:
        """
        Calculate mean IoU across all predictions.
        
        Args:
            pred_boxes: List of predicted boxes
            gt_boxes: List of ground truth boxes
            iou_threshold: IoU threshold for matching
            
        Returns:
            Mean IoU value
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return 0.0
        
        ious = []
        for pred_box in pred_boxes:
            max_iou = max([BBoxMetrics.iou(pred_box, gt_box) for gt_box in gt_boxes])
            if max_iou >= iou_threshold:
                ious.append(max_iou)
        
        return np.mean(ious) if len(ious) > 0 else 0.0
