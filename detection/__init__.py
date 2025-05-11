# Detection module initialization
from .ssd import SSDDetector
from .yolo import YOLODetector

__all__ = ['SSDDetector', 'YOLODetector']