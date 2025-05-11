# Configuration for training with Pascal VOC dataset only

config = {
    # Dataset configuration
    'dataset': {
        'name': 'pascal_voc',
        'root_dir': 'data/VOCdevkit',
        'year': '2012',
        'train_split': 'train',
        'val_split': 'val',
        'batch_size': 8,
        'num_workers': 2,
    },
    
    # Model configuration
    'model': {
        'backbone': 'resnet50',  # Options: resnet18, resnet34, resnet50, resnet101
        'pretrained': True,
        'num_classes': 20,  # Pascal VOC has 20 classes
    },
    
    # Training configuration
    'training': {
        'epochs': 30,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_scheduler': 'step',  # Options: step, cosine
        'lr_step_size': 10,
        'lr_gamma': 0.1,
    },
    
    # Paths and logging
    'paths': {
        'output_dir': 'output/voc_only',
        'checkpoint_dir': 'checkpoints/voc_only',
        'log_dir': 'logs/voc_only',
    },
    
    # Evaluation configuration
    'evaluation': {
        'eval_frequency': 1,  # Evaluate every N epochs
    },
}