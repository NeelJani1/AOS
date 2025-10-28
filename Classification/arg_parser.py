import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # Basic parameters
    parser.add_argument('--dataset', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet', 'tiny_imagenet'],
                        help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'resnet50', 'mobilenet_v2', 'vgg16_bn'],
                        help='model architecture')
    parser.add_argument('--data_dir', default='data', type=str,
                        help='data directory')
    parser.add_argument('--workers', default=8, type=int,
                        help='number of data loading workers')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--print_freq', default=50, type=int,
                        help='print frequency')
    
    # Unlearning parameters
    parser.add_argument('--unlearn', type=str, choices=['GA', 'FT', 'RL'],
                        help='unlearning method: GA (Gradient Ascent), FT (Fine-tuning), RL (Retain Learning)')
    parser.add_argument('--unlearn_epochs', default=10, type=int,
                        help='number of unlearning epochs')
    parser.add_argument('--unlearn_lr', default=1e-4, type=float,
                    help='unlearning learning rate')
    parser.add_argument('--num_indexes_to_replace', default=4500, type=int,
                        help='number of samples to mark for forgetting')
    parser.add_argument('--forget_percentage', default=0.1, type=float,
                        help='percentage of data to forget (0.1 to 1.0)')
    
    # Mask parameters
    parser.add_argument('--mask_path', type=str,
                        help='path to mask file for parameter freezing')
    parser.add_argument('--experiment_dir', type=str, default=None,
                    help='Specific experiment directory to use for masks and results')
    # In arg_parser.py, inside parse_args()
    parser.add_argument('--no_mask', action='store_true',
                        help='Run unlearning without loading or applying any mask.')
        # In arg_parser.py, inside parse_args()
    parser.add_argument('--kl_lambda', type=float, default=0.1,
                        help='Weight for KL divergence penalty in regularized finetuning.')
    # In arg_parser.py
    parser.add_argument('--ewc_lambda', type=float, default=0.0, help='Lambda for EWC regularization (FT --no_mask only)')
    
    # Otsu parameters
    parser.add_argument('--otsu_method', default='conservative', type=str,
                        choices=['basic', 'conservative', 'bounded', 'layer_aware'],
                        help='Otsu thresholding method')
    parser.add_argument('--otsu_conservatism', default=0.3, type=float,
                        help='conservatism factor for Otsu (higher = more conservative)')
    parser.add_argument('--otsu_min_retention', default=0.3, type=float,
                        help='minimum parameter retention rate')
    parser.add_argument('--otsu_max_retention', default=0.7, type=float,
                        help='maximum parameter retention rate')
    parser.add_argument('--test_otsu_methods', action='store_true',
                        help='test all Otsu methods and compare')
    
    # Other parameters
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')
    parser.add_argument('--print', action='store_true', default=False,
                        help='print model architecture')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume from checkpoint')
    parser.add_argument('--seed', default=2, type=int,
                        help='seed for initializing training')
    parser.add_argument('--save_dir', default='saved_models', type=str,
                        help='path to save models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to model checkpoint')
    
    # ImageNet specific
    parser.add_argument('--imagenet_arch', action='store_true',
                        help='use ImageNet architecture')
    parser.add_argument('--decreasing_lr', default='100,150', type=str,
                        help='decreasing strategy')
    parser.add_argument('--warmup', default=5, type=int,
                        help='warmup epochs')
    
    # AMP
    parser.add_argument('--amp', action='store_true',
                        help='use mixed precision training')
    
    # Class to replace (for unlearning)
    parser.add_argument('--class_to_replace', default=-1, type=int,
                        help='class to replace with unlearning, -1 for random')
    
    # Output organization parameters
    parser.add_argument('--experiment_name', default='adaptive_otsu', type=str,
                        help='name for this experiment run')
    parser.add_argument('--output_base_dir', default='results', type=str,
                        help='base directory for all results')
    parser.add_argument('--organize_by_method', action='store_true', default=True,
                        help='organize outputs by method (GA/FT/RL)')
    parser.add_argument('--organize_by_percentage', action='store_true', default=True,
                        help='organize outputs by forgetting percentage')

    args = parser.parse_args()
    return args