import argparse
from time import strftime

def add_model_config_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-config', type=str, 
                       help='model configuration file')
    group.add_argument('--vocab-file', type=str, 
                       help='model vocab file')
    return parser

def add_training_args(parser: argparse.ArgumentParser):
    """Training arguments."""
    #parser.add_argument('-input-dataset','--list', nargs='+', help='List of input dataset', required=True)
    group = parser.add_argument_group('train', 'training configurations')
    group.add_argument('--gradient-accumulate', type=int, default=1,
                       help="gradient accumulate" )
    group.add_argument('--test-dataset', type=str, default=None,
                       help="test dataset directory." )
    group.add_argument('--base-path', type=str, default=None,
                       help='Path to the project base directory.')
    group.add_argument('--data-path', type=str, default=None,
                       help='Name of the dataset')
    group.add_argument('--input-dataset', type=str, default=None,
                       help='Name of the dataset')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-iters', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument('--log-iters', type=int, default=100,
                       help='number of iterations between saves')
    group.add_argument('--valid-iters', type=int, default=500,
                       help='number of iterations between validation')                  
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--train-iters', type=int, default=20000,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--max-length', type=int, default=512,
                       help='max length of input')
    group.add_argument('--start-step', type=int, default=0,
                       help='step to start or continue training')
    group.add_argument('--seed', type=int, default=42,
                       help='random seed for reproducibility')

    group.add_argument('--epochs', type=int, default=1,
                       help='total number of epochs to train over all training runs')

    # Learning rate.
    group.add_argument('--lr', type=float, default=7e-2,
                       help='initial learning rate')
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=524288,
                       help='loss scale')

    group.add_argument('--warmup-iters', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam'],
                       help='learning rate decay function')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    return parser



def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    
    args = parser.parse_args()
    return args
