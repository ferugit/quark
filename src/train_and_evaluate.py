# Fernando López Gavilánez, 2023

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import keyword_spotting.trainer as trainer

from keyword_spotting.model import LeNetAudio


def main(args):

    if (not os.path.ispath(args.data_path)) or (not os.path.ispath(args.partition_path)):
        raise Exception('Non valid data or partition paths!')
    
    model = LeNetAudio(
        num_classes,
        window_size=int(args.window_size*args.sampling_rate)
        )

    trainer = trainer.AudioTrainer(
        model,
        args.seed,
        args.num_classes,
        args.cuda,
        args.window_size,
        args.sampling_rate
    )

    trainer.prepare_data(args.data_path)

    trainer.train(
        args.epochs,
        args.optimizer,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.balance
        )

    # evaluate here

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to partitions for donateacry dataset")
    
    # Hyperparameters
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--use_sampler', dest='use_batch_sampler', action='store_true', help='use Weighted Random Sampler to deal with data unbalance')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--lr_patience', type=int, default=20, help='number of epochs of no loss improvement before updating lr')
    parser.add_argument('--patience', type=int, default=60, help='number of epochs of no loss improvement before stop training')

    # Optimizer
    parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam | rmsprop')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD', help='weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

    # Audio
    parser.add_argument('--sampling_rate', type=float, default=16000, metavar='SR', help='sampling rate of the audio signal')
    parser.add_argument('--time_window', type=float, default=1.0, metavar='TW', help='time window covered by every data sample')

    # Data and partitions
    parser.add_argument('--data_path', default='', help='root path of the audio dataset')
    parser.add_argument('--partition_path', default='', help='path to the partition folder containing the train, dev and test dataframes')

    args = parser.parse_args()

    main(args)