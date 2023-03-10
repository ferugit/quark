# Fernando López Gavilánez, 2023

import os
import json
import datetime
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import keyword_spotting.trainer as model_trainer
from keyword_spotting.model import LeNetAudio
from keyword_spotting.reporter import Reporter
from utils.experiments_utils import *


def main(args):

    if ((not os.path.exists(args.data_path)) or (not os.path.exists(args.partition_path)) or
        (not os.path.exists(args.results_path))):
        raise Exception('Non valid data or partition paths!')

    # TODO: create a versioning system to no overwrite experiment
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_id = 'lenet'
    experiment_path = os.path.join(args.results_path, model_id)
    check_path(experiment_path)

    # Set Reporter
    reporter = Reporter(experiment_path, model_id  + '_report.json')
    reporter.report('arguments', vars(args))
    reporter.report('experiment_date', current_time)

     # Summary 
    tensorboard_writer = SummaryWriter(experiment_path)
    tb_metrics_dict = {
        'best_val_loss':0,
        'best_epoch':0,
        'training_time':0,
        'macro-avg-f1-score':0,
        'test_loss':0,
        'test_accuracy':0,
    }
    exp, ssi, sei = hparams(vars(args), metric_dict=tb_metrics_dict)
    tensorboard_writer.file_writer.add_summary(exp)
    tensorboard_writer.file_writer.add_summary(ssi)                 
    tensorboard_writer.file_writer.add_summary(sei)
    
    # Calculate number of classes from dataset
    num_classes = len(list(json.load(open(os.path.join(args.partition_path, 'classes_index.json'))).keys()))
    
    # TODO: select architecture name from arguments
    model = LeNetAudio(
        num_classes,
        window_size=int(args.window_size*args.sampling_rate)
        )

    # TODO: create data augments

    trainer = model_trainer.AudioTrainer(
        model,
        args.seed,
        num_classes,
        args.cuda,
        args.window_size,
        args.sampling_rate,
        tensorboard_writer,
        reporter,
        experiment_path,
        model_id,
        args.batch_size
    )

    trainer.prepare_data(
        args.partition_path,
        args.data_path
        )

    trainer.train(
        args.epochs,
        args.optimizer,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.use_batch_sampler,
        args.patience
        )

    # Load best checkpoint
    trainer.load_best_checkpoint()

    # evaluate here
    trainer.test()

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

    # GPU/CPU
    parser.add_argument('--cuda', type=bool, default=False, metavar='C', help='flag to use or not GPU')

    # Optimizer
    parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam | rmsprop')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD', help='weight decay for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

    # Audio
    parser.add_argument('--sampling_rate', type=float, default=16000, metavar='SR', help='sampling rate of the audio signal')
    parser.add_argument('--window_size', type=float, default=1.0, metavar='TW', help='time window covered by every data sample')

    # Data and partitions
    parser.add_argument('--data_path', default='', help='root path of the audio dataset')
    parser.add_argument('--partition_path', default='', help='path to the partition folder containing the train, dev and test dataframes')
    parser.add_argument('--results_path', default='', help='path to place the results of the experiments')

    args = parser.parse_args()

    main(args)