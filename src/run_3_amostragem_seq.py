"""Training of neural network."""

import argparse
from numbers import Integral
from pathlib import Path
from torch import nn
import pandas as pd
from braindecode.samplers import SequenceSampler
from matplotlib import pyplot as plt
from numpy import array
from skorch.dataset import ValidSplit


import mlflow
import mne
import numpy as np
import torch
from braindecode import EEGClassifier
from braindecode.models import SleepStagerChambon2018, SleepStagerBlanco2020, USleep, TimeDistributed
from braindecode.util import set_random_seeds
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report, confusion_matrix,
)
from braindecode.visualization import plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight, compute_sample_weight
from skorch.callbacks import EarlyStopping, Checkpoint
from skorch.callbacks import EpochScoring, LRScheduler
from skorch.helper import predefined_split, SliceDataset
from shhs_dataset import read_and_pre_processing_SHHS
from dataset import read_and_pre_processing

from EEGConformer import EEGConformer
from utils import get_exp_name, log_mlflow, set_determinism
from functools import partial
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
from braindecode.models import TimeDistributed

from sklearn.model_selection import GridSearchCV, KFold
import seaborn as sns



def main(args):



    # Instantiate the EEGConformer model



    # %% 1- General stuff, set determinism and other things

    set_determinism(args.seed)
    cuda = torch.cuda.is_available()  # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seeds(seed=args.seed, cuda=cuda)



    # Check and create dir for checkpoints
    output_dir = Path(args.run_dir)
    run_dir = output_dir / args.run_name
    print(run_dir)


    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume_macro = True
    else:
        resume_macro = False
        run_dir.mkdir(exist_ok=True)
    print(f"Macro run directory: {str(run_dir)}")

    # Create savedir
    experiment_name = get_exp_name(args.dataset, args.model)

    # Creating the list by subject:
    subject_ids = range(83)

    # %% 2- Load, preprocess and window data
    if args.type_dataset == "EDF":
        print("using the wrong dataset")
    elif args.type_dataset == "DREAMS":
        windows_dataset, list_records, sfreq = read_and_pre_processing(path=args.dataset_path,
                                                                       savepath=args.save_dir,
                                                                       n_jobs=args.num_workers)
    elif args.type_dataset == "SHHS":
        print(f"experimento número {args.number_exp} {args.model}")
        print("Lendo o dataset")
        windows_dataset, list_records, sfreq = read_and_pre_processing_SHHS(path=args.dataset_path,
                                                                       savepath=args.save_dir,
                                                                       n_jobs=args.num_workers)
        print("lido")

    k_fold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    subjects = list_records

    split_idx = [{"train_valid": train_index, "test": test_index}
                 for fold_id, (train_index, test_index) in
                 enumerate(k_fold.split(subjects))]

    split_ids = split_idx[args.part]

    # %% 3- Get the part of indice for split the training.

    batch = args.batch_size

    mlflow.set_experiment(f"{experiment_name}")

    train_valid_ix, test_ix = split_ids["train_valid"], split_ids["test"]

    run_report = run_dir / str(args.number_exp)
    if run_report.exists() & (run_report / "final_model.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)
        print(f"Micro run directory: {str(run_dir)}")

    if not resume:

        experiment = mlflow.get_experiment_by_name(f"{experiment_name}")

        active_run = mlflow.start_run(run_name=args.mlflow_dir, experiment_id=experiment.experiment_id,nested=True)

        print(f"Starting run {active_run.info.run_id}")

        train_ix, valid_ix = train_test_split(train_valid_ix, train_size=args.train_size, random_state=args.seed)
        df = windows_dataset.description

        train_ix = df[ df['subject'].isin(train_ix)].index.tolist()
        valid_ix = df[ df['subject'].isin(valid_ix)].index.tolist()
        test_ix = df[ df['subject'].isin(test_ix)].index.tolist()

        split_ids = dict(train=train_ix, test=test_ix, valid=valid_ix)

        print(f"Splitting dataset")

        splits = windows_dataset.split(split_ids)

        train_set, test_set, valid_set = splits["train"], splits["test"], splits["valid"]

        y_true_train = train_set.get_metadata()['target'].to_numpy()
        y_true_all = np.concatenate([train_set.get_metadata()['target'].to_numpy(),
                                     valid_set.get_metadata()['target'].to_numpy(),
                                     test_set.get_metadata()['target'].to_numpy()])
        class_weights = compute_class_weight('balanced', classes=np.unique(y_true_all), y=y_true_all)

        n_chans = train_set[0][0].shape[0]
        input_window_samples = train_set[0][0].shape[1]

        n_classes = 5

        print(f"Getting model")
        n_windows = 5  # Sequences of 3 consecutive windows
        n_windows_stride = 1  # Maximally overlapping sequences

        n_channels, input_size_samples = train_set[0][0].shape
        if args.model == 'stager':
            lr = 1e-3
            n_chans = 2
            model = SleepStagerChambon2018(
                n_channels=n_chans,
                n_conv_chs=12,
                sfreq=sfreq,
                apply_batch_norm=True,
            )
        elif args.model == 'blanco':

            model = SleepStagerBlanco2020(
                n_channels,
                sfreq,
                n_conv_chans=12,


                dropout=0.5,
                apply_batch_norm=True,
            )
        elif args.model == 'usleep':
            model = USleep(
                sfreq=sfreq,
                depth=16,


            )
        elif args.model == 'eegconformer':
            hidden_channels=32
            feature_extract = EEGConformer(n_channels=n_chans, n_classes=n_classes,
                                 att_depth=6,
                                 pool_time_stride=30,
                                 att_heads=10,
                                 input_window_samples=input_window_samples,
                                 final_fc_length='auto', 
                                 hidden_channels=32)
            model = nn.Sequential(
                TimeDistributed(feature_extract),  # apply model on each 30-s window
                nn.Sequential(  # apply linear layer on concatenated feature vectors
                    nn.Flatten(start_dim=1),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_channels * n_windows, n_classes)
                )
            )


        train_sampler = SequenceSampler(
            train_set.get_metadata(), n_windows, n_windows_stride, randomize=True
        )
        valid_sampler = SequenceSampler(valid_set.get_metadata(), n_windows, n_windows_stride)

        # Print number of examples per class
        print('Training examples: ', len(train_sampler))
        print('Validation examples: ', len(valid_sampler))

        ######################################################################


        y_train = [train_set[idx][1][1] for idx in train_sampler]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cuda = torch.cuda.is_available()  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        set_random_seeds(seed=31, cuda=cuda)


        if cuda:
            model.cuda()


        def balanced_accuracy_multi(model, X, y):
            y_pred = model.predict(X)
            return balanced_accuracy_score(y.flatten(), y_pred.flatten())

        train_bal_acc = EpochScoring(
            scoring=balanced_accuracy_multi,
            on_train=True,
            name='train_bal_acc',
            lower_is_better=False,
        )
        valid_bal_acc = EpochScoring(
            scoring=balanced_accuracy_multi,
            on_train=False,
            name='valid_bal_acc',
            lower_is_better=False,
        )

        patient = EarlyStopping(patience=args.patience)

        cp = Checkpoint(dirname=run_report)

        callbacks = [
            ('cp', cp),
            ('train_bal_acc', train_bal_acc),
            ('valid_bal_acc', valid_bal_acc),
            ('patient', patient),
            ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=args.n_epochs - 1))
        ]

        clf = EEGClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss,
            criterion__weight=torch.Tensor(class_weights).to(device),
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=False,
            iterator_train__sampler=train_sampler,
            iterator_valid__sampler=valid_sampler,
            train_split=predefined_split(valid_set),  # using valid_set for validation
            optimizer__lr=args.lr,
            batch_size=args.batch_size,
            callbacks=callbacks,
            device=device,
            classes=np.unique(y_train),
        )
        clf.set_params(callbacks__valid_acc=None)

        clf.fit(train_set, y=None, epochs=args.n_epochs)





        # Extract loss and balanced accuracy values for plotting from history object
        df = pd.DataFrame(clf.history.to_list())
        df.index.name = "Epoch"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        df[['train_loss', 'valid_loss']].plot(color=['r', 'b'], ax=ax1)
        df[['train_bal_acc', 'valid_bal_acc']].plot(color=['r', 'b'], ax=ax2)
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Balanced accuracy')
        ax1.legend(['Train', 'Valid'])
        ax2.legend(['Train', 'Valid'])
        fig.tight_layout()
        plt.show()
        nome = f"{args.model}_150s_comEA_lr4_{args.number_exp}.pdf"
        nome2 = f"{args.model}_150s_comEA_lr4_{args.number_exp}.csv"
        fig.savefig(nome)


        df.to_csv(nome2, index=True)

        ######################################################################












if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models')

    parser.add_argument('--seed', type=int, default=96,
                        help='random seed (default: 96)')

    parser.add_argument('--part', type=int, default=0,
                        help='define number between 0 and 5, related with cv')
    parser.add_argument('--number_exp', type=int, default=6829,
                        help='define number of the experiment')

    parser.add_argument('--dataset', type=str, default='SHHS',
                        help='name of the dataset.')

    parser.add_argument('--model', type=str, default='eegconformer',
                        help='name of the model.', choices=['stager', 'blanco', 'usleep','eegconformer'])

    parser.add_argument('--run_dir', type=str, default='/workspace/runs',
                        help='name of the run dir.')

    parser.add_argument('--save_dir', type=str, default='~/',
                        help='name of the save dir.')

    parser.add_argument('--mlflow_dir', type=str, default='/projects/mlruns/',
                        help='name of the run dir.')

    parser.add_argument('--run_name', type=str, default='test',
                        help='name of the run name.')

    parser.add_argument('--type_dataset', type=str, default="SHHS",
                        help='', choices=['EDF', 'DREAMS','SHHS'])

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='define if we gonna reload the dataset or not.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='define the value for batch size')

    parser.add_argument('--n_epochs', type=int, default=2,
                        help='define the number of epochs.')
    parser.add_argument('--lr', type=int, default=1e-4,
                        help='define the number of epochs.')

    parser.add_argument('--patience', type=int, default=2,
                        help='define the patience parameter')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='define the number of workers.')

    parser.add_argument('--train_size', type=float, default=0.8,
                        help='define the porcent of train-validation used.')

    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='define the weight_decay parameter.')

    args = parser.parse_args()
    for part in range(5):
        args.part = part
        args.number_exp = args.number_exp + 1
        main(args)
