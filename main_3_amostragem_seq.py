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
from sklearn.utils import compute_class_weight
from skorch.callbacks import EarlyStopping, Checkpoint
from skorch.callbacks import EpochScoring, LRScheduler
from skorch.helper import predefined_split, SliceDataset
from shhs_dataset import read_and_pre_processing_SHHS
from dataset import read_and_pre_processing
from read import load_data
from EEGConformer import EEGConformer
from utils import get_exp_name, log_mlflow, set_determinism
from functools import partial
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
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
        windows_dataset, list_records, sfreq = load_data(args.dataset_path)
    elif args.type_dataset == "DREAMS":
        windows_dataset, list_records, sfreq = read_and_pre_processing(path=args.dataset_path,
                                                                       savepath=args.save_dir,
                                                                       n_jobs=args.num_workers)
    elif args.type_dataset == "SHHS":
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

        active_run = mlflow.start_run(run_name=args.mlflow_dir, experiment_id=experiment.experiment_id)

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
        n_classes = 5
        n_chans = train_set[0][0].shape[0]
        input_window_samples = train_set[0][0].shape[1]

        n_classes = 5

        print(f"Getting model")

        n_channels, input_size_samples = train_set[0][0].shape
        if args.model == 'stager':
            lr = 1e-3
            n_chans = 2
            feat_extractor = SleepStagerChambon2018(
                n_channels=n_chans,
                n_classes=n_classes,
                n_conv_chs=12,
                sfreq=sfreq,
                input_size_s=input_size_samples / sfreq,
                apply_batch_norm=True,
                return_feats=True
            )
        elif args.model == 'blanco':
            lr = 5e-3
            feat_extractor = SleepStagerBlanco2020(
                n_channels,
                sfreq,
                n_classes=n_classes,
                input_size_s=input_size_samples / sfreq,
                dropout=0.5,
                apply_batch_norm=True,
                return_feats=True,
            )
        elif args.model == 'eegconformer':
            feat_extractor = EEGConformer(
                n_classes,
                n_channels,
                return_features=True

            )
        n_windows = 5#30s

        n_windows_stride = 0

        train_sampler = SequenceSampler(
            train_set.get_metadata(), n_windows, n_windows_stride
        )
        valid_sampler = SequenceSampler(valid_set.get_metadata(), n_windows, n_windows_stride)


        def get_center_label(x):
            if isinstance(x, Integral):
                return x
            return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x

        train_set.target_transform = get_center_label
        valid_set.target_transform = get_center_label
        y_train = [train_set[idx][1] for idx in train_sampler]
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        cuda = torch.cuda.is_available()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        set_random_seeds(seed=31, cuda=cuda)

        n_classes = 5

        n_channels, input_size_samples = train_set[0][0].shape






        model = nn.Sequential(
            TimeDistributed(feat_extractor),
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.5),
                nn.Linear(feat_extractor.get_fc_size(),n_classes)
            )
        )

        if cuda:
            model.cuda()
        lr = 5e-3
        batch_size = 64
        n_epochs = 2

        train_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
            lower_is_better=False)
        valid_bal_acc = EpochScoring(
            scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
            lower_is_better=False)
        callbacks = [
            ('train_bal_acc', train_bal_acc),
            ('valid_bal_acc', valid_bal_acc)
        ]

        clf = EEGClassifier(
            feat_extractor,
            criterion=torch.nn.CrossEntropyLoss,
            criterion__weight=torch.Tensor(class_weights).to(device),
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=False,
            iterator_train__sampler=train_sampler,
            iterator_valid__sampler=valid_sampler,
            train_split=predefined_split(valid_set),
            optimizer__lr=lr,
            batch_size=batch_size,
            callbacks=callbacks,
            device=device,
            classes=np.unique(y_train),
        )
        clf.fit(train_set, y =None, epochs = n_epochs)
        clf.initialize()
        clf.load_params(checkpoint=cp)

        print(f"Training finished!")
        print(f"Saving final model...")
        torch.save(model.state_dict(), str(run_report / 'final_model.pth'))

        clf.train_split = None  # Avoid pickling the validation set


        y_true = test_set.get_metadata()['target'].to_numpy()
        y_pred = clf.predict(test_set)
        y_prob = clf.predict_proba(test_set)

        y_prob_valid = clf.predict_proba(valid_set)
        y_prob_train = clf.predict_proba(train_set)

        y_true_test = test_set.get_metadata()['target'].to_numpy()
        y_pred_test = clf.predict(test_set)
        y_prob_test = clf.predict_proba(test_set)

        y_true_valid = valid_set.get_metadata()['target'].to_numpy()
        y_pred_valid = clf.predict(valid_set)
        y_prob_valid = clf.predict_proba(valid_set)

        y_true_train = train_set.get_metadata()['target'].to_numpy()
        y_pred_train = clf.predict(train_set)
        y_prob_train = clf.predict_proba(train_set)

        outfile = run_report / (args.model + str(args.part)+ str(args.number_exp) + ".npz")
        print("Saving output to " + str(outfile))

        np.savez(outfile, test_ix=test_ix, valid_ix=valid_ix, train_ix=train_ix,
             y_true_test=y_true_test, y_true_valid=y_true_valid, y_true_train=y_true_train,
             y_pred_test=y_pred_test, y_pred_valid=y_pred_valid, y_pred_train=y_pred_train,
             y_prob_test=y_prob_test, y_prob_valid=y_prob_valid, y_prob_train=y_prob_train)


        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        cohen_kappa = cohen_kappa_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        y_true_test = np.array([y for y in SliceDataset(test_set, 1)])
        y_pred_test = clf.predict(test_set)
        y_prob_test = clf.predict_proba(test_set)

        y_true_valid = np.array([y for y in SliceDataset(valid_set, 1)])
        y_pred_valid = clf.predict(valid_set)
        y_prob_valid = clf.predict_proba(valid_set)

        y_true_train = np.array([y for y in SliceDataset(train_set, 1)])
        y_pred_train = clf.predict(train_set)
        y_prob_train = clf.predict_proba(train_set)

        outfile = str(run_dir / (args.model + str(args.part)+str(args.number_exp)  + ".npz"))
        print("Saving label information to " + outfile)

        np.savez(
                outfile,
                y_true_test=y_true_test,
                y_true_valid=y_true_valid,
                y_true_train=y_true_train,
                y_pred_test=y_pred_test,
                y_pred_valid=y_pred_valid,
                y_pred_train=y_pred_train,
                y_prob_test=y_prob_test,
                y_prob_valid=y_prob_valid,
                y_prob_train=y_prob_train,
        )

        log_mlflow(active_run, model, args, run_report, str(test_ix[0]), y_true, y_pred, y_prob,
                   report, balanced_accuracy, cohen_kappa, y_prob_train=y_prob_train,
                   y_prob_valid=y_prob_valid,
                   split_ids=split_ids)

        print(f"Run {active_run.info.run_id} over")
        print("---------------------------------------")
    else:
        print(f"Model already trained, saved in: {str(run_report)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models')

    parser.add_argument('--seed', type=int, default=96,
                        help='random seed (default: 96)')

    parser.add_argument('--part', type=int, default=0,
                        help='define number between 0 and 5, related with cv')
    parser.add_argument('--number_exp', type=int, default=1420,
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

    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='define the number of epochs.')

    parser.add_argument('--patience', type=int, default=80,
                        help='define the patience parameter')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='define the number of workers.')

    parser.add_argument('--train_size', type=float, default=0.8,
                        help='define the porcent of train-validation used.')

    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='define the weight_decay parameter.')

    args = parser.parse_args()
    main(args)
