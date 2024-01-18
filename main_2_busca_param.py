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

from model import BENDR
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
from model2 import EEGConformer
from model3 import EEGConformer2
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

        y_true_all = np.concatenate([train_set.get_metadata()['target'].to_numpy(),
                                     valid_set.get_metadata()['target'].to_numpy(),
                                     test_set.get_metadata()['target'].to_numpy()])
        n_classes = 5
        n_chans = train_set[0][0].shape[0]
        input_window_samples = train_set[0][0].shape[1]

        print(f"Getting model")

        n_channels, input_size_samples = train_set[0][0].shape
        if args.model == 'stager':

            n_chans = 2
            model = partial(SleepStagerChambon2018,
                            n_channels=n_chans,
                            n_classes=n_classes,
                            n_conv_chs=16,
                            sfreq=sfreq,
                            input_size_s=input_size_samples / sfreq,
                            apply_batch_norm=True,

                            )
        elif args.model == 'blanco':

            model = partial(SleepStagerBlanco2020,
                n_channels,
                sfreq,
                n_classes=n_classes,
                input_size_s=input_size_samples / sfreq,
                dropout=0.5,
                apply_batch_norm=True,
            )
        elif args.model == 'usleep':

            batch = 256

            model = partial(USleep,
                            in_chans=n_chans,
                            n_classes=n_classes,
                            sfreq=sfreq,
                            depth=12,
                            input_size_s=input_size_samples / sfreq,
                            with_skip_connection=True,

                            )
        elif args.model == 'eegconformer':


            model = partial(EEGConformer, n_channels=n_chans, n_classes=n_classes,
                            att_depth=6,
                            att_heads=10,
                            input_window_samples=input_window_samples,
                            final_fc_length='auto', )

            train_set = train_set


            mne.set_log_level('WARNING')










            batch_size = 64
            n_epochs = 500
            train_bal_acc = EpochScoring(
                scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
                lower_is_better=False)
            valid_bal_acc = EpochScoring(
                scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
                lower_is_better=False)
            patient = EarlyStopping(patience=args.patience)
            clf = EEGClassifier(
                model,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.AdamW,
                optimizer__lr=[],
                batch_size=batch_size,


                callbacks=[
                    ('train_bal_acc', train_bal_acc),
                    ('valid_bal_acc', valid_bal_acc),
                    ("patience", patient)
                    ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),

                ],
                classes=[0, 1, 2, 3, 4],
                iterator_train__shuffle=True,
                train_split=predefined_split(valid_set),
                iterator_train__num_workers=args.num_workers,
                iterator_valid__num_workers=args.num_workers,
                device=device,
            )

            train_X = SliceDataset(train_set, idx=0)
            train_y = array([y for y in SliceDataset(train_set, idx=1)])

            cv = KFold(n_splits=2, shuffle=True, random_state=args.seed)
            learning_rates = [1e-5]
            depth=[12]
            fit_params = {'epochs': n_epochs}
            param_grid = {
                'optimizer__lr': learning_rates,
                'module__depth':depth
            }
            search = GridSearchCV(
                estimator=clf,
                param_grid=param_grid,
                return_train_score=True,
                scoring='balanced_accuracy',
                cv = cv,
                refit=True,
                verbose=1,
                error_score='raise',
                n_jobs=1,
            )

            search.fit(train_X, train_y, **fit_params)


            search_results = pd.DataFrame(search.cv_results_)
            search_results.to_csv('resultado_eegconformer_shallow_euclidean_alignment_usleep_2.csv')

            pivot_table = search_results.pivot(index='param_optimizer__lr',
                                               columns='param_module__depth',
                                               values='mean_test_score')

            fig, ax = plt.subplots()
            sns.heatmap(pivot_table, annot=True, fmt=".3f",
                        cmap="YlGnBu", cbar=True)
            plt.title('Grid Search Mean Test Scores')
            plt.ylabel('Learning Rate')
            plt.xlabel('depth')
            plt.tight_layout()
            plt.show()
            plt.savefig("Resultados dos hiperparâmetros_eegconformer_shallow_euclidean_alignment_usleep_2.png")
            best_run = search_results[search_results['rank_test_score'] == 1].squeeze()
            print(
                f"Best hyperparameters were {best_run['params']} which gave a validation "
                f"accuracy of {best_run['mean_test_score'] * 100:.2f}% (training "
                f"accuracy of {best_run['mean_train_score'] * 100:.2f}%).")

            eval_X = SliceDataset(valid_set, idx=0)
            eval_y = array([y for y in SliceDataset(valid_set, idx=1)])
            score = search.score(eval_X, eval_y)
            print(f"Eval bal_accuracy is {score * 100:.2f}%.")



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

    parser.add_argument('--model', type=str, default='usleep',
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