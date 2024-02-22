import copy
import numpy as np
import pandas as pd
from braindecode.datasets import create_from_X_y
from pyriemann.utils.covariance import covariances


def euclidean_alignment(data):
    data = copy.deepcopy(data).transpose(0,2,1)


    assert len(data.shape) == 3

    r = 0
    cov = covariances(data)

    r = np.sum(cov, 0)



    r = r / len(data)

    try:
        r_op = np.linalg.inv(r)
    except np.linalg.LinAlgError:
        print("Inverse of covariance matrix cannot be computed.")
        r_op = np.eye(r.shape[0])

    result = np.matmul(r_op, data)
    print(f"O sujeito foi lido")




    return result, r_op

def read_SHHS1(exp_n: int, data_path: str, actual_subj_path: str, euclid_alignment: bool):
    subjects = pd.read_csv(actual_subj_path, sep=" ", header=None)
    numeric_columns = subjects.select_dtypes(include=['number'])
    numeric_columns.fillna(numeric_columns.mean(), inplace=True)
    subjects.update(numeric_columns)
    list_sub = subjects.values[exp_n][0]
    data_raw = np.load(data_path + list_sub)

    data_raw_x = data_raw['x']
    data_raw_y = data_raw['y']

    if(euclid_alignment):
        aligned_data_x, alignment_matrix = euclidean_alignment(data_raw_x)
        return aligned_data_x, data_raw_y
    data_raw_x = copy.deepcopy(data_raw_x).transpose(0, 2, 1)
    return data_raw_x, data_raw_y






def read_and_pre_processing_SHHS(window_size_s=30, sfreq=125, savepath='', path=None,
                                 data_path="/workspace/dataset/", n_jobs=1,
                                 actual_subj="/workspace/actual_subjects/healthy_subjects.txt", euclid_alignment = True):
    range_subjects = list(range(0, 5))
    list_X = []
    list_y = []
    list_subject_trial = []

    for subject in range_subjects:
        X, y = read_SHHS1(subject, data_path, actual_subj , euclid_alignment)

        list_X.append(X)
        #if(euclid_alignment == False):
            #print(f"Carregando o sujeito {subject} sem o alinhamento")


        list_y.append(y)

        list_subject_trial.append([subject] * len(y))
    list_X = [value for value in list_X if value is not None]
    X_append = np.concatenate(list_X, axis=0)
    Y_append = np.concatenate(list_y, axis=0)




    metainfo = np.concatenate(list_subject_trial, axis=0)


    sequence_len = int(sfreq * window_size_s)
    stride_len = int(sfreq * window_size_s)


    mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }
    windows = create_from_X_y(X=X_append, y=Y_append, window_size_samples=sequence_len,
                              window_stride_samples=stride_len, drop_last_window=False, sfreq=sfreq)





    df = windows.description
    df['subject'] = metainfo
    windows.set_description(df, overwrite=True)


    return windows, range_subjects, sfreq




