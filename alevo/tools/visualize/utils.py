import json
import os
import numpy as np


def get_all_samples_and_scores(path, valid_only=False, optimal_value=None):
    """Given a log path, return all functions and scores.
    """

    def path_to_int(path):
        num = int(path.split('.')[0].split('_')[1])
        return num

    path = os.path.join(path, 'samples')
    all_func = []
    all_score = []
    dirs = list(os.listdir(path))
    dirs = sorted(dirs, key=path_to_int)
    for dir in dirs:
        file_name = os.path.join(path, dir)
        with open(file_name, 'r') as f:
            sample = json.load(f)
        func = sample['function']
        acc = sample['score']
        if valid_only:
            if acc is not None:
                all_func.append(func)
                all_score.append(acc)
        else:
            all_func.append(func)
            all_score.append(acc)

    if optimal_value is not None:
        for i, score in enumerate(all_score):
            if score is not None:
                all_score[i] = abs((abs(optimal_value) - abs(score)) / optimal_value)
    else:
        for i, score in enumerate(all_score):
            if score is not None:
                all_score[i] = abs(score)

    return all_func, all_score


def get_funcs_and_scores_with_extend(path, num_samples, valid_only=False, optimal_value=None):
    all_func, all_score = get_all_samples_and_scores(path, valid_only, optimal_value=optimal_value)
    if len(all_func) >= num_samples:
        all_func = all_func[:num_samples]
        all_score = all_score[:num_samples]
    else:
        append_num = num_samples - len(all_func)
        all_func = all_func + [all_func[-1]] * append_num
        all_score = all_score + [all_score[-1]] * append_num

    for i in range(len(all_score)):
        if all_score[i] is None:
            all_score[i] = float('inf')

    return all_func, all_score


def get_best_function(path):
    funcs, scores = get_all_samples_and_scores(path, valid_only=True)
    idx = np.argmin(scores)
    return funcs[idx], scores[idx]
