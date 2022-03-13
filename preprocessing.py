import argparse
import os

import numpy as np
import pandas as pd

from typing import Optional, Tuple, List


def dis(x: float, y: float, xx: float, yy: float) -> float:
    return (x - xx) ** 2 + (y - yy) ** 2


def gen_velocity(pos: np.ndarray) -> np.ndarray:
    """
    @param pos: (1, T, N, 2)
    """
    v = pos[:, 1:, :, :] - pos[:, 0:-1, :, :]
    # assume v_start == v_{start+1}
    v = np.concatenate((v[:, 0:1, :, :], v), axis=1)
    assert v.shape == pos.shape

    return v


def get_neighbor_consecutive(df: pd.DataFrame, nb_id_set: List[int], start_frame: int, end_frame: int,
                             nb_num=6) -> np.ndarray:
    cond = (start_frame <= df['frame']) & (end_frame >= df['frame'])
    ret: Optional[np.ndarray] = None
    for i, frame in enumerate(sorted(np.unique(df[cond]['frame'].values))):
        new = list()
        for nb_id in nb_id_set:
            if nb_id == 0:
                new.append([0.0, 0.0])
                continue

            t_frame = frame if not df[(df['frame'] == frame) & (df['id'] == nb_id)].empty else start_frame
            t_cond = (df['frame'] == t_frame) & (df['id'] == nb_id)
            new.append([float(df[t_cond]['x'].values), float(df[t_cond]['y'].values)])

        assert len(new) == nb_num, 'find neighbor'
        new = np.array(new)
        # (nb_num, 2) -> (1, nb_num, 2)
        new = np.expand_dims(new, axis=0)
        ret = new if i == 0 else np.concatenate([ret, new], axis=0)

    # (frame num, nb_num, 2) -> (1, frame num, nb_num, 2)
    ret = np.expand_dims(ret, axis=0)
    return ret


def nearest_nb(df: pd.DataFrame, frame: int, end_frame: int, target: int, nb_num=6) -> List[int]:
    id_set = np.unique(df[df['frame'] == frame]['id'])
    # exclude itself
    id_set = id_set[id_set != target]
    nb_set = list()

    cond = (df['frame'] == frame) & (df['id'] == target)
    x = df[cond]['x'].values
    y = df[cond]['y'].values
    assert len(x) == 1, "incorrect lth x"
    assert len(y) == 1, "incorrect lth y"

    for ped_id in id_set:
        # check whether current ped exist at following 20 secs
        cond = (df['frame'] >= frame) & (df['frame'] <= end_frame) & (df['id'] == ped_id)
        if df[cond].shape[0] < 20:
            continue

        cond = (df['frame'] == frame) & (df['id'] == ped_id)
        nb_set.append(
            (ped_id, dis(x, y, df[cond]['x'].values, df[cond]['y'].values))
        )
        assert len(df[cond]['x'].values) == 1, "incorrect lth nb_x"
        assert len(df[cond]['y'].values) == 1, "incorrect lth nb_y"

    nb_set.sort(key=lambda t: t[1])
    nb_set = [t[0] for t in nb_set]

    while len(nb_set) < nb_num:
        nb_set.append(0)

    nb_set = nb_set[:nb_num]
    assert len(nb_set) == 6, "wrong number for nearest neighbor"

    return nb_set


def gen_consecutive(df: pd.DataFrame, ped_id: int, start: int, end: int, nb_set: List[int]) -> np.ndarray:
    # (T, 2) -> (T, 1, 2)
    new_self = np.expand_dims(df[(df['id'] == ped_id) &
                                 (start <= df['frame']) &
                                 (end >= df['frame'])][['x', 'y']].values,
                              axis=1)
    # (T, 1, 2) -> (1, T, 1, 2)
    new_self = np.expand_dims(new_self, axis=0)
    new_nb = get_neighbor_consecutive(df, nb_set, start, end, len(nb_set))
    new = np.concatenate([new_self, new_nb], axis=2)
    # (1, T, nb_num, 2) -> (1, T, nb_num, 4)
    new_v = gen_velocity(new)
    new = np.concatenate([new, new_v], axis=3)

    return new


def process_data(df: pd.DataFrame, nb_num=6, name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    id_set = pd.unique(df['id'])
    x = np.zeros((1, 8, 4 + nb_num * 4))
    y = np.zeros((1, 12, 4))

    print("process data: " + name)
    for ped_id in id_set:
        if len(df[df['id'] == ped_id]) < 20:
            continue

        frames = df[df['id'] == ped_id]['frame'].values
        for idx in range(len(frames[:-20])):
            nb_set = nearest_nb(df, frames[idx], frames[idx + 19], ped_id, nb_num)
            # todo whether this constrain is valid
            if len(nb_set) <= 2:
                continue

            new = gen_consecutive(df, ped_id, frames[idx], frames[idx + 7], nb_set)
            assert new.shape == (1, 8, 7, 4)
            x = new if x.any() == 0 else np.concatenate((x, new), axis=0)

            new = gen_consecutive(df, ped_id, frames[idx + 8], frames[idx + 19], nb_set)
            assert new.shape == (1, 12, 7, 4)
            y = new if y.any() == 0 else np.concatenate((y, new), axis=0)

    print('x shape: ', x.shape)
    print('y shape: ', y.shape)

    return x, y


def prepare_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    print(os.listdir(file_path))
    xx: Optional[np.ndarray] = None
    yy: Optional[np.ndarray] = None
    for i, file in enumerate(os.listdir(file_path)):
        df: pd.DataFrame = pd.read_csv(file_path + file, sep="\t", header=None)
        df.columns = ['frame', 'id', 'x', 'y']
        df[['frame']] = df[['frame']].astype(int)
        df.sort_values(['frame', 'id'])

        x, y = process_data(df, 6, file)
        if i == 0:
            xx, yy = x, y
        else:
            xx = np.concatenate([xx, x], axis=0)
            yy = np.concatenate([yy, y], axis=0)

    return xx, yy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='test', type=str, help='which dataset to be used')
    args = parser.parse_args()

    data_path = 'test/' if args.data == 'test' else 'train/'
    print(os.listdir(data_path))
    past, future = prepare_data(data_path)

    if args.data == 'train':
        np.save('x_train.npy', past)
        np.save('y_train.npy', future)
    else:
        np.save('x_test.npy', past)
        np.save('y_test.npy', future)
