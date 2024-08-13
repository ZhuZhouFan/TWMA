import os
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import random
import numpy as np


def sub_job(date, data_dir, lag_order, horizon):
    data_info = list()
    date_path = os.path.join(data_dir, date)
    for root, dirs, files in os.walk(date_path):
        for dir_ in dirs:
            feature_path = os.path.join(date_path, dir_, f'I{lag_order}.jpeg')
            label_path = os.path.join(date_path, dir_, f'R{horizon}.npy')
            if os.path.exists(feature_path) & os.path.exists(label_path):
                data_info.append(os.path.join(date_path, dir_))
    return np.array(data_info)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--img-path', type=str, default='Your/Image/Path/Here',
                        help='Directory of images')
    parser.add_argument('--data-path', type=str, default='Your/Data/Path/Here',
                        help='Directory of other data')
    parser.add_argument('--img-type', type=str, default='ohlc',
                        help='Type of images')
    parser.add_argument('--start-time', type=str, default='2014-01-01',
                        help='Training+valid samples start time')
    parser.add_argument('--test-time', type=str, default='2021-01-01',
                        help='Testing sample start time (i.e., Training+valid samples end time)')
    parser.add_argument('--end-time', type=str, default='2023-05-01',
                        help='Testing sample end time')
    parser.add_argument('--lag-order', type=int, default=20,
                        help='Number of trading days included in each image')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Forecasting horizon')
    parser.add_argument('--worker', type=int, default=75,
                        help='Number of cpus used in this program.')

    args = parser.parse_args()

    data_dir = f'{args.img_path}/{args.img_type}'
    date_list = os.listdir(data_dir)
    date_list.sort()
    train_valid_date_list = list(
        filter(lambda x: (x >= args.start_time) & (x < args.test_time), date_list))
    test_date_list = list(
        filter(lambda x: (x >= args.test_time) & (x <= args.end_time), date_list))

    train_valid_info = np.hstack(Parallel(n_jobs=args.worker)(delayed(sub_job)
                                                              (date, data_dir,
                                                               args.lag_order, args.horizon)
                                                              for date in tqdm(train_valid_date_list)))
    random.shuffle(train_valid_info)
    split_point = int(len(train_valid_info) * 0.7)
    train_info = train_valid_info[:split_point]
    valid_info = train_valid_info[split_point:]

    test_info = np.hstack(Parallel(n_jobs=args.worker)(delayed(sub_job)
                                                       (date, data_dir,
                                                        args.lag_order, args.horizon)
                                                       for date in tqdm(test_date_list)))

    if not os.path.exists(f'{args.data_path}/data_info'):
        os.makedirs(f'{args.data_path}/data_info')

    np.save(f'{args.data_path}/data_info/train_{args.img_type}_{args.start_time}_{args.test_time}_{args.lag_order}_{args.horizon}.npy',
            train_info)
    np.save(f'{args.data_path}/data_info/valid_{args.img_type}_{args.start_time}_{args.test_time}_{args.lag_order}_{args.horizon}.npy',
            valid_info)
    np.save(f'{args.data_path}/data_info/test_{args.img_type}_{args.start_time}_{args.test_time}_{args.lag_order}_{args.horizon}.npy',
            test_info)
