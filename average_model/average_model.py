import os
import argparse
import glob
import numpy as np
import torch

# e.g., python average_model.py --dst_model MFA_Conformer_cycliclr_vox12_tta/model_new/model_46_47_48_49_50.model --src_path ../save/MFA_Conformer_cycliclr_vox12/model/ --num 5 --min_epoch 46 --max_epoch 50
# e.g., python average_model.py --dst_model MFA_Conformer_cycliclr_vox12_tta/model_new/model_28_50.model --src_path ../save/MFA_Conformer_cycliclr_vox12_tta/model/ --num 2 --select

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', type=str, default='', help='averaged model')
    parser.add_argument('--src_path',  type=str, default='', help='src model path for average')
    parser.add_argument('--num',       type=int, default=10,  help='nums for averaged model')
    parser.add_argument('--min_epoch', type=int, default=41, help='min epoch used for averaging model')
    parser.add_argument('--max_epoch', type=int, default=50, help='max epoch used for averaging model')
    parser.add_argument('--select',    dest='select', action='store_true', help='select mode')
    args = parser.parse_args()
    print(args)

    #path_list = glob.glob('{}/*.model'.format(args.src_path))
    #path_list = sorted(path_list, key=os.path.getmtime)
    #path_list = path_list[-args.num:]

    if args.select:
        select_epoch = [28-1, 50-1]
        #select_epoch = [28-1, 31-1, 50-1]
        path_list = glob.glob('{}/*.model'.format(args.src_path))
        path_list = np.array(sorted(path_list))
        path_list = path_list[select_epoch]
    else:
        path_list = glob.glob('{}/*.model'.format(args.src_path))
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[args.min_epoch-1:args.max_epoch]
    print(path_list)

    avg = None
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(args.dst_model))
    torch.save(avg, args.dst_model)
