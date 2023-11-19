import detect_featureextraction
import argparse
from utils.general import print_args
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from utils.featmap_comparison import save_difs_mean, save_difs

def run(
        convsam,
        conv,
        source,
        pathsam= 'sam',
        pathconv= 'conv',
        savedir= 'meanfeaturemaps',
        visualize=False,
        savealldiffs=False,
):
    detect_featureextraction.run(weights=convsam,source=source,name=pathsam, visualize=True)
    detect_featureextraction.run(weights=conv, source=source, name=pathconv, visualize=True)

    path_dir_sam = os.getcwd()+os.path.join(r'\runs\detect', pathsam)
    path_dir_CneXt = os.getcwd()+os.path.join(r'\runs\detect', pathconv)
    path_save = os.getcwd()+os.path.join(r'\runs\detect', savedir)

    if savealldiffs:
        save_difs(path_dir_sam, path_dir_CneXt, path_save)
    else:
        save_difs_mean(path_dir_sam, path_dir_CneXt, path_save)

    if not visualize:
        shutil.rmtree(path_dir_CneXt)
        shutil.rmtree(path_dir_sam)

    folder0 = source.split('/')[-1].split('.')
    folder = '.'.join(folder0[0:-1])

    print(f'results saved in {path_save}\\{folder}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convsam', nargs='+', type=str)
    parser.add_argument('--conv', nargs='+', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--pathsam', default='sam')
    parser.add_argument('--pathconv', default='conv')
    parser.add_argument('--savedir', default='meanfeaturemaps')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--savealldiffs', action='store_true')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


