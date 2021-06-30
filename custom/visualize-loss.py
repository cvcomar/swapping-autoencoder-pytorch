import matplotlib.pyplot as plt
import regex
import torch
import argparse


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./loss_log.txt')

    return parser.parse_args()


def main():
    args = parse_arg()
    path_log = args.path

    logs = []
    with open(path_log) as f:

        for ln in f.readlines():
            if 'iters' not in ln:
                continue
            parse = regex.findall('\w+: [0-9.]+', ln)
            dic = {}
            for p in parse:
                name, val = p.split(': ')
                val = float(val)
                dic[name] = val

            logs.append(dic)

    logs = list(filter(lambda x : len(x.keys()) == 16, logs))
    iters = [ log['iters'] for log in logs ] 
    # labels = ['D_R1', 'D_mix', 'D_real', 'D_rec', 'D_total', 'G_GAN_mix', 'G_GAN_rec', 'G_L1', 'G_mix', 'L1_dist', 'PatchD_mix', 'PatchD_real']
    labels = ['G_L1', 'G_GAN_rec', 'G_GAN_mix', 'D_total']


    for label in labels:
        vals = [ log[label] for log in logs ] 
        plt.plot(iters, vals)

    plt.legend(labels)

    plt.show()

if __name__ == '__main__':
    main()
