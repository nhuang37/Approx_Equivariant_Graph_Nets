from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
import pickle


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader) #debug

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        orbits, idx_swap_back = pickle.load( open(args.orbit_filename, "rb"))
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, aut_flag=args.aut, orbits=orbits, idx_swap_back=idx_swap_back, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--aut', default=False, type=bool, help='Set to true to use aut-gnn')
    parser.add_argument('--orbit_filename', default="orbit_idx.p", type=str, help='Load the list of lists that contain the group orbits/communinties')

    args = parser.parse_args()
    main(args)
