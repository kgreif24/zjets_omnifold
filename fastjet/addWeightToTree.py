""" addWeightToTree.py - This script adds a set of omnifold weights to the MC train or test tree.
This step needs to be run before you can plot arbitrary observables using fastjet. One issue with this 
is that we need to duplicate a huge amount of data, which is not ideal. Should put some thought into
the optimal way to do this.
"""


import argparse
import uproot
import numpy as np 


def main():

    parser = argparse.ArgumentParser(description='Add weights to MC tree')
    parser.add_argument('--input', type=str, default='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root', help='Input file')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--weights', type=str, help='Weights file')
    parser.add_argument('--get_train', action='store_true', help='Append train weights instead of test')
    args = parser.parse_args()

    fileMC = uproot.open(args.input)
    treeMC = fileMC['OmniTree']
        
    branches_mc = {name: treeMC[name].array() for name in treeMC.keys()}
    weights = np.load(args.weights)['train' if args.get_train else 'test']
    branches_mc["omni_weight"] = weights


    with uproot.recreate(args.output) as new_file:
        new_file["OmniTree"] = branches_mc

    return


if __name__ == "__main__":
    main()
    pass 

