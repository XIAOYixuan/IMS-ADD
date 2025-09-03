# encoding: utf-8
# author: Yixuan
#
#
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from onebit.config import ConfigManager
from onebit.task import Evaluator 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True
    )

    args, unk = parser.parse_known_args()
    args.cli_args = unk
    return args 

def main():
    args = parse_args()

    config_manager = ConfigManager(args.config)
    if args.cli_args:
        config_manager.merge_with_cli(args.cli_args)
    
    eval = Evaluator(config_manager)
    eval.start()

if __name__ == '__main__':
    main()