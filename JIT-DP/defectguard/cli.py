import argparse, os, random, sys, torch
from defectguard.utils.logger import logger
from defectguard.utils.utils import SRC_PATH
from datetime import datetime
import numpy as np
from defectguard.inferencing import inferencing
from defectguard.mining import mining
from defectguard.training import training
from defectguard.evaluating import evaluating

__version__ = "0.1.33"

def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    available_languages = ["Python", "Java", "C++", "C", "C#", "JavaScript", "TypeScript", "Ruby", "PHP", "Go", "Swift"]
    models = ["deepjit", "simcom", "lapredict", "tlel", "lr"]
    modes = ["local", "remote"]

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-dg_save_folder", default=".", type=str, help="")
    common_parser.add_argument("-mode", type=str, default="local", help="Mode of extractor", choices=modes)
    common_parser.add_argument("-repo_name", type=str, default="", help="Repo name")
    common_parser.add_argument("-repo_owner", type=str, default="", help="Repo owner name")
    common_parser.add_argument("-repo_path", type=str, default="", help="Path to git repository")
    common_parser.add_argument("-repo_clone_url", type=str, default=None, help="URL to repository")
    common_parser.add_argument("-repo_clone_path", type=str, default=None, help="Path to clone repository")
    common_parser.add_argument("-repo_language", type=str, default="", choices=available_languages, help="Main language of repo")
    common_parser.add_argument("-uncommit", action="store_true", help="Include uncommit in list when using -top")
    common_parser.add_argument("-reextract", action="store_true", help="Re-extract data from repository")
    
    mining_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    mining_parser.set_defaults(func=mining)
    mining_parser.add_argument("-pyszz_path", type=str, default="", help="Path to pyszz")
    mining_parser.add_argument("-szz", type=str, choices=["bszz", "vszz"], default= "bszz", help="Variant of Szz Algorithm")
    mining_parser.add_argument("-workers", type=int, default=1, help="Number of parallel workers")

    inferencing_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    inferencing_parser.set_defaults(func=inferencing)
    inferencing_parser.add_argument("-models", nargs="+", type=str, default=[], choices=models, help="List of models")
    inferencing_parser.add_argument("-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1")
    inferencing_parser.add_argument("-sort", action="store_true", help="Sort output of model by predict score")
    inferencing_parser.add_argument("-vsc", action="store_true", help="Output for vsc")
    inferencing_parser.add_argument("-threshold", type=float, default=0.5, help="Threshold for warning")
    inferencing_parser.add_argument("-no_warning", action="store_true", help="Supress output warning")
    inferencing_parser.add_argument("-commit_hash", nargs="+", type=str, default=[], help="List of commit hashes")
    inferencing_parser.add_argument("-top", type=int, default=0, help="Number of top commits")

    training_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    training_parser.set_defaults(func=training)
    training_parser.add_argument("-model", type=str, default=[], choices=models, help="List of models")
    training_parser.add_argument("-from_pretrain", action="store_true", help="")
    training_parser.add_argument("-epochs",type=int,default=1, help="")
    training_parser.add_argument("-dictionary",type=str,default=None, help="")
    training_parser.add_argument("-hyperparameters",type=str,default=None, help="")
    training_parser.add_argument("-feature_train_set", type=str, default=None, help="")
    training_parser.add_argument("-commit_train_set", type=str, default=None, help="")
    training_parser.add_argument("-commit_val_set", type=str, default=None, help="")
    training_parser.add_argument("-learning_rate", type=float, default=5e-5, help="")
    training_parser.add_argument("-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1")

    evaluating_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    evaluating_parser.set_defaults(func=evaluating)
    evaluating_parser.add_argument("-model", type=str, default=[], choices=models, help="List of models")
    evaluating_parser.add_argument("-from_pretrain", action="store_true", help="")
    evaluating_parser.add_argument("-dictionary",type=str,default=None, help="")
    evaluating_parser.add_argument("-hyperparameters",type=str,default=None, help="")
    evaluating_parser.add_argument("-feature_test_set", type=str, default=None, help="")
    evaluating_parser.add_argument("-commit_test_set", type=str, default=None, help="")
    evaluating_parser.add_argument("-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1")

    parser = argparse.ArgumentParser(prog="DefectGuard", description="A tool for mining, training, evaluating for Just-in-Time Defect Prediction")
    parser.add_argument("-version", action="version", version="%(prog)s " + __version__)
    parser.add_argument("-debug", action="store_true", help="Turn on system debug print")
    parser.add_argument("-log_to_file", action="store_true", help="Logging to file instead of stdout")
    subparsers = parser.add_subparsers()
    subparsers.add_parser('mining', parents=[mining_parser], help='Mining Function')
    subparsers.add_parser('inferencing', parents=[inferencing_parser], help='Inferencing Function')
    subparsers.add_parser('training', parents=[training_parser], help='Training Function')
    subparsers.add_parser('evaluating', parents=[evaluating_parser], help='Evaluating Function')

    options = parser.parse_args(args)

    if not options.debug:
        logger.disable()

    if options.log_to_file:
        # Create a folder named 'logs' if it doesn't exist
        if not os.path.exists(f"{SRC_PATH}/logs"):
            os.makedirs(f"{SRC_PATH}/logs")

        # Define a file to log IceCream output
        log_file_path = os.path.join(f"{SRC_PATH}/logs", "logs.log")

        # Replace logging configuration with IceCream configuration
        logger.configureOutput(
            prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ',
            outputFunction=lambda x: open(log_file_path, "a").write(x + "\n"),
        )

    if not hasattr(options, 'func'):
        parser.print_help()
        exit(1)
    options.func(options)

if __name__ == "__main__":
    main()