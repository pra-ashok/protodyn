import argparse, pathlib
# from protodyn.dataset import GraphDatasetPipeline
# from protodyn.training import train_on_gpu
# from protodyn.model_test import test_on_gpu
from protodyn.data.download import download_traj_data
# import protodyn.pdb_to_graph as ptg
# import protodyn.prediction as pred
import multiprocessing as mp
# from protodyn.model import *

# mp.set_start_method('fork')

# architecture_mapping = {
#     "BasicGINE": GINE_MultiTask,
#     "GINEAttention": GINEAttentionModel,
#     "GINEMultiHeadAttention": GINEMultiHeadAttnModel,
#     "ResGATModel": ResGATModel,
#     "ResTransformerModel": ResTransformerModel
# }

def main():
    parser = argparse.ArgumentParser(description='A package for studying protein dynamics using Graph Neural Network')
    subparsers = parser.add_subparsers(dest="command")

    # # Dataset Commands
    # dataset_parser = subparsers.add_parser("dataset", help="Build Trajectory dataset")
    # dataset_parser.add_argument('-b','--build_dataset',  
    #                     action='store_true',
    #                     help='Use this flag to build dataset',
    #                     )
    # dataset_parser.add_argument('-r','--train_test_split_ratio', 
    #                     type=int, help=' train:test ratio -> train:1', default=4)
    # dataset_parser.add_argument('-o','--dataset_dir', type=pathlib.Path,
    #                     help='Path to the directory where we expect the output dataset')
    # dataset_parser.add_argument('-i','--traj_dataset_folder',
    #                     type=pathlib.Path, help='Path of the trajectory_analysis directory')
    # dataset_parser.add_argument('-c','--hpc',  
    #                     action='store_true',
    #                     help='Use this flag if running on HPC (High Performance Computer)',
    #                     )
    # dataset_parser.add_argument('-p','--preprocess',  
    #                     action='store_true',
    #                     help='Use this flag to preprocess ',
    #                     )
    # dataset_parser.add_argument('-f','--pdb_file', type=pathlib.Path,
    #                     help='Path to the directory where we expect the output dataset')
    # dataset_parser.add_argument('-k','--bulk_process',  
    #                     action='store_true',
    #                     help='Preprocess all the pdb_files in a directory',
    #                     )
    # dataset_parser.add_argument('-d','--pdb_dir', type=pathlib.Path,
    #                     help='Path to the directory where we expect the PDBs to preprocess')
    
    # # Build Graph subcommand
    # bg_cmd_parser = subparsers.add_parser("build_graph", 
    #         help="Builds a Protein-GINE_Graph from PDB structure")
    # bg_cmd_parser.add_argument("-f","--file", type=pathlib.Path,
    #     help="Path to the PDB file",
    #     required=True)
    # bg_cmd_parser.add_argument('-p','--save_pickle',  
    #     action='store_true',
    #     help='Use this flag to save the pickle file of the Graph generated',
    # )
    # bg_cmd_parser.add_argument("-s","--selection", type=str, 
    #     help="It can be either 'chain [name]' or 'protein'(default) to be processed", default="protein"
    # )
    
    # Download subcommand
    download_parser = subparsers.add_parser("download", help="Download the trajectory dataset")
    download_parser.add_argument("-f","--file", type=pathlib.Path,
        help="Path to text file that contains the list of PDBs to be downloaded",
        required=True)
    download_parser.add_argument("-t","--thread_count", type=int, help="Number of threads to download the data", 
        default=4)
    
    download_parser.add_argument("-o","--output_dir", type=pathlib.Path, help="Path to Output Directory", required=True)
    # train_parser.add_argument("-b","--batch_size", type=int, help="Batch size of the training data", default=8)

    # # Train subcommand
    # train_parser = subparsers.add_parser("train", help="Train a model")
    # train_parser.add_argument("-d","--data", type=pathlib.Path, help="Path to training data", required=True)
    # train_parser.add_argument("-o","--model_output", type=pathlib.Path, help="Path to save trained model", required=True)
    # train_parser.add_argument("-e","--epochs", type=int, help="Number of training epochs", default=10)
    # # train_parser.add_argument("-b","--batch_size", type=int, help="Batch size of the training data", default=8)
    # train_parser.add_argument("-l","--loss_file_path", type=pathlib.Path, help="Path to save Losses", required=True)
    # # train_parser.add_argument("-w","--weight_file_path", type=pathlib.Path, help="Path to save specific Weights after each epoch", required=True)
    # train_parser.add_argument("-i","--cuda_device_id", type=int, help="Cuda Device ID to train", default=0)
    # train_parser.add_argument("-a", "--architecture", choices=["BasicGINE", "GINEAttention", 
    #                     "GINEMultiHeadAttention","ResGATModel","ResTransformerModel"],
    #                     help="Model architecture to use", required=True)
    
    # # Test subcommand
    # test_parser = subparsers.add_parser("test", help="Test a trained model")
    # test_parser.add_argument("-d","--data", type=pathlib.Path, help="Path to test data", required=True)
    # test_parser.add_argument("-m","--model", type=pathlib.Path, help="Path to trained model", required=True)
    # # train_parser.add_argument("-b","--batch_size", type=int, help="Batch size of the training data", default=8)

    # # Predict subcommand
    # predict_parser = subparsers.add_parser("predict", help="Make predictions using a trained model")
    # predict_parser.add_argument("-f","--pkl_file", type=pathlib.Path, 
    #     help="Path to the Pickle file of the graph for prediction", required=True)
    # predict_parser.add_argument("--model", type=pathlib.Path, help="Path to trained model", required=True)
    # predict_parser.add_argument("--output", type=pathlib.Path, help="Path to save predictions", required=False)

    cmd_args = parser.parse_args()
    print(cmd_args)
    # if cmd_args.command == "dataset":
    #     assert cmd_args.dataset_dir.exists(), f"{str(cmd_args.dataset_dir)} does not exist"
    #     assert cmd_args.traj_dataset_folder.exists(), f"{str(cmd_args.traj_dataset_folder)} does not exist"
    #     if cmd_args.build_dataset:
    #         graph_dataset = GraphDatasetPipeline(
    #             cmd_args.train_test_split_ratio, str(cmd_args.dataset_dir), str(cmd_args.traj_dataset_folder))
    #         if cmd_args.hpc:
    #             graph_dataset.build_dataset_hpc()
    #         else:
    #             graph_dataset.build_dataset()
    
    # elif cmd_args.command == "build_graph":
    #     assert cmd_args.file.exists(), f"{str(cmd_args.file)} does not exist"
    #     ptg.main(cmd_args)

    if cmd_args.command == "download":
        assert cmd_args.file.exists(), f"{str(cmd_args.file)} does not exist"
        assert cmd_args.output_dir.exists(),f"{str(cmd_args.output_dir)} does not exist"
        
        download_traj_data(str(cmd_args.file),str(cmd_args.output_dir),cmd_args.thread_count)

    # elif cmd_args.command == "train":
    #     assert cmd_args.data.exists(), f"{str(cmd_args.data)} does not exist"
    #     assert cmd_args.model_output.exists(),f"{str(cmd_args.model_output)} does not exist"
    #     assert cmd_args.architecture in architecture_mapping, f"Invalid architecture selected: {args.architecture}"

    #     print(cmd_args)
    #     selected_architecture = architecture_mapping[cmd_args.architecture]
    #     print("Selected Model Architecture:",selected_architecture)

    #     train_on_gpu(dataset_dir = cmd_args.data,
    #                  epochs = cmd_args.epochs,
    #                  model_save_path = cmd_args.model_output,
    #                  log_file_path = cmd_args.loss_file_path,
    #                  cuda_device = cmd_args.cuda_device_id,
    #                  GNNArch = selected_architecture
    #                 )
        
    # elif cmd_args.command == "test":
    #     assert cmd_args.data.exists(), f"{str(cmd_args.data)} does not exist"
    #     assert cmd_args.model.exists(),f"{str(cmd_args.model)} does not exist"

    #     test_on_gpu(cmd_args.data,
    #                 cmd_args.model)

    #     print(cmd_args)
        # test(args)
    # elif cmd_args.command == "predict":
    #     print(cmd_args)
    #     pred.spgpo(cmd_args.pkl_file,cmd_args.model, cmd_args.output)
        # predict(args)

if __name__ == '__main__':
    main()
