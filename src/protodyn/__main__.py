import argparse, pathlib
from protodyn.data.download import download_traj_data
from protodyn.data.preprocessing import preprocess_protein, bulk_preprocess_proteins
import multiprocessing as mp

def main():
    parser = argparse.ArgumentParser(description='A package for studying protein dynamics using Graph Neural Network')
    subparsers = parser.add_subparsers(dest="command")

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", 
            help="Preprocess a single protein trajectory to extract graph features"
    )
    preprocess_parser.add_argument("-f","--pdb", type=pathlib.Path,
        help="Path to the PDB file",
        required=True)
    preprocess_parser.add_argument("-s","--selection", type=str, default="protein")
    preprocess_parser.add_argument("-c","--chain", type=str, default="A")
    preprocess_parser.add_argument("-x","--xtc", type=pathlib.Path,
        help="Path to the XTC file",
        required=True)
    preprocess_parser.add_argument("-o","--output", type=pathlib.Path,
        help="Path to save the preprocessed data",
        required=True)
    
    # preprocess bulk subcommand
    preprocess_bulk_parser = subparsers.add_parser("preprocess_bulk", 
            help="Preprocess all the protein trajectories in the directory"
    )
    preprocess_bulk_parser.add_argument("-d","--directory", type=pathlib.Path,
        help="Path to the directory containing the PDB and XTC files",
        required=True)
    preprocess_bulk_parser.add_argument("-o","--output", type=pathlib.Path,
        help="Path to save the preprocessed data",
        required=True)
    # preprocess_bulk_parser.add_argument("-s","--selection", type=str, default="protein")
    preprocess_bulk_parser.add_argument("-t","--thread_count", type=int, default=mp.cpu_count())
    
    # Download subcommand
    download_parser = subparsers.add_parser("download", help="Download the trajectory dataset")
    download_parser.add_argument("-f","--file", type=pathlib.Path,
        help="Path to text file that contains the list of PDBs to be downloaded",
        required=True)
    download_parser.add_argument("-t","--thread_count", type=int, help="Number of threads to download the data", 
        default=4)
    
    download_parser.add_argument("-o","--output_dir", type=pathlib.Path, help="Path to Output Directory", required=True)
    
    cmd_args = parser.parse_args()
    

    if cmd_args.command == "download":
        assert cmd_args.file.exists(), f"{str(cmd_args.file)} does not exist"
        assert cmd_args.output_dir.exists(),f"{str(cmd_args.output_dir)} does not exist"
        
        download_traj_data(str(cmd_args.file),str(cmd_args.output_dir),cmd_args.thread_count)
    
    elif cmd_args.command == "preprocess":
        assert cmd_args.pdb.exists(), f"{str(cmd_args.pdb)} does not exist"
        assert cmd_args.xtc.exists(),f"{str(cmd_args.xtc)} does not exist"
        assert cmd_args.output.exists(),f"{str(cmd_args.output)} does not exist"

        if cmd_args.bulk:
            # Preprocess all the protein trajectories in the directory
            bulk_preprocess_proteins(
                str(cmd_args.directory),
                str(cmd_args.output)
            )
        else:
            # Preprocess the protein
            preprocess_protein(
                str(cmd_args.pdb),
                str(cmd_args.xtc),
                str(cmd_args.output),
                cmd_args.selection,
                cmd_args.chain
            )

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
