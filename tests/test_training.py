import pytest
from protodyn.training import train_on_gpu
from protodyn.utils import list_pkl_files_in_directory

def test_train_on_gpu():
    import os
    epochs = 2
    model_save_path = "/workspace/protodyn_2/outputs/models"
    log_file_path = "/workspace/protodyn_2/outputs/logs/train.log"
    cuda_device = 0
    list_of_files = list_pkl_files_in_directory("/workspace/protodyn_2/outputs/preprocessed")
    list_of_files_1 = [os.path.join("/workspace/protodyn_2/outputs/preprocessed", f) for f in list_of_files]

    # check if the list of files is not empty
    assert len(list_of_files) > 0

    train_on_gpu(epochs, model_save_path, log_file_path, cuda_device, list_of_files_1)

    # # Check if the log file is created
    # with open(log_file_path, "r") as f:
    #     lines = f.readlines()
    #     assert len(lines) > 0

    # # Check if the model file is created
    # import os
    # assert os.path.exists(model_save_path)

    # Check if the model file is not empty
    # assert os.path.getsize(model_save_path) > 0

    # # Clean up
    # os.remove(model_save_path)
    # os.remove(log_file_path)