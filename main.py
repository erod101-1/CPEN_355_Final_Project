import kagglehub
import shutil
import os
import torch
from prepare_dataset import main as prepare_dataset_main, create_data_loaders
from CNN import CNNModel, train_cnn, test_cnn
from DNN import DNNModel, train_dnn, test_dnn

# --- Flags ---
PREP_DATASET = True   # Set to False to skip download/preparation and use existing data
TRAIN_CNN    = True   # Set to False to skip CNN training (loads cnn_model.pth instead)
TRAIN_DNN    = True   # Set to False to skip DNN training (loads dnn_model.pth instead)
TEST_CNN     = True   # Set to False to skip CNN evaluation
TEST_DNN     = True   # Set to False to skip DNN evaluation

# --- Hyperparameters ---
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-3
PREPARED_DATASET_DIR = "prepared_dataset"


if __name__ == "__main__":

    if PREP_DATASET:
        print("--- Preparing Dataset ---")
        path = kagglehub.dataset_download("luischuquimarca/healthy-and-defective-fruits")

        current_dir = os.getcwd()
        dataset_name = "healthy-and-defective-fruits"
        destination = os.path.join(current_dir, dataset_name)

        if os.path.exists(destination):
            shutil.rmtree(destination)

        shutil.copytree(path, destination)
        print("Path to dataset files:", path)
        print("Copied to current directory:", destination)
        prepare_dataset_main()

    if TRAIN_CNN or TRAIN_DNN or TEST_CNN or TEST_DNN:
        print("\nLoading data loaders...")
        train_loader, val_loader, test_loader, classes = create_data_loaders(PREPARED_DATASET_DIR)
        num_classes = len(classes)
        print(f"Classes ({num_classes}): {classes}")

    # --- CNN ---
    if TRAIN_CNN:
        print("\n--- Training CNN ---")
        cnn_model = CNNModel(num_classes=num_classes).to(DEVICE)
        cnn_model = train_cnn(cnn_model, train_loader, val_loader,
                              num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)
        torch.save(cnn_model.state_dict(), "cnn_model.pth")
        print("CNN model saved to cnn_model.pth")

    if TEST_CNN:
        print("\n--- Testing CNN ---")
        if not TRAIN_CNN:
            cnn_model = CNNModel(num_classes=num_classes).to(DEVICE)
            cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location=DEVICE))
        cnn_results = test_cnn(cnn_model, test_loader, device=DEVICE)
        print("CNN results:", cnn_results)

    # --- DNN ---
    if TRAIN_DNN:
        print("\n--- Training DNN ---")
        dnn_model = DNNModel(num_classes=num_classes).to(DEVICE)
        dnn_model = train_dnn(dnn_model, train_loader, val_loader,
                              num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)
        torch.save(dnn_model.state_dict(), "dnn_model.pth")
        print("DNN model saved to dnn_model.pth")

    if TEST_DNN:
        print("\n--- Testing DNN ---")
        if not TRAIN_DNN:
            dnn_model = DNNModel(num_classes=num_classes).to(DEVICE)
            dnn_model.load_state_dict(torch.load("dnn_model.pth", map_location=DEVICE))
        dnn_results = test_dnn(dnn_model, test_loader, device=DEVICE)
        print("DNN results:", dnn_results)
