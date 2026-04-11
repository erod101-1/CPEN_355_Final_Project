import kagglehub
import shutil
import os
import torch
from prepare_dataset import main as prepare_dataset_main, create_data_loaders
from CNN import CNNModel, train_cnn, test_cnn
from DNN import DNNModel, train_dnn, test_dnn
from resnet50 import ResNet50Model, train_resnet50, test_resnet50

# --- Flags ---
PREP_DATASET   = False   # Set to False to skip download/preparation and use existing data
TRAIN_CNN      = False   # Set to False to skip CNN training (loads cnn_model.pth instead)
TRAIN_DNN      = False   # Set to False to skip DNN training (loads dnn_model.pth instead)
TRAIN_RESNET50 = False   # Set to False to skip ResNet50 training (loads resnet50_model.pth instead)
TEST_CNN       = True   # Set to False to skip CNN evaluation
TEST_DNN       = True   # Set to False to skip DNN evaluation
TEST_RESNET50  = True   # Set to False to skip ResNet50 evaluation
SHOW_GRID      = False   # Set to False to skip the sample image grid after evaluation

# --- Hyperparameters ---
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-3
PREPARED_DATASET_DIR = "prepared_dataset"
MODELS_DIR           = "models"


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

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

    if any([TRAIN_CNN, TRAIN_DNN, TRAIN_RESNET50, TEST_CNN, TEST_DNN, TEST_RESNET50]):
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
        torch.save(cnn_model.state_dict(), f"{MODELS_DIR}/cnn_model.pth")
        print(f"CNN model saved to {MODELS_DIR}/cnn_model.pth")

    if TEST_CNN:
        print("\n--- Testing CNN ---")
        if not TRAIN_CNN:
            cnn_model = CNNModel(num_classes=num_classes).to(DEVICE)
            cnn_model.load_state_dict(torch.load(f"{MODELS_DIR}/cnn_model.pth", map_location=DEVICE))
        cnn_results = test_cnn(cnn_model, test_loader, device=DEVICE, class_names=classes, show_grid=SHOW_GRID)
        print("CNN results:", cnn_results)

    # --- DNN ---
    if TRAIN_DNN:
        print("\n--- Training DNN ---")
        dnn_model = DNNModel(num_classes=num_classes).to(DEVICE)
        dnn_model = train_dnn(dnn_model, train_loader, val_loader,
                              num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)
        torch.save(dnn_model.state_dict(), f"{MODELS_DIR}/dnn_model.pth")
        print(f"DNN model saved to {MODELS_DIR}/dnn_model.pth")

    if TEST_DNN:
        print("\n--- Testing DNN ---")
        if not TRAIN_DNN:
            dnn_model = DNNModel(num_classes=num_classes).to(DEVICE)
            dnn_model.load_state_dict(torch.load(f"{MODELS_DIR}/dnn_model.pth", map_location=DEVICE))
        dnn_results = test_dnn(dnn_model, test_loader, device=DEVICE, class_names=classes, show_grid=SHOW_GRID)
        print("DNN results:", dnn_results)

    # --- ResNet50 ---
    if TRAIN_RESNET50:
        print("\n--- Training ResNet50 ---")
        resnet50_model = ResNet50Model(num_classes=num_classes).to(DEVICE)
        resnet50_model = train_resnet50(resnet50_model, train_loader, val_loader,
                                        num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)
        torch.save(resnet50_model.state_dict(), f"{MODELS_DIR}/resnet50_model.pth")
        print(f"ResNet50 model saved to {MODELS_DIR}/resnet50_model.pth")

    if TEST_RESNET50:
        print("\n--- Testing ResNet50 ---")
        if not TRAIN_RESNET50:
            resnet50_model = ResNet50Model(num_classes=num_classes).to(DEVICE)
            resnet50_model.load_state_dict(torch.load(f"{MODELS_DIR}/resnet50_model.pth", map_location=DEVICE))
        resnet50_results = test_resnet50(resnet50_model, test_loader, device=DEVICE, class_names=classes, show_grid=SHOW_GRID)
        print("ResNet50 results:", resnet50_results)
