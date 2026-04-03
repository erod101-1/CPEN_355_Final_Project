
import kagglehub
import shutil
import os
from prepare_dataset import main as prepare_dataset_main

prep_dataset: bool = True  # Set to False to skip dataset preparation and use existing data


if __name__ == "__main__":

    if prep_dataset:
        # Download latest version (downloads to kagglehub cache)
        path = kagglehub.dataset_download("luischuquimarca/healthy-and-defective-fruits")

        # Copy to current directory
        current_dir = os.getcwd()
        dataset_name = "healthy-and-defective-fruits"
        destination = os.path.join(current_dir, dataset_name)

        if os.path.exists(destination):
            shutil.rmtree(destination)

        shutil.copytree(path, destination)

        print("Path to dataset files:", path)
        print("Copied to current directory:", destination)
        prepare_dataset_main()