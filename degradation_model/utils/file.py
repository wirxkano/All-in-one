import gdown
import zipfile
import os


def download_file_from_google_drive(
    drive_url, output_path, extract=False, extract_dir=None
):
    print("Downloading file...")
    gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
    print(f"Downloaded '{output_path}' successfully.")

    if extract and extract_dir:
        os.makedirs(extract_dir, exist_ok=True)

        print(f"Extracting '{output_path}' to '{extract_dir}'...")
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        print("Extraction complete!")


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    print(f"Folder '{folder_path}' has been zipped into '{output_path}'.")


if __name__ == "__main__":
    folder_to_zip = "benchmarks"
    output_zip = "SRbenchmarks.zip"
    zip_folder(folder_to_zip, output_zip)
