#!/usr/bin/env python3
import requests
import bz2
import os

def download_file(url, output_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    percent = int(100 * downloaded / total_size) if total_size > 0 else 0
                    print(f"\rProgress: {percent}% ({downloaded / 1024 / 1024:.1f} MB)", end="")
        print("\nDownload complete!")
        return True
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return False

def extract_bz2(bz2_path, output_path):
    print(f"Extracting {bz2_path}...")
    with open(bz2_path, 'rb') as source, open(output_path, 'wb') as dest:
        dest.write(bz2.decompress(source.read()))
    print("Extraction complete!")

if __name__ == "__main__":
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = "shape_predictor_68_face_landmarks.dat.bz2"
    output_path = "shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(output_path):
        print(f"{output_path} already exists!")
    else:
        if not os.path.exists(bz2_path):
            if download_file(url, bz2_path):
                extract_bz2(bz2_path, output_path)
        else:
            print(f"{bz2_path} already exists, extracting...")
            extract_bz2(bz2_path, output_path)