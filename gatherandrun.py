#!/usr/bin/env python3
"""
Script Name: run_create_templates.py

Description:
    This script reads a CSV file (`keep.csv`) containing information about subject images,
    extracts unique subject IDs (`subj_id`), and for each subject:
        1. Creates an output directory named after the `subj_id` in `/mnt/colony/dbm/out`.
        2. Gathers all associated image filenames, appends `.nii.gz` to each, and verifies their existence.
        3. Copies the verified image files from the input directory to the subject's output directory.
        4. Invokes `create_template.py` to perform template creation using the copied images.

Usage:
    python run_create_templates.py

    Ensure that `keep.csv` is in the same directory as this script or provide the correct path.
"""

import os
import sys
import pandas as pd
import subprocess
import shutil  # Added for copying files

def main():
    # Define paths
    csv_file = 'keep.csv'  # Path to the CSV file
    input_dir = '/mnt/colony/dbm/in'  # Input directory containing subject images
    resources_dir = '/mnt/colony/dbm/resources'  # Resources directory containing the MNI template
    base_output_dir = '/mnt/colony/dbm/out'  # Base output directory for all templates
    create_template_script = '001_create_template.py'  # Path to the template creation script

    # Verify CSV file exists
    if not os.path.isfile(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        sys.exit(1)

    # Check required columns in CSV
    required_columns = {'image', 'subj_id'}
    if not required_columns.issubset(df.columns):
        print(f"Error: CSV file must contain the following columns: {required_columns}")
        sys.exit(1)

    # Extract unique subj_id values
    unique_subj_ids = df['subj_id'].unique()
    print(f"Found {len(unique_subj_ids)} unique subject IDs.")

    # Verify input_dir exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Verify resources_dir exists
    if not os.path.isdir(resources_dir):
        print(f"Error: Resources directory '{resources_dir}' does not exist.")
        sys.exit(1)

    # Verify create_template.py exists
    if not os.path.isfile(create_template_script):
        print(f"Error: Template creation script '{create_template_script}' not found.")
        sys.exit(1)

    # Iterate over each unique subj_id
    for subj_id in unique_subj_ids:
        print(f"\nProcessing Subject ID: {subj_id}")

        # Step 1: Create Output Directory
        output_dir = os.path.join(base_output_dir, subj_id)
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            continue  # Skip to the next subject

        # Step 2: Gather Associated Images
        subj_images_df = df[df['subj_id'] == subj_id]
        image_base_names = subj_images_df['image'].tolist()
        image_files = [f"{image_name}.nii.gz" for image_name in image_base_names]

        # Verify existence of each image file in input_dir
        missing_images = [img for img in image_files if not os.path.isfile(os.path.join(input_dir, img))]
        if missing_images:
            print(f"Warning: The following images for subject '{subj_id}' are missing in '{input_dir}':")
            for img in missing_images:
                print(f"  - {img}")
            # Proceed with available images
            image_files = [img for img in image_files if img not in missing_images]
            if not image_files:
                print(f"No valid images found for subject '{subj_id}'. Skipping.")
                continue

        print(f"Found {len(image_files)} images for subject '{subj_id}'.")

        # Step 3: Copy Images to Output Directory
        print("Copying images to the output directory...")
        for img in image_files:
            src_path = os.path.join(input_dir, img)
            dest_path = os.path.join(output_dir, img)
            try:
                shutil.copy(src_path, dest_path)
                print(f"Copied '{img}' to '{output_dir}'.")
            except Exception as e:
                print(f"Error copying '{img}' to '{output_dir}': {e}")
                # Optionally, decide to skip this image or continue
                # Here, we continue with other images
                continue

        # Step 4: Invoke create_template.py
        # Construct the command:
        # python create_template.py /mnt/colony/dbm/out/{subj_id} img1.nii.gz img2.nii.gz ... /mnt/colony/dbm/resources --output_dir /mnt/colony/dbm/out/{subj_id}

        command = [
            'python',
            create_template_script,
            output_dir,  # data_dir
            *image_files,  # subject_images
            resources_dir,  # resources_dir
            '--output_dir', output_dir  # output_dir
        ]

        print(f"Executing command: {' '.join(command)}")

        try:
            # Execute the command
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"Template creation for subject '{subj_id}' completed successfully.")
            print("Output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error during template creation for subject '{subj_id}'.")
            print("Standard Output:")
            print(e.stdout)
            print("Standard Error:")
            print(e.stderr)
            # Optionally, continue to the next subject or exit
            continue

    print("\nAll subjects have been processed.")

if __name__ == "__main__":
    main()

