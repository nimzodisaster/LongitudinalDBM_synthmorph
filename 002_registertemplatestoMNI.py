import os
import sys
import argparse
import time
import shutil  # Added for file operations
import numpy as np
import nibabel as nib
import fsl.transform.flirt as fsl_transform  # Import fslpy transform module
from fsl.wrappers import fslmaths, fslstats
import docker
import ants
import datetime
import random
import string
allowed_chars = string.ascii_letters + string.digits
random_string = ''.join(random.choices(allowed_chars, k=8))
print(random_string)
start_time = time.time()
# Initialize Docker client
client = docker.from_env()

# Define global constants for Docker images and container names
SYNTHMORPH_DOCKER_IMAGE = 'freesurfer/synthmorph:latest'
FREESURFER_DOCKER_IMAGE = 'jkilee/freesurfer_app:7.4.1.licensed'
SYNTHMORPH_CONTAINER_NAME = f'synthmorph_container-{random_string}'
FREESURFER_CONTAINER_NAME = 'freesurfer_container-{random_string}'
MNI_FILENAME = "MNI152_T1_1mm_brain.nii.gz"  # Filename of the MNI template
jj = 32  # CPU cores

def run_command(cmd):
    """Run a shell command."""
    print(f"Running command: {cmd}")
    result = os.system(cmd)
    if result != 0:
        print(f"Error running command: {cmd}")
        sys.exit(1)

def run_command_in_container(cmd, container_name, setup_command=None, env=None):
    """Run a command inside a running docker container."""
    try:
        container = client.containers.get(container_name)
        if setup_command:
            full_cmd = f"{setup_command} && {cmd}"
        else:
            full_cmd = cmd
        print(f"Running command in {container_name}: {full_cmd}")
        # Execute the command with environment variables
        exec_result = container.exec_run(
            cmd=['/bin/bash', '-c', full_cmd],
            stderr=True,
            stdout=True,
            environment=env
        )
        exit_code = exec_result.exit_code
        output = exec_result.output.decode()
        if exit_code != 0:
            print(f"Error running command in container {container_name}: {full_cmd}\n{output}")
            sys.exit(1)
        else:
            print(output)
    except docker.errors.NotFound:
        print(f"Container {container_name} not found.")
        sys.exit(1)

def average_images(image_list, output_image):
    """Average a list of images and save the result."""
    imgs = [nib.load(img) for img in image_list]
    data = np.mean([img.get_fdata() for img in imgs], axis=0)
    avg_img = nib.Nifti1Image(data, imgs[0].affine)
    nib.save(avg_img, output_image)
    print(f"Averaged image saved to {output_image}")

def average_affine_transforms(affine_transform_list, output_lta, src_image, trg_image):
    """Average a list of affine transforms and save the result."""
    fsl_matrices = []
    for lta_file in affine_transform_list:
        # Convert LTA to FSL format
        fsl_mat = os.path.splitext(lta_file)[0] + '.mat'
        
        # **Corrected Path:** LTA files are in /data, not /input or /output
        lta_file_docker = f'/data/{os.path.basename(lta_file)}'
        fsl_mat_docker = f'/data/{os.path.basename(fsl_mat)}'

        cmd = f'lta_convert --inlta {lta_file_docker} --outfsl {fsl_mat_docker}'
        run_command_in_container(
            cmd,
            FREESURFER_CONTAINER_NAME,
            setup_command='source $FREESURFER_HOME/SetUpFreeSurfer.sh',
            env={'FSLOUTPUTTYPE': 'NIFTI_GZ'}
        )

        # Read FSL matrix using fslpy
        matrix = fsl_transform.readFlirt(fsl_mat)
        fsl_matrices.append(matrix)

    # Compute average matrix
    avg_matrix = sum(fsl_matrices) / len(fsl_matrices)

    # Save average matrix to FSL format using fslpy
    avg_fsl_mat = os.path.join(os.path.dirname(output_lta), 'average_affine.mat')
    fsl_transform.writeFlirt(avg_matrix, avg_fsl_mat)

    # **Corrected Paths:**
    # - Source Image is in /data
    # - Target Image is in /data
    src_image_docker = f'/data/{os.path.basename(src_image)}'
    trg_image_docker = f'/data/{os.path.basename(trg_image)}'
    avg_fsl_mat_docker = f'/data/{os.path.basename(avg_fsl_mat)}'
    output_lta_docker = f'/data/{os.path.basename(output_lta)}'

    cmd = (
        f'lta_convert --infsl {avg_fsl_mat_docker} --outlta {output_lta_docker} '
        f'--src {src_image_docker} --trg {trg_image_docker}'
    )
    run_command_in_container(
        cmd,
        FREESURFER_CONTAINER_NAME,
        setup_command='source $FREESURFER_HOME/SetUpFreeSurfer.sh',
        env={'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    )
    print(f"Averaged affine transform saved to {output_lta}")

def average_deformation_fields(deformation_field_list, output_field):
    """Average deformation fields and save the result."""
    fields = [nib.load(fld).get_fdata() for fld in deformation_field_list]
    avg_field_data = np.mean(fields, axis=0)
    affine = nib.load(deformation_field_list[0]).affine
    avg_field_img = nib.Nifti1Image(avg_field_data, affine)
    nib.save(avg_field_img, output_field)
    print(f"Averaged deformation field saved to {output_field}")

def scale_deformation_field(deformation_field, gradient_step, output_field):
    """Scale a deformation field by a gradient step."""
    img = nib.load(deformation_field)
    scaled_data = img.get_fdata() * gradient_step
    scaled_img = nib.Nifti1Image(scaled_data, img.affine)
    nib.save(scaled_img, output_field)
    print(f"Scaled deformation field saved to {output_field}")

def run_synthmorph_registration(subj_image, fixed_image, output_transform, model, init_transform=None, symmetric=False,outputname=None,extent=256,currdir=None):
    """
    Run SynthMorph registration.
    """
    basecurrdir = os.path.basename(currdir)
    subj_image_docker = f'/data/{basecurrdir}/{os.path.basename(subj_image)}'
    fixed_image_docker = f'/data/{basecurrdir}/{os.path.basename(fixed_image)}'
    output_transform_docker = f'/data/{basecurrdir}/{os.path.basename(output_transform)}'
    output_tomni_docker = f'/data/{basecurrdir}/{os.path.basename(outputname)}'
    cmd = f'mri_synthmorph register -m {model} -t {output_transform_docker} -e {extent} {subj_image_docker} {fixed_image_docker} '
    if model in ["rigid", "affine"]:
        cmd += ' -g'  # Use GPU for the registrations
    if init_transform:
        init_transform_docker = f'/data/{os.path.basename(init_transform)}'
        cmd += f' -i {init_transform_docker}'

    if symmetric and model == 'deform':
        cmd += ' -M'
    if outputname:
        cmd += f' -o {output_tomni_docker}'
    run_command_in_container(cmd, SYNTHMORPH_CONTAINER_NAME)

def invert_affine_transform(affine_lta, output_lta):
    """Invert an affine transform using lta_convert."""
    affine_lta_docker = f'/data/{os.path.basename(affine_lta)}'
    output_lta_docker = f'/data/{os.path.basename(output_lta)}'
    cmd = f'lta_convert --inlta {affine_lta_docker} --outlta {output_lta_docker} --invert'
    run_command_in_container(
        cmd,
        FREESURFER_CONTAINER_NAME,
        setup_command='source $FREESURFER_HOME/SetUpFreeSurfer.sh',
        env={'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    )
    print(f"Inverted affine transform saved to {output_lta}")

def apply_transforms_to_image(input_image, transform_list, output_image):
    """Apply a list of transforms to an image using mri_synthmorph."""
    input_image_docker = f'/data/{os.path.basename(input_image)}'
    output_image_docker = f'/data/{os.path.basename(output_image)}'
    # Prepare transform arguments
    transform_args = ''
    for transform in transform_list:
        transform_docker = f'/data/{os.path.basename(transform)}'
        transform_args += f' {transform_docker}'
    cmd = f'mri_synthmorph apply {transform_args} {input_image_docker} {output_image_docker}'
    run_command_in_container(cmd, SYNTHMORPH_CONTAINER_NAME)
    print(f"Applied transforms and saved aligned image to {output_image}")

def main(output_dir='output'):
    """Main function to create the unbiased template."""
    # Start the Docker containers
    print("Starting Docker containers")
    try:
        # Start the synthmorph_container
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
        synthmorph_container = client.containers.run(
            SYNTHMORPH_DOCKER_IMAGE,
            name={SYNTHMORPH_CONTAINER_NAME},
            entrypoint=["tail", "-f", "/dev/null"],
            detach=True,
            device_requests=device_requests,
            volumes={
                os.path.abspath(output_dir): {'bind': '/data', 'mode': 'rw'},  # Mount data_dir as read-write
            },
        )
        # Wait for containers to be fully up and running
        for container_name in [SYNTHMORPH_CONTAINER_NAME]:
            print(f"Waiting for {container_name} to be running...")
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    container = client.containers.get(container_name)
                    container.reload()
                    if container.status == 'running':
                        print(f"{container_name} is running.")
                        break
                    else:
                        time.sleep(1)
                except docker.errors.NotFound:
                    print(f"Container {container_name} not found.")
                    sys.exit(1)
            else:
                print(f"Error: {container_name} did not start properly.")
                sys.exit(1)
        #MAIN FUNCTION
        subdirectories = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,d))]
        total_subdirs = len(subdirectories)
        for i, subdir in enumerate(subdirectories,start=1):
            now = datetime.datetime.now()
            current_time =  now.strftime("%H:%M:%S")
            print("Current Time = ", current_time)
            currdir = os.path.join(output_dir,subdir)
            subj_template=os.path.join(currdir,"template_final.nii.gz")
            MNI_template=os.path.join(currdir,"MNI152_T1_1mm_brain.nii.gz")
            def_field = os.path.join(currdir, f'finaltemplate_to_MNI_deform.nii.gz')
            outputname = os.path.join(currdir, f'finaltemplate_to_MNI_space.nii.gz')
            if os.path.exists(outputname): continue
            # Check if required files exist
            if not os.path.exists(subj_template):
                error_message = f"Missing subject template file: {subj_template}\n"
                error_filename = os.path.join(currdir, "missing_files_error.log")
                with open(error_filename, "a") as error_file:
                    error_file.write(error_message)
                print(error_message)
                continue
                # Perform deformable registration without init_transform and symmetric=False
            try:
                run_synthmorph_registration(
                    subj_image=subj_template,
                    fixed_image=MNI_template,
                    output_transform=def_field,
                    model='joint',
                    init_transform=None,
                    symmetric=False,
                    outputname=outputname,
                    extent=256,
                    currdir=currdir
                )
            except Exception as e:
                error_message = f"Error occurred during registration for {subj_template}:\n{str(e)}\n"
                error_filename = f"{subj_template}_error.log"
                with open(error_filename, "w") as error_file:
                    error_file.write(error_message)
            end_time = datetime.datetime.now()
            timediff = end_time - now
            print(f"{i} of {total_subdirs} finished")
            print("Processing Time: ", timediff)
    finally:
        # Stop the Docker containers
        print("Stopping Docker containers")
        for container_name in [SYNTHMORPH_CONTAINER_NAME, FREESURFER_CONTAINER_NAME]:
            try:
                container = client.containers.get(container_name)
                container.stop()
                container.remove()
                print(f"Container {container_name} stopped and removed.")
            except docker.errors.NotFound:
                print(f"Container {container_name} does not exist or has already been removed.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Template Creation Time elapsed: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register_templates to MNI")
    parser.add_argument(
        "--output_dir", default="", help="Directory to store the output templates and intermediate files."
    )
    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
    )

