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
random_string = ''.join(random.choices(allowed_chars, k=8))
SYNTHMORPH_CONTAINER_NAME = f'synthmorph_container-{random_string}'
FREESURFER_CONTAINER_NAME = f'freesurfer_container-{random_string}'
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

def run_synthmorph_registration(subj_image, fixed_image, output_transform, model, init_transform=None, symmetric=False):
    """
    Run SynthMorph registration.
    """
    subj_image_docker = f'/data/{os.path.basename(subj_image)}'
    fixed_image_docker = f'/data/{os.path.basename(fixed_image)}'
    output_transform_docker = f'/data/{os.path.basename(output_transform)}'

    cmd = f'mri_synthmorph register -m {model} -t {output_transform_docker} {subj_image_docker} {fixed_image_docker}'
    if model in ["rigid", "affine"]:
        cmd += ' -g'  # Use GPU for the registrations
    if init_transform:
        init_transform_docker = f'/data/{os.path.basename(init_transform)}'
        cmd += f' -i {init_transform_docker}'

    if symmetric and model == 'deform':
        cmd += ' -M'

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

def main(subject_images, data_dir, resources_dir, num_iterations=3, gradient_step=0.2, blending_weight=0.75, output_dir='output'):
    """Main function to create the unbiased template."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # **Copy the MNI template to the data directory**
    mni_source = os.path.join(resources_dir, MNI_FILENAME)
    mni_destination = os.path.join(data_dir, MNI_FILENAME)
    if not os.path.isfile(mni_source):
        print(f"Error: MNI template '{MNI_FILENAME}' not found in resources directory '{resources_dir}'.")
        sys.exit(1)
    shutil.copy(mni_source, mni_destination)
    print(f"MNI template copied to {mni_destination}")

    # **Define the path to the MNI template within the data directory**
    reference_image = mni_destination  # Uses the copied MNI template
    reference_basename = os.path.splitext(os.path.basename(reference_image))[0]

    # Start the Docker containers
    print("Starting Docker containers")
    try:
        # Start the synthmorph_container
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
        synthmorph_container = client.containers.run(
            SYNTHMORPH_DOCKER_IMAGE,
            name=SYNTHMORPH_CONTAINER_NAME,
            entrypoint=["tail", "-f", "/dev/null"],
            detach=True,
            device_requests=device_requests,
            volumes={
                os.path.abspath(data_dir): {'bind': '/data', 'mode': 'rw'},  # Mount data_dir as read-write
            },
        )

        # Start the freesurfer_container
        freesurfer_container = client.containers.run(
            FREESURFER_DOCKER_IMAGE,
            name=FREESURFER_CONTAINER_NAME,
            entrypoint="bash",
            command=["-c", "export FREESURFER_HOME=/usr/local/freesurfer && source $FREESURFER_HOME/SetUpFreeSurfer.sh && tail -f /dev/null"],
            environment={'FSLOUTPUTTYPE': 'NIFTI_GZ'},
            detach=True,
            volumes={
                os.path.abspath(data_dir): {'bind': '/data', 'mode': 'rw'},  # Mount data_dir as read-write
            },
        )

        # Wait for containers to be fully up and running
        for container_name in [SYNTHMORPH_CONTAINER_NAME, FREESURFER_CONTAINER_NAME]:
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

        if len(subject_images) == 1:
            # Single image case: rigidly register to MNI and save as final template
            print("Single input image detected. Performing rigid registration to MNI.")

            subj = subject_images[0]
            subj_basename = os.path.splitext(os.path.basename(subj))[0]
            rigid_lta = os.path.join(data_dir, f'{subj_basename}_to_{reference_basename}_rigid.lta')
            aligned_subj = os.path.join(data_dir, f'{subj_basename}_rigid_aligned_to_MNI.nii.gz')

            # Perform rigid registration
            run_synthmorph_registration(
                subj_image=os.path.join(data_dir, subj),
                fixed_image=reference_image,  # Uses the copied MNI template
                output_transform=rigid_lta,
                model='rigid',
                init_transform=None,
                symmetric=False
            )

            # Apply rigid transform
            apply_transforms_to_image(os.path.join(data_dir, subj), [rigid_lta], aligned_subj)

            # Save the aligned image as the final template
            final_template = os.path.join(data_dir, 'final_template.nii.gz')
            os.rename(aligned_subj, final_template)
            print(f"Final template created at {final_template}")

        else:
            # Multiple images case: proceed with iterative template building
            print("Multiple input images detected. Proceeding with iterative template building.")

            # Step 1: Initial Rigid Registration

            print("Step 1: Initial Rigid Registration")

            # Create a list to store rigid transforms
            rigid_transforms = []

            # Rigidly register all subjects to the MNI reference
            for subj in subject_images:
                subj_basename = os.path.splitext(os.path.basename(subj))[0]
                rigid_lta = os.path.join(data_dir, f'{subj_basename}_to_{reference_basename}_rigid.lta')
                run_synthmorph_registration(
                    subj_image=os.path.join(data_dir, subj),
                    fixed_image=reference_image,  # Uses the copied MNI template
                    output_transform=rigid_lta,
                    model='rigid',
                    init_transform=None,
                    symmetric=False
                )
                rigid_transforms.append(rigid_lta)

            # Apply rigid transforms to all subjects
            print("Applying rigid transforms to all subjects")
            aligned_images = []
            for subj, rigid_lta in zip(subject_images, rigid_transforms):
                subj_basename = os.path.splitext(os.path.basename(subj))[0]
                aligned_subj = os.path.join(data_dir, f'{subj_basename}_rigid_aligned_to_MNI.nii.gz')
                if rigid_lta is not None:
                    apply_transforms_to_image(os.path.join(data_dir, subj), [rigid_lta], aligned_subj)
                else:
                    # Reference image remains unchanged (Not applicable here since all are registered to MNI)
                    aligned_subj = os.path.join(data_dir, subj)
                aligned_images.append(aligned_subj)

            # Step 2: Initialize the template by averaging the rigidly aligned images
            print("Step 2: Initialize the template by averaging the rigidly aligned images")
            template_prev = os.path.join(data_dir, 'template0.nii.gz')
            average_images(aligned_images, template_prev)

            # Step 3: Iterative Template Building
            for iter_num in range(1, num_iterations + 1):
                print(f"\n=== Iteration {iter_num} ===")
                xavgNew_images = []
                affine_transforms = []
                deformation_fields = []

                target_template_num = iter_num - 1  # Since we are registering to template_{iter_num - 1}

                # Affine Registration
                print("Affine Registration")
                for subj in subject_images:
                    subj_basename = os.path.splitext(os.path.basename(subj))[0]
                    aff_lta = os.path.join(data_dir, f'{subj_basename}_to_template{target_template_num}_affine.lta')
                    run_synthmorph_registration(
                        subj_image=os.path.join(data_dir, subj),
                        fixed_image=template_prev,
                        output_transform=aff_lta,
                        model='affine',
                        init_transform=None,
                        symmetric=False
                    )
                    affine_transforms.append(aff_lta)

                # Apply affine transforms to all subjects
                print("Applying affine transforms to all subjects")
                affine_aligned_images = []
                for subj, aff_lta in zip(subject_images, affine_transforms):
                    subj_basename = os.path.splitext(os.path.basename(subj))[0]
                    aligned_subj = os.path.join(data_dir, f'{subj_basename}_affine_aligned_to_template{target_template_num}.nii.gz')
                    apply_transforms_to_image(os.path.join(data_dir, subj), [aff_lta], aligned_subj)
                    affine_aligned_images.append(aligned_subj)

                # Average affine transforms
                print("Averaging affine transforms")
                avg_affine_lta = os.path.join(data_dir, f'average_affine_to_template{target_template_num}.lta')
                average_affine_transforms(
                    affine_transform_list=affine_transforms,
                    output_lta=avg_affine_lta,
                    src_image=os.path.join(data_dir, subject_images[0]),
                    trg_image=template_prev
                )

                # Nonlinear Registration
                print("Nonlinear Registration")
                for subj_aligned, subj in zip(affine_aligned_images, subject_images):
                    subj_basename = os.path.splitext(os.path.basename(subj))[0]
                    def_field = os.path.join(data_dir, f'{subj_basename}_to_template{target_template_num}_deform.nii.gz')
                    # Perform deformable registration without init_transform and symmetric=False
                    run_synthmorph_registration(
                        subj_image=subj_aligned,
                        fixed_image=template_prev,
                        output_transform=def_field,
                        model='deform',
                        init_transform=None,
                        symmetric=False
                    )
                    deformation_fields.append(def_field)

                if iter_num != num_iterations:
                    # Apply deformation fields to the affine aligned images
                    print("Applying deformation fields to affine aligned images")
                    for subj_aligned, def_field in zip(affine_aligned_images, deformation_fields):
                        subj_basename = os.path.splitext(os.path.basename(subj_aligned))[0].replace('_affine_aligned_to_template', '')
                        subj_warped = os.path.join(data_dir, f'{subj_basename}_warped_to_template{target_template_num}.nii.gz')
                        apply_transforms_to_image(subj_aligned, [def_field], subj_warped)
                        xavgNew_images.append(subj_warped)

                    # Average the warped images to create xavgNew
                    print("Averaging warped images to create xavgNew")
                    xavgNew = os.path.join(data_dir, f'xavgNew_{iter_num}.nii.gz')
                    average_images(xavgNew_images, xavgNew)

                    # Invert average affine transform
                    print("Inverting average affine transform")
                    inv_avg_affine_lta = os.path.join(data_dir, f'inverse_average_affine_to_template{target_template_num}.lta')
                    invert_affine_transform(avg_affine_lta, inv_avg_affine_lta)

                    # Apply inverse average affine transform to xavgNew
                    print("Applying inverse average affine transform to xavgNew")
                    xavgNew_inv_affine = os.path.join(data_dir, f'xavgNew_inv_affine_{iter_num}.nii.gz')
                    apply_transforms_to_image(xavgNew, [inv_avg_affine_lta], xavgNew_inv_affine)

                    # Average deformation fields
                    print("Averaging deformation fields")
                    avg_deformation_field = os.path.join(data_dir, f'average_deform_to_template{target_template_num}.nii.gz')
                    average_deformation_fields(deformation_fields, avg_deformation_field)

                    # Scale the average deformation field by negative gradient_step
                    print("Scaling the average deformation field by negative gradient_step")
                    scaled_deformation_field = os.path.join(data_dir, f'scaled_deform_{iter_num}.nii.gz')
                    scale_deformation_field(avg_deformation_field, -gradient_step, scaled_deformation_field)

                    # Apply scaled deformation field to xavgNew_inv_affine to get the new template
                    template_curr = os.path.join(data_dir, f'template{iter_num}.nii.gz')

                    print("Updating template")
                    # Apply scaled deformation field to update the template
                    apply_transforms_to_image(xavgNew_inv_affine, [scaled_deformation_field], template_curr)

                                            # In the main function, during the blending and sharpening step
                    if blending_weight < 1.0:
                        # Load the template image using ants
                        template_image = ants.image_read(template_curr)
                        # Apply sharpening
                        sharpened_image = ants.iMath(template_image, "Sharpen")
                        # Blend the images
                        blended_image = (template_image * blending_weight) + (sharpened_image * (1.0 - blending_weight))
                        # Save the blended image to a temporary file first
                        temp_template_curr = template_curr + '.temp.nii.gz'
                        ants.image_write(blended_image, temp_template_curr)
                        # Rename the temporary file to the final output file
                        os.replace(temp_template_curr, template_curr)
                        print("Blending and sharpening the new template")


                    # Update template_prev for next iteration
                    template_prev = template_curr
                else:
                    # Final iteration: use joint registration
                    print("Final iteration: generating final template with joint registration")
                    # Step 1: Use run_synthmorph_registration from the original input images to the template using model='joint'
                    joint_deformation_fields = []
                    final_warped_images = []

                    for subj in subject_images:
                        subj_basename = os.path.splitext(os.path.basename(subj))[0]
                        deform_field = os.path.join(data_dir, f'{subj_basename}_to_template_final_deform.nii.gz')
                        run_synthmorph_registration(
                            subj_image=os.path.join(data_dir, subj),
                            fixed_image=template_prev,
                            output_transform=deform_field,
                            model='joint',
                            init_transform=None,
                            symmetric=False
                        )
                        joint_deformation_fields.append(deform_field)

                    # Step 2: Apply the warps to the input images from step 1
                    print("Applying joint deformation fields to input images")
                    for subj, deform_field in zip(subject_images, joint_deformation_fields):
                        subj_basename = os.path.splitext(os.path.basename(subj))[0]
                        warped_image = os.path.join(data_dir, f'{subj_basename}_warped_to_template_final.nii.gz')
                        apply_transforms_to_image(os.path.join(data_dir, subj), [deform_field], warped_image)
                        final_warped_images.append(warped_image)

                    # Step 3: Average the transformed images. Name this image as template_final.nii.gz
                    print("Averaging final warped images to create template_final")
                    final_template = os.path.join(data_dir, 'template_final.nii.gz')
                    average_images(final_warped_images, final_template)

            if len(subject_images) > 1:
                print(f"\nFinal template is {final_template}")

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
    parser = argparse.ArgumentParser(description="Create an unbiased template using iterative registration.")
    parser.add_argument(
        "data_dir",
        help="Path to the data directory containing subject images and to store output files.",
    )
    parser.add_argument(
        "subject_images",
        nargs="+",
        help="Filenames of the subject images to include in the template creation. These should be located within data_dir.",
    )
    parser.add_argument(
        "resources_dir",
        help="Path to the resources directory containing the MNI template.",
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of template refinement iterations."
    )
    parser.add_argument(
        "--gradient_step", type=float, default=0.25, help="Gradient step for deformation field scaling."
    )
    parser.add_argument(
        "--blending_weight", type=float, default=0.75, help="Weight for blending and sharpening the template."
    )
    parser.add_argument(
        "--output_dir", default="output", help="Directory to store the output templates and intermediate files."
    )

    args = parser.parse_args()

    main(
        subject_images=args.subject_images,
        data_dir=args.data_dir,
        resources_dir=args.resources_dir,
        num_iterations=args.iterations,
        gradient_step=args.gradient_step,
        blending_weight=args.blending_weight,
        output_dir=args.output_dir,
    )

