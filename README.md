# Longitudinal DBM Pipeline Relying on Synthmorph Registration
Longitudinal DBM pipeline implementation using FreeSurfer's Synthmorph

This is an implementation of the pipeline described in Germann et al., 2024 (https://www.biorxiv.org/content/10.1101/2024.08.12.607581v1.full.pdf)
of a longitudinal deformation based morphology pipeline, but relying on FreeSurfer's SynthMorph registration method (https://martinos.org/malte/synthmorph/), which is very fast.
These scripts rely on docker images from FreeSurfer and will need to be edited to accomodate which version of those docker images install. GPU not needed.

Usage: Idosyncracies to my own data setup are sure to exist. This script is rough and mostly for me. Nevertheless,

1. gatherandrun.py is used to orchestrate calls to 001_create_template.py to build a custom template for each participant.
2. 002_registertemplatestoMNI.py registers all the subject-specific templates straight to template. (later ill implement the study custom template creation featured in the Germann et al., pipeline.)
3. 003_create_absandrelative_Jacobians.py is used to create the absolute and relative Jacobians.
