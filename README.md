# LongitudinalDBM_synthmorph
Longitudinal DBM pipeline implementation using FreeSurfer's Synthmorph

This is an implementation of the pipeline described in Germann et al., 2024 (https://www.biorxiv.org/content/10.1101/2024.08.12.607581v1.full.pdf)
of a longitudinal deformation based morphology pipeline, but relying on FreeSurfer's SynthMorph registration method (https://martinos.org/malte/synthmorph/), which is very fast.
These scripts rely on docker images from FreeSurfer and will need to be edited to accomodate which version of those docker images install. GPU not needed.

