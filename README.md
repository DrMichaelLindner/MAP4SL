
#   MAP4SL - Multiband Artefact Probe for Slice Leakage


MAP4SL is a Matlab tool which is designed to identify the effect of an artefact in MRI data on other areas depending on the slice leakage in MR multiband EPI functional images.

This tool calculates the possible artefact positions of a given seed region depending on the multislice acceleration factor (and GRAPPA if used). 
Then it creates a sphere or disk around this positions and extracts and plots the time courses of all voxels in these regions. 
Additionally, two independent control regions are extracted to compare with. Further this tool does all by all voxel
correlations between the voxels of the seed area and all voxels of other possible artefact and control areas.
The given seed region is always labeled as A and the expected artefact positions in other slices are B,C,D,... depending on the multislice acceleration factor.
If GRAPPA was used, each expected artefact positions has a second region in the same slice where the artefact can occur depending on the parallel imaging. They are labels A_g, B_g, etc.


## *USAGE*
You can use MAP4SL as a function giving the following inputs via a script in the following way:
MAP4SL(cfg,filename,outputfolder)
You can also start/run MAP4SL directly. In this case, when no input is given, MAP4SL will open three input dialogues to select the
data file and outputfolder and to specify the input values directly.


## *INPUT*
- filename - '<yourpath2BIDSsubjectfolder\niifilename>'
    The data needs to be in the BIDS format (at least nii.gz and a corresponding json file is needed) 
    
- cfg is a configuration structure with the following fields:
  cfg.seed                 Coordinate of the seed voxel (e.g. [69,83,14]);
  cfg.sphere_radius   Radius of a sphere or a disk around the estimated artefact positions
  cfg.disk                  'yes' if want a disk or no if you want a sphere around the estimated artefact positions
  cfg.corr_thresh       Threshold for thresholded images of all correlation coefficients rho                       
  cfg.saveprefix         Prefix of the outpuf files
  cfg.flipLR                Flip left and right of seed position ("yes" or "no")
  cfg.Shift_FOVdevX  CAIPI shift of your sequence in form of FieldOfView divided by this number (e.g. normally 3)

- outputfolder          The folder in which the output files will be saved;


## *OUTPUT*
MAP4SL provided various output files
- masks in nifti format containing:
      -- possible artefact regions (including seed): 
          single voxel 'prefix_artefact_mask_single_voxel.nii' and
          sphere/disk   'prefix_artefact_mask_conv.nii' 
      -- control regions 
          single voxel 'prefix_control_mask_single_voxel.nii' and
          sphere/disk   'prefix_control_mask_conv.nii' 
      -- only seed
          single voxel 'prefix_seed_mask_single_voxel.nii' and
          sphere/disk   'prefix_seed_mask_conv.nii' 
- text file with the a table of all artefact and control positions and their labels in the images. 
- Bitmaps showing all expected artefact (and projected control) positions
          in 2D and 3D
- MP4 movie showing all slices containing expected artefact positions over time.
- Bitmap of time courses of the seed and all possible artefact regions
- Bitmap of time courses of both control regions
- Bitmaps of correlations coefficients rho between seed and artefact and control positions(*_rho.bmp)
- Bitmaps of all absolute correlations coefficients (*_abs_rho.bmp)
- Bitmaps of all thresholded absolute correlations coefficients (*_corr_abs_rho_thresholded.bmp)
- Matlab .m file containing all correlation values (rho and p)
- Matlab .m file containing all time series (*_data_correlations.mat)


## *Example script:*
cfg.seed = [31,85,22];  
cfg.sphere_radius = 6;  
cfg.disk = 'yes';  
cfg.corr_thresh = 0.6;  
cfg.saveprefix = 'TEST';  
cfg.flipLR = 'no';  
cfg.Shift_FOVdevX = 3;  
inputfilename = 'D:\BIDSfolder\s001\func\sub-01_task-X_bold.nii.gz';  
outputfolder='D:\MAP4SL_output';  

MAP4SL(cfg, inputfilename, outputfolder)


## *DEPENDECIES*
MAP4SL needs the toolbox "Tools for nifti and analyze image" for 
Matlab in the MATLAB path. The toolbox can be downloaded from here:
https://uk.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image


## *Licence* 
MAP4SL by Michael Lindner is licensed under CC BY 4.0
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
  
  
  
Author:  
Michael Lindner  
University of Reading, 2018  
School of Psychology and Clinical Language Sciences  
Center for Integrative Neuroscience and Neurodynamics
