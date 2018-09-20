function MAP4SL(cfg,filename,outputfolder)

% MAP4SL - Multiband Artefact Probe for Slice Leakage
% 
% MAP4SL is a Matlab bases tool which is designed to identify the effect of
% an artefact in MRI data on other areas depending on the slice leakage in 
% MR multiband EPI functional images.
% 
% This tool calculates the possible artefact positions of a given seed 
% region depending on the multislice acceleration factor (and GRAPPA if 
% used). Then it creates a sphere or disk around this positions and 
% extracts and plots the time courses of all voxels in these regions. 
% Additionally, two independent control regions are extracted to compare 
% with. Further this tool does all by all voxel correlations between the 
% voxels of the seed area and all voxels of other possible artefact and 
% control areas. 
% The given seed region is always labeled as A and the expected artefact 
% positions in other slices are B,C,D,... depending on the multislice 
% acceleration factor. If GRAPPA was used, each expected artefact positions 
% has a second region in the same slice where the artefact can occur 
% depending on the parallel imaging. They are labels A_g, B_g, etc.
% 
% 
% USAGE
% You can use MAP4SL as a function giving the following inputs via a script
% in the following way:
%                   MAP4SL(cfg,filename,outputfolder)
% You can also start/run MAP4SL directly. In this case, when no input is 
% given, MAP4SL will open input dialogues to select the data file and 
% outputfolder and to specify the input values directly.
% 
% 
% INPUT
% - filename - '<yourpath2BIDSsubjectfolder\niifilename>'
%       The data needs to be in the BIDS format (at least nii.gz and a 
%       corresponding json file is needed) 
%     
% - cfg is a configuration structure with the following fields:
%   cfg.seed           Coordinate of the seed voxel (e.g. [69,83,14]);
%   cfg.sphere_radius  Radius of a sphere or a disk around the estimated 
%                      artefact positions
%   cfg.disk           'yes' if want a disk or no if you want a sphere 
%                      around the estimated artefact positions
%   cfg.corr_thresh    Threshold for thresholded images of all correlation
%                      coefficients rho                       
%   cfg.saveprefix     Prefix of the outpuf files
%   cfg.flipLR         Flip left and right of seed position ("yes" or "no")
%   cfg.Shift_FOVdevX  CAIPI shift of your sequence in form of FieldOfView 
%                      divided by this number (e.g. normally 3)
% 
% - outputfolder       The folder in which the output files will be saved;
% 
% 
% OUTPUT
% MAP4SL provided various output files
% - masks in nifti format containing:
%       -- possible artefact regions (including seed): 
%           single voxel 'prefix_artefact_mask_single_voxel.nii' and
%           sphere/disk   'prefix_artefact_mask_conv.nii' 
%       -- control regions 
%           single voxel 'prefix_control_mask_single_voxel.nii' and
%           sphere/disk   'prefix_control_mask_conv.nii' 
%       -- only seed
%           single voxel 'prefix_seed_mask_single_voxel.nii' and
%           sphere/disk   'prefix_seed_mask_conv.nii' 
% - text file with the a table of all artefact and control positions and 
%           their labels in the images. 
% - Bitmaps showing all expected artefact (and projected control) positions
%           in 2D and 3D
% - MP4 movie showing all slices containing expected artefact positions 
%           over time.
% - Bitmap of time courses of the seed and all possible artefact regions
% - Bitmap of time courses of both control regions
% - Bitmaps of correlations coefficients rho between seed and artefact and 
%           control positions(*_rho.bmp)
% - Bitmaps of all absolute correlations coefficients (*_abs_rho.bmp)
% - Bitmaps of all thresholded absolute correlations coefficients 
%           (*_corr_abs_rho_thresholded.bmp)
% - Matlab .m file containing all correlation values (rho and p)
% - Matlab .m file containing all time series (*_data_correlations.mat)
% 
% 
% Example script:
% cfg.seed = [31,85,22];
% cfg.sphere_radius = 6;
% cfg.disk = 'yes';
% cfg.corr_thresh = 0.6;
% cfg.saveprefix = 'TEST';
% cfg.flipLR = 'no';
% cfg.Shift_FOVdevX = 3;
% inputfilename = 'D:\BIDSfolder\s001\func\sub-01_task-X_bold.nii.gz';
% outputfolder='D:\MAP4SL_output';
% 
% MAP4SL(cfg, inputfilename, outputfolder)
% 
% 
% DEPENDECIES
% MAP4SL needs the toolbox "Tools for nifti and analyze image" for 
% Matlab in the MATLAB path. The toolbox can be downloaded from here:
% https://uk.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
% 
% 
% LICENCE
% MAP4SL by Michael Lindner is licensed under CC BY 4.0
% This program is distributed in the hope that it will be useful, 
% but WITHOUT ANY WARRANTY;
% 
% AUTHOR
% Michael Lindner  
% University of Reading, 2018  
% School of Psychology and Clinical Language Sciences  
% Center for Integrative Neuroscience and Neurodynamics


if nargin<2
    [fn,pn] = uigetfile({'*.gz';'*.nii'},'Select nfiti file (.nii.gz)');
    filename = fullfile(pn,fn);
end
if nargin<1
    %     'Specify control region(s) [LR,AP,slice] separated with ; for multiples if blank, center voxel will be used ',...

    
    prompts={'Specify seed region [LR,AP,slice]',...
        'Radius of sphere/disk for masks',...
        'Disk (yes) or sphere (no) around the expected artefact positions?',...
        'Threshold of absolut correlation coefficient images',...
        'Savename prefix',...
        'Flip left right artefact positions?',...
        'CAIPI shift (FOV dev X):'};
    defaults={'[69,83,14]',...
        '6',...
        'yes',...
        '0.6',...
        'PREFIX',...
        'no',...
        '3'};
    dims = [1 60];
    answer = inputdlg(prompts,'Input parameters',dims,defaults);
    
    cfg.seed = str2num(answer{1}); %#ok<*ST2NM>
    cfg.sphere_radius = str2num(answer{2});
    cfg.disk =  answer{3};
    cfg.abs_corr = str2num(answer{4});
    cfg.saveprefix = answer{5};
    cfg.flipLR = answer{6};
    cfg.Shift_FOVdevX = str2num(answer{7});
    
end
if nargin<2
    outputfolder = uigetdir('Select output folder','Select output folder');
end

% encodedir

labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

% ------------------------------------------------------------
% Load nifti data
% ------------------------------------------------------------

% unzip file
if strcmp(filename(end-2:end),'.gz')
    fprintf('\nGunzip nii.gz file')
    gunzip(filename)
    filename = filename(1:end-3);
    deletelater = 1;
    fprintf(' - ok')
else
    deletelater = 0;
end


% load nifti
fprintf('\nLoad nifti data')
nii = load_untouch_nii(filename);
D=nii.img; % get image
S=size(D); % get size of image
voxelsize = nii.hdr.dime.pixdim(2:4); % get voxel size
fprintf(' - ok')


% load json
fprintf('\nLoad json file')
jfilename = [filename(1:end-4), '.json'];
fid = fopen(jfilename);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
val = jsondecode(str);
fprintf(' - ok')

grappadim = 1;
try %#ok<*TRYNC>
    if isfield(val, 'ParallelReductionFactorInPlane')
        cfg.Grappa = 'yes';
        cfg.Grappafactor = val.ParallelReductionFactorInPlane;
        grappadim = cfg.Grappafactor;
    end
    
end
% cfg.Grappa = 'no';

% delete unzipped file
if deletelater == 1
    fprintf('\nDelete unzipped file')
    delete(filename)
    fprintf(' - ok')
end


% ------------------------------------------------------------
% create sphere
% ------------------------------------------------------------


% based on euclidian distance approach
fprintf('\nCreate spheres')
r = cfg.sphere_radius -1;
t = randn((r*2+1),(r*2+1),(r*2+1));
dist = nan(1,numel(t));
center = ceil(size(t)/2) ;
for ii = 1:numel(t)
    [dd(1,1),dd(1,2),dd(1,3)]=ind2sub(size(t),ii);
    dd(2,:)=center;
    dist(ii)=pdist(dd)*mean(voxelsize);
end
distmatrix = reshape(dist,size(t));
spheremask=zeros(size(t));

if strcmp(cfg.disk,'yes')
    sp=zeros(size(t));
    sp(distmatrix<=cfg.sphere_radius) = 1;
    spheremask(:,:,cfg.sphere_radius) = sp(:,:,cfg.sphere_radius);
    spname = ['_disk',num2str(cfg.sphere_radius)];
else
    spheremask(distmatrix<=cfg.sphere_radius)=1;
    spname = ['_sphere',num2str(cfg.sphere_radius)];
end
seedstring = num2str(cfg.seed);
seedstring = strrep(seedstring,' ','_');
cfg.saveprefix = [cfg.saveprefix, '_seed_',seedstring, spname ];
fprintf(' - ok')


% ------------------------------------------------------------
% create artefact mask
% ------------------------------------------------------------

fprintf('\nCreate artefact mask')
% direction of artefact
if strcmp(val.PhaseEncodingDirection, 'j')
    bslice = S(2)/cfg.Shift_FOVdevX;
elseif strcmp(val.PhaseEncodingDirection, 'j-')
    bslice = -S(2)/cfg.Shift_FOVdevX;
else
    errordlg('This tool only supports phase encoding directions A>P and P>A!')
end
inplane = S(2)/2;
% inplanec = S(2)/3.5;


% get simultaneous slices
SliceTime = val.SliceTiming;
% get number of conseq. slice "packages"
SeparateAcquisitions = unique(SliceTime);
% get MB accelation factor
accfactor = length(find(SliceTime==0));
% define simultaneous slices
SimulataneousSlices = cell(length(SeparateAcquisitions),1);
SimulataneousSlicesVec = zeros(length(SeparateAcquisitions),1);
for ii=1:length(SeparateAcquisitions)
    x = find(SliceTime==SeparateAcquisitions(ii));
    SimulataneousSlices{ii} = x;
    SimulataneousSlicesVec(x) = ii;
end


% empty mask
mask = zeros(S(1),S(2),S(3));
controlmask = zeros(S(1),S(2),S(3));
seedmask = zeros(S(1),S(2),S(3));

% Loop over seeds

% Seed positions
%     UD = S(3) - cfg.seed(ss,3) + 1;
%     UD = cfg.seed(ss,3);
VH = cfg.seed(2);
if strcmp(cfg.flipLR,'yes')
    LR = S(1) - cfg.seed(1);
else
    LR = cfg.seed(1);
end


% artefact positions
% ----------------------
% slices
sim = cfg.seed(3):length(SeparateAcquisitions):cfg.seed(3)+length(SeparateAcquisitions)*accfactor;
sim=sim(1:accfactor);
[sim,idx] = sort(mod(sim,size(nii.img,3)));
pp=-accfactor:accfactor;
xr = pp((accfactor+1)-(find(idx==1)-1):(accfactor+1)-(find(idx==1)-1)+accfactor-1);
realseed = find(sim==cfg.seed(3));

% sim = circshift(sim,-(realseed-1));
% xr = circshift(xr,-(realseed-1));


% controlsslice
simc = cfg.seed(3) : (size(nii.img,3)/accfactor)/2 : size(nii.img,3)+cfg.seed(3)-1;
simc=round(mod(simc,size(nii.img,3)));
% simc(simc==0) = [];
% simc = circshift(simc,-(realseed-1));


% AP
xm = xr * bslice;
x = mod(VH+xm, S(1));

%AP >Grappa
if strcmp(cfg.Grappa, 'yes')
    x2 = mod(x+inplane, S(1));
end

% CP
x3=cfg.seed(2)-(cfg.seed(2)-x(2))/4 : -(size(nii.img,2)/length(simc)) : -size(nii.img,2)+cfg.seed(2)-(cfg.seed(2)-x(2))/4-1;
x3 = round(mod(x3,size(nii.img,2)));
% cc2 = 0:S(1)/5:1000;
% if LR>S(1)/2
%     cc2=cc2*-1;
% end

if cfg.seed(1) < size(nii.img,1)/2
    LRc = cfg.seed(1) : (size(nii.img,1)/length(simc)) : size(nii.img,1)+cfg.seed(1)-1;
else
    LRc = cfg.seed(1) : -(size(nii.img,1)/length(simc)) : -size(nii.img,1)+cfg.seed(1)-1;
end
LRc = round(mod(LRc,size(nii.img,1)));

if ~isempty(find(sim==0, 1))
%     sim(find(sim==0)) = S(3);
    sim(sim==0) = 1;
end

sim = circshift(sim,-(realseed-1));
x = circshift(x,-(realseed-1));


if strcmp(cfg.Grappa, 'yes')
    if ~isempty(find(x2==0, 1))
        x2(x2==0) = 1;
    end
    x2 = circshift(x2,-(realseed-1));
end




f = find(x3==0);
LRc(f) = [];
x3(f) = [];
simc(f) = [];
f = find(simc==0);
LRc(f) = [];
x3(f) = [];
simc(f) = [];
f = find(LRc==0);
LRc(f) = [];
x3(f) = [];
simc(f) = [];

idx2redc = find(simc>sim(1));
idx2redc = idx2redc([1,3]);
LRc = LRc(idx2redc);
x3 = x3(idx2redc);
simc = simc(idx2redc);

% save textfile with artefact positions
sn = [cfg.saveprefix, '_label_position_link.txt'];
sn1 = fullfile(outputfolder,sn);
fid = fopen(sn1,'w');

poslabel = cell(length(sim),grappadim);
artefactpositions = cell(length(sim),grappadim);

for ii=1:length(sim)
    artefactpositions{ii,1} = [LR, x(ii), sim(ii)];
    poslabel{ii,1} = labels(ii);
    fprintf(fid,[labels(ii), ': ',num2str([LR, x(ii), sim(ii)]),'\n']);
%     fprintf(fid,'\r')
    if strcmp(cfg.Grappa, 'yes')
        artefactpositions{ii,2} = [LR, x2(ii), sim(ii)];
        poslabel{ii,2} = [labels(ii), '_g'];
        fprintf(fid,[labels(ii), '_g: ',num2str([LR, x2(ii), sim(ii)]),'\n']);
%         fprintf(fid,'\r')
    end
end

controlpositions = cell(length(simc),1);
for ii=1:length(simc)
    controlpositions{ii,1} = [LRc(ii), x3(ii), simc(ii)];
    fprintf(fid,['Control',num2str(ii),': ',num2str([LRc(ii), x3(ii), simc(ii)]),'\n']);
%     fprintf(fid,'\r')
end
fclose(fid);



% add position to mask
seedmask(LR, x(1), sim(1)) = 1;
maskcell = cell(length(sim),grappadim);
for ii=1:length(sim)
    mask(LR, x(ii), sim(ii)) = 1;
    
    mmm = zeros(S(1),S(2),S(3));
    mmm(LR, x(ii), sim(ii)) = 1;
    maskcell{ii,1} = mmm; 
    clear mmm
    
    if strcmp(cfg.Grappa, 'yes')
        mask(LR, x2(ii), sim(ii)) = 1;
        mmm = zeros(S(1),S(2),S(3));
        mmm(LR, x2(ii), sim(ii)) = 1;
        maskcell{ii,2} = mmm;
        clear mmm
    end
end
controlmaskcell = cell(length(simc),1);
for ii=1:length(simc)
    controlmask(LRc(ii), x3(ii), simc(ii)) = 1;
    
    mmm = zeros(S(1),S(2),S(3));
    mmm(LRc(ii), x3(ii), simc(ii)) = 1;
    controlmaskcell{ii,1} = mmm; 
    clear mmm
end


maskc = convn(mask,spheremask,'same');
controlmaskc = convn(controlmask,spheremask,'same');
seedmaskc = convn(seedmask,spheremask,'same');
fprintf(' - ok')

% save masks
fprintf('\nSave masks')
niix = nii;
nii.fileprefix = 'mask';
nii.hdr.dime.dim(5)=1;
nii.hdr.dime.glmax=1;


nii.img = seedmaskc;
sn = [cfg.saveprefix, '_seed_mask_conv.nii'];
sn1 = fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);
nii.img = seedmask;
sn = [cfg.saveprefix, '_seed_mask_single_voxel.nii'];
sn1 = fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);

nii.img = maskc;
sn = [cfg.saveprefix, '_artefact_mask_conv.nii'];
sn1 = fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);
nii.img = mask;
sn = [cfg.saveprefix, '_artefact_mask_single_voxel.nii'];
sn1 = fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);


nii.img = controlmask;
sn = [cfg.saveprefix, '_control_mask_single_voxel.nii'];
sn1 = fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);

nii.img = controlmaskc;
sn = [cfg.saveprefix, '_control_mask_conv.nii'];
sn1 = fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);

fprintf(' - ok')


% ---------------------------------------------------------
% plot masks images
% ---------------------------------------------------------

fprintf('\nPlot masks')

hm = figure('Name', ['artefact positions seed ',num2str(cfg.seed)], ...
    'Visible', 'on', 'Units', 'pixels', ...
    'MenuBar', 'none', 'ToolBar', 'none', ...
    'NumberTitle', 'off',...
    'Position', [1 1 350 1050]);

tt = size(D,3):-1:1;
tl = tt(sim);
tlab = cell(length(tl),1);
for ii = 1:length(tl)
    tlab{ii} = sim(length(tl)+1-ii);
end

subplot(3,1,1)
oo = squeeze(D(LR,:,:,1));
oo2 = int16(squeeze(mask(cfg.seed(1),:,:)));
oo(oo2==1) = max(oo(:))*1.1;
imagesc(flip(oo',1)); colormap gray
yticks(sort(tl))
yticklabels(tlab)
ylabel('z')
xlabel('y')
title(['x = ',num2str(LR)])

subplot(3,1,2)
oo = squeeze(D(LR,:,:,1));
oo2 = int16(squeeze(maskc(cfg.seed(1),:,:)));
oo(oo2==1) = max(oo(:))*1.1;
set(gca,'YDir','reverse')
imagesc(flip(oo',1)); colormap gray
yticks(sort(tl))
yticklabels(tlab)
ylabel('z')
xlabel('y')
title(['x = ',num2str(LR)])

subplot(3,1,3)
oo = squeeze(D(LR,:,:,1));
a = int16(squeeze(sum(controlmaskc,1)));
a(a>0) = 1;
oo(a==1) = max(oo(:))*0.5;
oo2 = int16(squeeze(maskc(cfg.seed(1),:,:)));
oo(oo2==1) = max(oo(:))*1.1;
set(gca,'YDir','reverse')
imagesc(flip(oo',1)); colormap gray
yticks(sort(tl))
yticklabels(tlab)
ylabel('z')
xlabel('y')
title(['gray = control positions projected on slice ',num2str(LR)])

saveas(hm,fullfile(outputfolder,[cfg.saveprefix,'_artefact_position_images.bmp']),'bmp')
fprintf(' - ok')

% ---------------------------------------------------------
% create slice movie
% ---------------------------------------------------------

% moviename = fullfile(outputfolder,['Movie_', num2str(cfg.sphere_radius), '.avi']);
% v = VideoWriter(moviename,'Uncompressed AVI');

moviename2 = fullfile(outputfolder,[cfg.saveprefix,'_slice_movie.mp4']);
v = VideoWriter(moviename2,'MPEG-4');
open(v)

x=size(D,1);
y=size(D,2);
D2=double(D);
D2=D2./(max(D2(:))/255)+1;
for nn=1:size(D,4)
    imdat = zeros(size(artefactpositions,1)*x,y);
    for aa = 1:size(artefactpositions,1)
        imdat((aa-1)*x+1:aa*x,:) = D2(:,:,artefactpositions{aa,1}(3),nn);
    end
    V = im2frame(flip(uint8(imdat)',1),gray(256));
    writeVideo(v,V);
end
close(v)




% ---------------------------------------------------------
% plot 3D
% ---------------------------------------------------------
ff = figure;
% a=patch(isosurface(maskc,0.5),'FaceColor','blue');
D2=D(:,:,:,1);
D2(D2<400)=0;
a = patch(isosurface(squeeze(D2(:,:,:,1)),0.5),'FaceColor','yellow'); %#ok<*NASGU>
alpha(0.5)
b = patch(isosurface(maskc,0.5),'FaceColor','red');
b = patch(isosurface(controlmaskc,0.5),'FaceColor','blue');
view(3)
saveas(ff,fullfile(outputfolder,[cfg.saveprefix,'_artefact_positions_3D.fig']),'fig')


% ---------------------------------------------------------------
% plot voxel time course
% ---------------------------------------------------------------

artefact_pos_vecs = cell(size(maskcell,2),size(maskcell,1));
DAT = cell(size(maskcell,2),size(maskcell,1));

h1=figure('Name', 'Voxel Time Courses - artefact positions', ...
    'Visible', 'on', 'Units', 'pixels', ...
    'MenuBar', 'none', 'ToolBar', 'none', ...
    'NumberTitle', 'off',...
    'Position', [100 100 1000 700]);
count=0;
for zz=1:size(maskcell,2)
    for yy=1:size(maskcell,1)
        count=count+1;
        
        subplot(size(maskcell,2),size(maskcell,1),count)
        m = maskcell{yy,zz};
        m = convn(m,spheremask,'same');
        s = find(m==1);
        [xind,yind,zind] = ind2sub(size(m),s);
        tt=nan(length(xind),S(4));
        artefact_pos_vecs{yy,zz}=[xind,yind,zind];
        for ii = 1:length(xind)
            tt(ii,:)=D(xind(ii),yind(ii),zind(ii),:);
        end
        imagesc(tt)
        colormap gray
        if yy==1
            if zz==1
                ylabel('voxel in artefact position (AP)')
            else
                ylabel({'GRAPPA';'voxel in ROI'})
            end
        end
        xlabel('volumes')
        
        if zz==1
            if yy==1
                title( ['SEED artefact ', labels(yy) , ' at slice ',num2str(sim(yy))] );
            else
                title( ['artefact ', labels(yy) , ' at slice ',num2str(sim(yy))] );
            end
        else
            title( ['artefact ', labels(yy) , '_g at slice ',num2str(sim(yy))] );
        end
        DAT{yy,zz}=tt;
    end
    
end
saveas(h1,fullfile(outputfolder,[cfg.saveprefix,'_voxel_time_courses_artefact_positions.bmp']),'bmp')


Control_pos_vecs = cell(size(controlmaskcell,1),1);
controlDAT = cell(size(controlmaskcell,1),1);

h1=figure('Name', 'Voxel Time Courses - control positions', ...
    'Visible', 'on', 'Units', 'pixels', ...
    'MenuBar', 'none', 'ToolBar', 'none', ...
    'NumberTitle', 'off',...
    'Position', [100 100 500 350]);
count=0;
for yy=1:size(controlmaskcell,1)
    count=count+1;
    
    subplot(1,size(controlmaskcell,1),count)
    m = controlmaskcell{yy};
    m = convn(m,spheremask,'same');
    s = find(m==1);
    [xind,yind,zind] = ind2sub(size(m),s);
    tt=nan(length(xind),S(4));
    Control_pos_vecs{yy}=[xind,yind,zind];
    for ii = 1:length(xind)
        tt(ii,:)=D(xind(ii),yind(ii),zind(ii),:);
    end
    imagesc(tt)
    colormap gray
    if yy==1
        ylabel({'CONTROL';'voxel in ROI'})
    end
    xlabel('volumes')
    
    title( ['control ROIs at slice ',num2str(simc(yy))] );
    controlDAT{yy}=tt;
end
saveas(h1,fullfile(outputfolder,[cfg.saveprefix,'_voxel_time_courses_control_positions.bmp']),'bmp')



% ---------------------------------------------------------------
% plot correlations
% ---------------------------------------------------------------

% create label set
qx = cell(1,length(sim));
qxg = cell(1,length(sim));
for ii = 1:length(sim)
    qx{ii} = labels(ii);
    qxg{ii} = [labels(ii),'_g' ];
end
try
    if cfg.Grappafactor>2
        for ii = 2:cfg.Grappafactor
            qxg = [qxg,qxg]; %#ok<AGROW>
        end
    end
end
qx = [qx,qxg];

name=['Correlation coeffs rho of seed_',num2str(cfg.seed), ' with artefact position ROIs'];
h3=figure('Name', name, ...
    'Visible', 'on', 'Units', 'pixels', ...
    'NumberTitle', 'off',...
    'Position', [100 100 1900 600]);

name=['Correlation coeffs abs rho of seed_',num2str(cfg.seed), ' with artefact position ROIs'];
h3a=figure('Name', name, ...
    'Visible', 'on', 'Units', 'pixels', ...
    'NumberTitle', 'off',...
    'Position', [100 100 1900 600]);

name=['Correlation coeffs abs thresh rho of seed_',num2str(cfg.seed), ' with artefact position ROIs'];
h3b=figure('Name', name, ...
    'Visible', 'on', 'Units', 'pixels', ...
    'NumberTitle', 'off',...
    'Position', [100 100 1900 600]);


count=1;
CORRcell = squeeze(DAT(:,:));
vals=1:size(maskcell,2)*size(maskcell,1);
vals(1)=[];
for yy=vals
    count=count+1;
    [r,p]=corr(CORRcell{yy}',CORRcell{1}');
    CORR_rho{count}=r'; %#ok<AGROW>
    CORR_p{count}=p'; %#ok<AGROW>
    
    figure(h3a)
    subplot(size(maskcell,2),size(maskcell,1),count)
    clims = [0 1];
    rt=abs(r);
%     rt(rt<cfg.corr_thresh)=0;
    imagesc(rt',clims)
    colorbar
    title(['abs rho for SEED A with ',qx{yy},' (',num2str(artefactpositions{yy}),')']);
    if yy==2
        ylabel(['seed ',num2str(cfg.seed)])
    end
    
    figure(h3b)
    subplot(size(maskcell,2),size(maskcell,1),count)
    clims = [0 1];
    rt=abs(r);
    rt(rt<cfg.corr_thresh)=0;
    imagesc(rt',clims)
    colorbar
    title(['thresh. abs rho for SEED A with ',qx{yy},' (',num2str(artefactpositions{yy}),')']);
    if yy==2
        ylabel(['seed ',num2str(cfg.seed)])
    end
    mrt = rt;
    mrt(mrt>0) = 1;
    CORR_mask{count}=mrt; %#ok<AGROW>
    
    figure(h3)
    subplot(size(maskcell,2),size(maskcell,1),count)
    clims = [-1 1];
    imagesc(r',clims)
    colorbar
    title(['rho for SEED A with ',qx{yy},' (',num2str(artefactpositions{yy}),')']);
    if yy==2
        ylabel(['seed ',num2str(cfg.seed)])
    end
    
end
saveas(h3,fullfile(outputfolder,[cfg.saveprefix,'_correlations_artefact_positions_rho','.bmp']),'bmp')
saveas(h3a,fullfile(outputfolder,[cfg.saveprefix,'_correlations_artefact_positions_abs_rho','.bmp']),'bmp')
saveas(h3b,fullfile(outputfolder,[cfg.saveprefix,'_correlations_artefact_positions_abs_rho_thresholded',num2str(cfg.corr_thresh),'.bmp']),'bmp')



% end

fprintf('\nSave data')
savename2 = [cfg.saveprefix,'_DATA.mat'];
save(fullfile(outputfolder,savename2), 'DAT', 'maskcell', 'artefactpositions', 'controlDAT', 'controlpositions', 'artefact_pos_vecs');
fprintf(' - ok')

f=zeros(size(nii.img));
for ii=2:length(CORR_mask)
    v = artefact_pos_vecs{ii}(sum(CORR_mask{ii}')>1,:); %#ok<*UDIM>
    for jj=1:size(v,1)
        f(v(jj,1),v(jj,2),v(jj,3)) = 1;
    end
    v1 = artefact_pos_vecs{1}(sum(CORR_mask{ii}')>1,:);
    for jj=1:size(v1,1)
        f(v1(jj,1),v1(jj,2),v1(jj,3)) = 1;
    end
end

fprintf('\nSave masks')
niix = nii;
nii.fileprefix = 'mask';
nii.hdr.dime.dim(5)=1;
nii.hdr.dime.glmax=1;

nii.img = f;
sn = [cfg.saveprefix, '_correlation_artefact_positions_mask.nii'];
sn1=fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);


% ---------------------------------------------------------------
% plot correlations control
% ---------------------------------------------------------------



name=['Correlation coeffs rho of seed_',num2str(cfg.seed), ' with control ROIs'];
h5=figure('Name', name, ...
    'Visible', 'on', 'Units', 'pixels', ...
    'NumberTitle', 'off',...
    'Position', [100 100 900 300]);

name=['Correlation coeffs abs rho of seed_',num2str(cfg.seed), ' with control ROIs'];
h5a=figure('Name', name, ...
    'Visible', 'on', 'Units', 'pixels', ...
    'NumberTitle', 'off',...
    'Position', [100 100 900 300]);

name=['Correlation coeffs abs rho thresh of seed_',num2str(cfg.seed), ' with control ROIs'];
h5b=figure('Name', name, ...
    'Visible', 'on', 'Units', 'pixels', ...
    'NumberTitle', 'off',...
    'Position', [100 100 900 300]);

count=0;
controlCORRcell = squeeze(controlDAT(:));
controlCORR_rho = cell(length(controlmaskcell),1);
controlCORR_p = cell(length(controlmaskcell),1);
control_CORR_mask = cell(length(controlmaskcell),1);
vals=1:length(controlmaskcell);
for yy=vals
    count=count+1;
    [r,p]=corr(controlCORRcell{yy}',CORRcell{1}');
    controlCORR_rho{count}=r';
    controlCORR_p{count}=p';
    
    figure(h5a)
    subplot(1,length(controlmaskcell),count)
    clims = [0 1];
    imagesc(abs(r'),clims)
    colorbar
    title(['abs rho for SEED A with control ',num2str(count),' (',num2str(controlpositions{yy}),') ' ]);
    if yy==2
        ylabel(['seed ',num2str(cfg.seed)])
    end
    
    figure(h5b)
    subplot(1,length(controlmaskcell),count)
    clims = [0 1];
    rt=abs(r);
    rt(rt<cfg.corr_thresh)=0;
    imagesc(rt',clims)
    colorbar
    title(['abs rho for SEED a with control ',num2str(count),' (',num2str(controlpositions{yy}),') ' ]);
%     colorbar
    if yy==2
        ylabel(['seed ',num2str(cfg.seed)])
    end
    
    mrt = rt;
    mrt(mrt>0) = 1;
    control_CORR_mask{count}=mrt;
    
    figure(h5)
    subplot(1,length(controlmaskcell),count)
    clims = [-1 1];
    imagesc(r',clims)
    colorbar
    title(['rho for SEED A with control ',num2str(count),' (',num2str(controlpositions{yy}),') ' ]);
%     colorbar
    if yy==2
        ylabel(['seed ',num2str(cfg.seed)])
    end
    
%     colorbar
end
saveas(h5,fullfile(outputfolder,[cfg.saveprefix,'_correlations_control_positions_rho','.bmp']),'bmp')
saveas(h5a,fullfile(outputfolder,[cfg.saveprefix,'_correlations_control_positions_abs_rho','.bmp']),'bmp')
saveas(h5b,fullfile(outputfolder,[cfg.saveprefix,'_correlations_control_positions_abs_rho_thresholded',num2str(cfg.corr_thresh),'.bmp']),'bmp')

f=zeros(size(nii.img));
for ii=1:length(control_CORR_mask)
    v = Control_pos_vecs{ii}(sum(control_CORR_mask{ii}')>1,:);
    for jj=1:size(v,1)
        f(v(jj,1),v(jj,2),v(jj,3)) = 1;
    end
    v1 = artefact_pos_vecs{1}(sum(control_CORR_mask{ii}')>1,:);
    for jj=1:size(v,1)
        f(v1(jj,1),v1(jj,2),v1(jj,3)) = 1;
    end
    
end

fprintf('\nSave masks')
niix = nii;
nii.fileprefix = 'mask';
nii.hdr.dime.dim(5)=1;
nii.hdr.dime.glmax=1;

nii.img = f;
sn = [cfg.saveprefix, '_correlation_control_position_mask.nii'];
sn1=fullfile(outputfolder,sn);
save_untouch_nii(nii,sn1);

savename2 = [cfg.saveprefix,'_DATA_correlations.mat'];
save(fullfile(outputfolder,savename2), 'CORR_rho', 'CORR_p', 'CORR_h','controlCORR_rho', 'controlCORR_p', 'controlCORR_h');



end
