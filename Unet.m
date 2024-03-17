load('matlab_complex_im.mat');
load('data1013.mat');

%getting all 10 test images (bmp) in one file image_patch%
for ii = 1:10
    im_pro=complex_im(:,:,ii);
    im_save=(angle(im_pro)+pi)*255/2/pi;
    file_path = sprintf("./image_patch/im_%d.bmp", ii);
    imwrite(im_save, colormap, file_path);
end

%utilizing folder label_patch with labeled images (bmp)%
pxDir='label_patch';
classNames = ["cell" "background" "nothing"];
pixelLabelID = [1 2 3];
imds_patch = imageDatastore('image_patch','FileExtensions','.bmp');
pxds_patch = pixelLabelDatastore(pxDir, classNames, pixelLabelID);
ds = pixelLabelImageDatastore(imds_patch,pxds_patch);
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',10,...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.9, ...
    'LearnRateDropPeriod', 1, ...
    'Plots','training-progress','Minibatchsize',20);
imageSize=[1024 1024];
numClasses = 3;
encoderDepth =4;%4
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth,'NumOutputChannels',32 );%64
net = trainNetwork(ds,lgraph,options);
