clc
clear, 
close all

im_folder = 'images\';
label_folder = 'PixelLabelData_1\';

load('matlab_complex_im.mat');
%max_abs = max(max(abs(complex_im(:,:,i)))); %changed "1" to "i"
imag_unit = sqrt(-1);
im_rgb = uint8(zeros(256, 256, 3));

augmentationSteps = 10:10:300; % Augmentation steps; variable, currently (30)
averageDiceCoefficients = zeros(length(augmentationSteps), 1);

%% Augmentation Iterations

% First for-loop for setting up k iterations (1 to #ofAug, currently (30))
for step = 1:length(augmentationSteps)
    k = augmentationSteps(step);

% Second for-loop for reading and resizing labels (1 to 10) done k times
%           (300 read and resized labels);
%           also for extracting only AMPLITUDE portion of original 10 bmp
%           images and resizes them to match label done k times (creates
%           300 amp images);
    for i=1:1:10
        label_file=[label_folder 'Label_' num2str(i) '.png'];
        label_im=imread(label_file,'png');
        label_im_resize=imresize(label_im,[256 256],"bilinear");
        % resize the image is important for network training
    
        max_abs = max(max(abs(complex_im(:,:,i)))); %changed "1" to "i"

        amplitude_im=abs(complex_im(:,:,i))/max_abs; 
        % pay attention to the index to match the labeling with the image
        amplitude_im_resize=imresize(amplitude_im,[256 256],"bilinear");
        figure

% Third for-loop for extracting PHASE portion of original 10 bmp images;
%           gets you k phase images (currently 30) from ith image (1 original bmp image);
%           creates k complex numbers (currently 30) to multiply by the k phase
%           images created from the ith image (1 original bmp image);
%           resizes k augmented phase images to be same size as label;
        for j = 1:1:k
            phase_im=angle(complex_im(:,:,i));
            
            complex_Num=exp(-imag_unit*pi*j/10);
            phase_im_adj=angle(complex_Num*complex_im(:,:,i));
            % multiple the same complex number to the complex imaging. This
            % effectively shifts the phase value at all the pixels. It also
            % changes where phase wrapping artifact happens

            phase_im_adj_resize= imresize(phase_im_adj,[256 256],"bilinear");
            phase_im_adj_resize=(phase_im_adj_resize+pi)/2/pi;
            im_rgb(:,:,1)=uint8(amplitude_im_resize*255);% one can use the amplitude for training or not
            im_rgb(:,:,2)=uint8(phase_im_adj_resize*255); %.* randn([256 256])*0.5);
      
            imwrite(label_im_resize,['label_augmented/label' num2str((i-1)*10+j) '.png']);
    
            imwrite(im_rgb,['im_augmented/im' num2str((i-1)*10+j) '.bmp']);
    
    
        end
    end

    % Training code
    imds = imageDatastore('im_augmented', 'FileExtensions', '.bmp');
    classNames = ["bk", "cell", "nocell"];
    pixelLabelID = [0 1 2];
    pxds = pixelLabelDatastore('label_augmented', classNames, pixelLabelID, 'FileExtensions', '.png');
    ds = combine(imds, pxds);

    options = trainingOptions('adam', ...
        'InitialLearnRate',0.001, ...
        'MaxEpochs',20, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.9, ...
        'LearnRateDropPeriod', 1, ...
        'Plots','training-progress','MiniBatchSize', 5); % added ('Plots','training-progress',) to show training graph, can be removed

    imageSize = [256 256 3];
    numClasses = 3;
    encoderDepth = 3;
    lgraph = unetLayers(imageSize, numClasses, 'EncoderDepth', encoderDepth, 'NumOutputChannels', 32); % Adjust as needed

    net = trainNetwork(ds, lgraph, options);

    % Testing and Dice coefficient calculation
    numImages = numel(pxds.Files);
    
    for i = 1:numImages
        trueLabel = imread(pxds.Files{i});
        img = imread(imds.Files{i});
        predictedLabel = semanticseg(img, net);
        % Assuming 'calculateDiceCoefficient' is a function you've defined
        diceCoefficients(i) = dice(double(predictedLabel=='cell'), double(trueLabel==1)); 
    end
    averageDiceCoefficients(step) = mean(diceCoefficients);
end

% Plotting the results
plot(augmentationSteps, averageDiceCoefficients);
xlabel('Number of Augmentations');
ylabel('Average Dice Coefficient');
title('Average Dice Coefficient vs Number of Augmentations');
