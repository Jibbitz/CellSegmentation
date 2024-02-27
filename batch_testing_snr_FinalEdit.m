clear, close all
im_folder = 'images\';
label_folder = 'PixelLabelData_1\';

load('matlab_complex_im.mat');
max_abs = max(max(abs(complex_im(:,:,1))));
imag_unit = sqrt(-1);
im_rgb = uint8(zeros(256, 256, 3));

snrValues = 0.1:0.1:1; % SNR values from 0.1 to 1
averageDiceCoefficients = zeros(length(snrValues), 1);

SNR = zeros(1,10);

for snrIndex = 1:length(snrValues)
    snr = snrValues(snrIndex);
    diceCoefficients = zeros(1, 10); % Store dice coefficients for each image
    
    noise_var = double(zeros(1,256,length(snrValues)));
    signal = double(zeros(1,10));


    for i = 1:10
        label_file = [label_folder 'Label_' num2str(i) '.png'];
        label_im = imread(label_file, 'png');
        label_im_resize = imresize(label_im, [256 256], "bilinear");

        amplitude_im = abs(complex_im(:,:,i)) / max_abs;
        amplitude_im_resize = imresize(amplitude_im, [256 256], "bilinear");

        for j = 1:10 % Number of augmentations
            phase_im = angle(complex_im(:,:,i));
            complex_Num = exp(-imag_unit * pi * j / 10);
            phase_im_adj = angle(complex_Num * complex_im(:,:,i));

            noise = randn(size(phase_im_adj))*(0.1*(snrIndex));

            phase_noise = phase_im_adj + noise; % Add noise based on SNR
            phase_noise_resize = imresize(phase_noise, [256 256], "bilinear");
            phase_noise_resize = (phase_noise_resize + pi) / (2 * pi);

            im_rgb(:,:,1) = uint8(amplitude_im_resize * 255);
            im_rgb(:,:,2) = uint8(phase_noise_resize * 255);

            
            noise=imresize(noise,[256 256],"bilinear");
            
            noise_var(:,:,j) = var(noise); % only need these 10 random noise matrices because they increase by snrIndex; also it's random, you can use that same radom for every i

            
            imwrite(label_im_resize, ['label_augmented/label_' num2str(snrIndex) '_' num2str(j) '_' num2str(i) '.png']);
            imwrite(im_rgb, ['im_augmented/im_' num2str(snrIndex) '_' num2str(j) '_' num2str(i) '.bmp']);
        end
    end

%%
    % Training code
    imds = imageDatastore('im_augmented', 'FileExtensions', '.bmp');
    classNames = ["bk", "cell", "nocell"];
    pixelLabelID = [0 1 2];
    pxds = pixelLabelDatastore('label_augmented', classNames, pixelLabelID, 'FileExtensions', '.png');
    ds = combine(imds, pxds);

    options = trainingOptions('adam', ...
        'InitialLearnRate',0.001, ...
        'MaxEpochs',4, ... %changed to 4 for time (originally 20
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.9, ...
        'LearnRateDropPeriod', 1, ...
        'MiniBatchSize', 5);

    imageSize = [256 256 3];
    numClasses = 3;
    encoderDepth = 3;
    lgraph = unetLayers(imageSize, numClasses, 'EncoderDepth', encoderDepth, 'NumOutputChannels', 32); % Adjust as needed

    net = trainNetwork(ds, lgraph, options);


%%    
    % Testing and Dice coefficient calculation
    numImages = numel(pxds.Files);
    
    for i = 1:numImages
        trueLabel = imread(pxds.Files{i});
        img = imread(imds.Files{i});
        predictedLabel = semanticseg(img, net);

        signal=double(predictedLabel=="cell");

        % Assuming calculateDiceCoefficient is a predefined function for calculating Dice coefficient
        diceCoefficients(i) = dice(double(predictedLabel=='cell'), double(trueLabel==1)); 
    end

    signal_mean = mean(signal);
   
    SNR(:,snrIndex) = 20*log10(signal_mean/noise_var(:,:,snrIndex));

    averageDiceCoefficients(snrIndex) = mean(diceCoefficients);
end

%%

% Plotting the results
plot(SNR, averageDiceCoefficients);
xlabel('SNR Value');
ylabel('Average Dice Coefficient');
title('Average Dice Coefficient vs SNR Value');
