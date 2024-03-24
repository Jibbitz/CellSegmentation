clc
clear, 
close all

im_folder = 'images_newBatch\';
load('matlab_complex_im.mat');
load('matlab_cell81.mat');

%% Writing new bmp images

    for i=1:1:81

            %max_abs = max(max(abs(v_data_save(:,:,i))));
            %amplitude_im=abs(v_data_save(:,:,i))/max_abs;
            %amplitude_im_adj=uint8(amplitude_im*255);

            phase_im=angle(v_data_save(:,:,i));
            phase_im_adj=(phase_im+pi)*255/2/pi;

            %file_path = sprintf("./amp_images_newBatch/amp_im_%d.bmp", i);
            %imwrite(amplitude_im_adj, colormap, file_path);
            file_path = sprintf("./images_newBatch/im_%d.bmp", i);
            imwrite(phase_im_adj, colormap, file_path);
    
    end
