clear all; clc;
fprintf('This is sudhanshu Kumar Singh \n');
input_dir= 'C:\Users\Sudhanshu\Downloads\Compressed\faces';

img_dims= [896, 592];
%filenames is an array of all image names
filenames= dir(fullfile(input_dir, '*.jpg'));
num_images= numel(filenames); %numel() returns the size of array
images= [];

for n= 1: num_images
    filename = fullfile(input_dir, filenames(n).name);
    img= imread(filename);
    img= rgb2gray(img);
    img= imresize(img, [64,48]);
    if n==1
        images= zeros(prod([64,48]), num_images); % creates a zero vector
    end
    %append img(:) to nth column of images
    images(:, n)= img(:); % arr(:) makes matrix a column vector
end

% steps 1 and 2: find the mean image and the mean-shifted input images
mean_face= mean(images, 2); %takes mean of each row of matrix
shifted_images= images - repmat(mean_face, 1, num_images);

% steps 3 and 4: calculate the ordered eigenvectors and eigenvalues
[evectors, score, evalues]= pca(images');

% step 5: only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces= 40;
evectors= evectors(:, 1:num_eigenfaces);

% step 6: project the images into the subspace to generate the feature vectors
features= evectors'*shifted_images;


path= fullfile('C:\Users\Sudhanshu\Downloads\Compressed\','download.jpg');
input_image= imread(path);
input_image= rgb2gray(input_image);
input_image= imresize(input_image, [64,48]);
%imshow(input_image);

% calculate the similarity of the input to each training image
feature_vec= evectors'*(double(input_image(:)) - mean_face);
similarity_score = arrayfun(@(n)1/(1+norm(features(:,n) - feature_vec)), 1:num_images);

% find the image with the highest similarity
[match_score, match_ix] = max(similarity_score);

% display the result
figure, imshow([input_image reshape(images(:, match_ix), [64 48])]);
title(sprintf('matches %s, score %f', filenames(match_ix).name, match_score));

% display the eigenvectors
figure;
for n = 1:num_eigenfaces
    subplot(2, ceil(num_eigenfaces/2), n);
    evector = reshape(evectors(:,n), [64 48]);
    imshow(evector);
end


path= fullfile(input_dir,'image_0012.jpg');
img= imread(path);
img= rgb2gray(img);
img= imresize(img, [64,48]);
%imshow(img);