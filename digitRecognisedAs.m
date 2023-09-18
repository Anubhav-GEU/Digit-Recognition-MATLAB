[filename,pathname] = uigetfile('*.*','Select the input grayscale');

filewithpath = strcat(pathname,filename);

I = imread(filewithpath);

figure
imshow(I);

label = classify(net, I);
title(['Digit Recognized as' char(label)]);