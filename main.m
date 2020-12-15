
   
load network.mat      
% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);


%%
% Load the ground truth data
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
%%
stopSignsAndCars = data.stopSignsAndCars;
%%


visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);

summary(stopSignsAndCars)
%%

stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign','carRear','carFront'});
%%
% Display one training image and the ground truth bounding boxes
I = imread(stopSigns.imageFilename{3});
I = insertObjectAnnotation(I,'Rectangle',stopSigns.stopSign{1},'stop sign','LineWidth',8);
I = insertObjectAnnotation(I,'Rectangle',stopSigns.carRear{1},'car','LineWidth',8);
I = insertObjectAnnotation(I,'Rectangle',stopSigns.carFront{1},'car','LineWidth',8);


figure
imshow(I)
%%

doTraining = false;

if doTraining
    
    % Set training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);
    
    % Train an R-CNN object detector. This will take several minutes.    
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])
else
    
    load traineddata.mat% trained network
end
%%
% Read test image
testImage = imread('image007.jpg');
imshow(testImage)
% Detect stop signs
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)
%%
% Display the detection results
% [score, idx] =score

% bbox = bboxes(idx, :);
for idx=1:3
annotation = [sprintf('%s: (Confidence = %f)', label(idx), score(idx));];
le{idx}=annotation;
end
outputImage = insertObjectAnnotation(testImage, 'rectangle', bboxes, le);
imshow(outputImage)
pause(1)
for i=1:3
    w=label(i,:);
tts(char(w));
end
% hold on

%%

