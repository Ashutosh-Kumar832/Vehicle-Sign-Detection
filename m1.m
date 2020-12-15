%% Load Network
load network.mat      
% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);


%%
% Load the ground truth data
%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
%%
stopSignsAndCars = data.stopSignsAndCars;
%%


visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);
                                                                                                                  
summary(stopSignsAndCars)