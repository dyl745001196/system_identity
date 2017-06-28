clear; clc; close all;
   
% Save the end results
save_name = 'accNN_xyz_v22';

% Number of hidden units
num_hidden = 100;

% Regularization factor
reg_factor = 0.1;

% Training parameters
max_epo = 500;
max_fail = 50;
trainingFcn = 'trainrp';
loss = 'mse';

% Load input data: 
load('./nnetinput_cf0_traj.mat');
xNN = cell2mat(inputNN);
clear('inputNN');
% fAct = cell2mat(fAct);
% fNom = cell2mat(fNom);
fAct = cell2mat(outputNN);
%X = cell2mat(X);
%U = cell2mat(U);

% Input to the Nnet
x = xNN;

% take out everything except angles
take_out_indices = [4:6,14:16];%v
x(take_out_indices,:) = [];
target_indices = 1:3;
 x1=x';
% Target data 
t = fAct(target_indices,:);
t1=t';
% t_nom = fNom(target_indices,:);

% Generate a neural net
% Using the cascadeforwardnet to have a Quadratic lag model
net = feedforwardnet([num_hidden], trainingFcn);

% Do not pre-process inputs or outputs
net.inputs{1}.processFcns = {};
% net.inputs{1}.processFcns{2} = 'mapstd';

% net.outputs{2}.processFcns = {};
net.outputs{2}.processFcns{2} = 'mapstd';
% DO NOT FORGET TO CHANGE THE 'STANDARD' SETTINGS OF MSE

% Configure the network
net = configure(net, x, t);

% Transfer function of the hidden layer neurons
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'purelin';

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.25;
net.divideParam.testRatio = 0.15;

% Initilaize ther overall net, all layers and their biases
net = init(net);
for i=1:net.numlayers
  % Set the layer initialization function
  net.layers{i}.initFcn = 'initwb';
  % Initialize bias weights
  net.biases{i}.initFcn = 'rands';
  net.b{i} = zeros(size(net.b{i}));
end

for i=1:length(net.inputweights)
% Initilaize all input weights
  net.inputweights{i}.initFcn = 'rands';
  net.IW{i} = randn(size(net.IW{i}));
end

% Initilaize all layer weights
for i=1:net.numlayers-1
  net.layerweights{i+1, i}.initFcn = 'rands';
  net.LW{i+1,i} = randn(size(net.LW{i+1,i}));
end
 
% Set the training functions and start training
% Choose a Performance Function and a Regularization factor
% For a list of all performance functions type: help nnperformance
net.performFcn = loss; 
net.performparam.regularization = reg_factor;
% net.performparam.normalization = 'standard';
% Maximum epochs and validation fails
net.trainparam.epochs = max_epo;
net.trainparam.max_fail = max_fail;
% Don't show the nntraintool
net.trainparam.showwindow = 1;
% Train the network
[net, tr] = train(net, x, t, 'UseGPU', 'no', 'useparallel', 'yes', 'ShowResources','yes');

% Save the set
save(save_name);

% Compute the validation MSE
val_in = x(:,tr.valInd);
val_target = t(:,tr.valInd);
val_y = net(val_in);
for i=1:size(val_y,1)
mse_val(i) = mse(val_target(i,:), val_y(i,:));
end
mse_val
sum(mse_val)/3