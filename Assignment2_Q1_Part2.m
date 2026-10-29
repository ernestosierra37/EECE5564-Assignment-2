
clear all, close all, clc

trainingSizes = [20 200 2000];

validation = load('Q1_Data_10000.mat');
xValidate = validation.x;
labelsValidate = validation.labels;


ModelType = 'logisticLinear';

% AI-assisted: Repeat training for the three required datasets
for k = 1:length(trainingSizes)

    Ntrain = trainingSizes(k);

    filename = sprintf('Q1_Data_%d.mat',Ntrain);
    data = load(filename);

    x = data.x;
    labels = data.labels;

    w0 = zeros(size(x,1)+1,1);

    NLL = @(w) binaryCrossEntropyCostFunction( ...
        w,x,labels,ModelType);

    % AI-assisted: Use fminsearch for optimization
    options = optimset('Display','off', ...
        'MaxIter',10000, ...
        'MaxFunEvals',20000, ...
        'TolX',1e-10, ...
        'TolFun',1e-10);

    [w,finalNLL,exitflag] = fminsearch(NLL,w0,options);

  
    [posterior,Zvalidate] = logisticGeneralizedLinearModel( ...
        xValidate,w,ModelType);
    decisions = posterior >= 0.5;

    Perror = sum(decisions ~= labelsValidate) ...
        /length(labelsValidate);

    fprintf('\n-----------------------------------\n');
    fprintf('Logistic-linear classifier (%d training samples)\n',Ntrain);

    fprintf('\nEstimated parameters:\n');
    disp(w);

    fprintf('Optimization exit flag: %d\n',exitflag);
    fprintf('Average negative log-likelihood: %.6f\n',finalNLL);

    fprintf('\nValidation error: %.6f\n',Perror);
    fprintf('Validation accuracy: %.2f%%\n',100*(1-Perror));

    fprintf('Predicted Class 0: %d\n',sum(decisions==0));
    fprintf('Predicted Class 1: %d\n',sum(decisions==1));

    fprintf('-----------------------------------\n');

end
