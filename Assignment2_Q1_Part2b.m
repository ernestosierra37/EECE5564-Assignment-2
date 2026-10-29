
clear all, close all, clc

trainingSizes = [20 200 2000];

validation = load('Q1_Data_10000.mat');
xValidate = validation.x;
labelsValidate = validation.labels;

ModelType = 'logisticQuadratic';

% AI-assisted: Train on each required dataset
for k = 1:length(trainingSizes)

    Ntrain = trainingSizes(k);

    filename = sprintf('Q1_Data_%d.mat',Ntrain);
    data = load(filename);

    x = data.x;
    labels = data.labels;

    n = size(x,1);
    w0 = zeros(1+n+n^2,1);

    NLL = @(w) binaryCrossEntropyCostFunction( ...
        w,x,labels,ModelType);

    % AI-assisted: Numerical optimization settings
    options = optimset('Display','off', ...
        'MaxIter',10000, ...
        'MaxFunEvals',20000, ...
        'TolX',1e-10, ...
        'TolFun',1e-10);

    [w,finalNLL,exitflag] = fminsearch(NLL,w0,options);

    [posterior,zValidate] = logisticGeneralizedLinearModel( ...
        xValidate,w,ModelType);

    decisions = posterior >= 0.5;


    Perror = sum(decisions ~= labelsValidate) ...
        /length(labelsValidate);

    fprintf('\n-----------------------------------\n');
    fprintf('Logistic-quadratic classifier (%d training samples)\n',Ntrain);

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
