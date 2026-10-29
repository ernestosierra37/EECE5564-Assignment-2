clear all, close all, clc
rng(1);

Ntrain = 100;
Nvalidate = 1000;

[xTrain,yTrain,xValidate,yValidate] = hw2q2(Ntrain,Nvalidate);

disp('Training dataset:')
disp(size(xTrain))
disp(size(yTrain))

disp('Validation dataset:')
disp(size(xValidate))
disp(size(yValidate))

% AI-assisted: Construct complete cubic polynomial features

Ztrain = [ones(1,Ntrain);
    xTrain(1,:);
    xTrain(2,:);
    xTrain(1,:).^2;
    xTrain(1,:).*xTrain(2,:);
    xTrain(2,:).^2;
    xTrain(1,:).^3;
    xTrain(1,:).^2.*xTrain(2,:);
    xTrain(1,:).*xTrain(2,:).^2;
    xTrain(2,:).^3];

Zvalidate = [ones(1,Nvalidate);
    xValidate(1,:);
    xValidate(2,:);
    xValidate(1,:).^2;
    xValidate(1,:).*xValidate(2,:);
    xValidate(2,:).^2;
    xValidate(1,:).^3;
    xValidate(1,:).^2.*xValidate(2,:);
    xValidate(1,:).*xValidate(2,:).^2;
    xValidate(2,:).^3];

% AI-assisted: ML least-squares estimator
wML = Ztrain'\yTrain';

yPredicted = Zvalidate'*wML;
MLerror = mean((yValidate' - yPredicted).^2);

fprintf('ML validation average squared error: %.6f\n',MLerror);

% AI-assisted: Estimate noise variance using ML training residuals
yTrainPredicted = Ztrain' * wML;

residuals = yTrain' - yTrainPredicted;

sigma2 = mean(residuals.^2);

fprintf('Estimated noise variance: %.6f\n',sigma2);

% AI-assisted: Evaluate MAP estimator for different gamma values

gammaValues = logspace(-4,4,50);

MAPerror = zeros(size(gammaValues));

for k = 1:length(gammaValues)

    gamma = gammaValues(k);

    wMAP = (Ztrain*Ztrain' + ...
        (sigma2/gamma)*eye(size(Ztrain,1))) ...
        \ (Ztrain*yTrain');

    yMAP = Zvalidate' * wMAP;

    MAPerror(k) = mean((yValidate' - yMAP).^2);

end

figure, clf

semilogx(gammaValues,MAPerror,'b-', ...
    'LineWidth',1.5)
hold on

yline(MLerror,'r--','LineWidth',1.5)

xlabel('Gamma')
ylabel('Validation Average Squared Error')
title('ML and MAP Estimation - Validation Error')
legend('MAP','ML','Location','best')
grid on
hold off

fprintf('Minimum MAP validation error: %.6f\n',min(MAPerror));
fprintf('ML validation error: %.6f\n',MLerror);

[minMAPerror,idx] = min(MAPerror);
fprintf('Best gamma: %.6f\n',gammaValues(idx));