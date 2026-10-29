clear all, close all, clc

C = 2;
M = 4;
n = 2;

gmmParameters.numberOfClasses = C;
gmmParameters.component2label = [0,0,1,1];
gmmParameters.priors = [0.3,0.3,0.2,0.2];

gmmParameters.meanVectors = ...
    [-0.9  0.8 -1.1  0.9;
    -1.1  0.75 0.9 -0.75];

for m = 1:M
    gmmParameters.covMatrices(:,:,m) = ...
        [0.75 0; 0 1.25];
end

Nvalues = [20,200,2000,10000];

for k = 1:length(Nvalues)
    N = Nvalues(k);

    [x,componentLabels] = generateDataFromGMM(N,gmmParameters);

    labels = zeros(1,N);

    for m = 1:M
        indm = find(componentLabels == m);
        labels(indm) = gmmParameters.component2label(m);
    end

    filename = sprintf('Q1_Data_%d.mat',N);
    save(filename,'x','labels','gmmParameters');

    fprintf('Dataset with %d samples generated.\n',N);
    fprintf('Class 0: %d samples\n',sum(labels==0));
    fprintf('Class 1: %d samples\n\n',sum(labels==1));
end

load('Q1_Data_10000.mat');

p0 = 0.5*evalGaussianPDF(x, ...
    gmmParameters.meanVectors(:,1), ...
    gmmParameters.covMatrices(:,:,1)) ...
    + 0.5*evalGaussianPDF(x, ...
    gmmParameters.meanVectors(:,2), ...
    gmmParameters.covMatrices(:,:,2));

p1 = 0.5*evalGaussianPDF(x, ...
    gmmParameters.meanVectors(:,3), ...
    gmmParameters.covMatrices(:,:,3)) ...
    + 0.5*evalGaussianPDF(x, ...
    gmmParameters.meanVectors(:,4), ...
    gmmParameters.covMatrices(:,:,4));

gamma = 0.6/0.4;
likelihoodRatio = p1./p0;
decisions = likelihoodRatio > gamma;

% AI-assisted: Calculate validation classification error
Perror = sum(decisions ~= labels)/length(labels);

fprintf('Theoretical optimal threshold: %.6f\n',gamma);
fprintf('Validation classification error: %.6f\n',Perror);
fprintf('Validation accuracy: %.2f%%\n',100*(1-Perror));
% AI-assisted: Generate the ROC curve
scores = likelihoodRatio;
thresholds = [Inf, sort(scores,'descend'), 0];

Pfp = zeros(size(thresholds));
Ptp = zeros(size(thresholds));

for k = 1:length(thresholds)
    decisionsROC = scores > thresholds(k);

    Pfp(k) = sum(decisionsROC==1 & labels==0)/sum(labels==0);
    Ptp(k) = sum(decisionsROC==1 & labels==1)/sum(labels==1);
end

% AI-assisted: Calculate the optimal ROC operating point
optimalFP = sum(decisions==1 & labels==0)/sum(labels==0);
optimalTP = sum(decisions==1 & labels==1)/sum(labels==1);

figure(1), clf
plot(Pfp,Ptp,'b-','LineWidth',1.5)
hold on
plot(optimalFP,optimalTP,'ro','MarkerSize',10,'LineWidth',2)

xlabel('False Positive Probability')
ylabel('True Positive Probability')
title('ROC Curve - Optimal Gaussian Mixture Classifier')
legend('ROC Curve','Minimum P(error)','Location','southeast')
grid on
axis([0 1 0 1])
hold off

fprintf('False positive probability: %.6f\n',optimalFP);
fprintf('True positive probability: %.6f\n',optimalTP);

exportgraphics(gca,'Q1_ROC.png');