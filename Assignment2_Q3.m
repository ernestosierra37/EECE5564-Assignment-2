clear all, close all, clc
rng(1);

truePosition = [0.4; 0.3];

sigmaNoise = 0.3;

sigmaX = 0.25;
sigmaY = 0.25;

Kvalues = 1:4;

% AI-assisted: Define grid for MAP objective contours
gridValues = linspace(-2,2,201);
[X,Y] = meshgrid(gridValues,gridValues);

contourLevels = 0:2:80;

for K = Kvalues

    angles = 2*pi*(0:K-1)/K;
    landmarks = [cos(angles); sin(angles)];

    trueDistances = sqrt(sum((landmarks - truePosition).^2,1));

    measurements = trueDistances + sigmaNoise*randn(1,K);

    % AI-assisted: Reject and resample negative measurements
    while any(measurements < 0)
        ind = measurements < 0;

        measurements(ind) = trueDistances(ind) + ...
            sigmaNoise*randn(1,sum(ind));
    end

    % AI-assisted: MAP objective function
    MAPobjective = @(p) ...
        sum((measurements - ...
        sqrt(sum((landmarks - p(:)).^2,1))).^2) ...
        /(2*sigmaNoise^2) ...
        + p(1)^2/(2*sigmaX^2) ...
        + p(2)^2/(2*sigmaY^2);

    initialPosition = [0;0];

    options = optimset('Display','off');

    [MAPestimate,minObjective] = ...
        fminsearch(MAPobjective,initialPosition,options);

    MAPestimate = MAPestimate(:);

    positionError = norm(MAPestimate - truePosition);

    fprintf('\n-----------------------------------\n');
    fprintf('K = %d landmarks\n',K);

    fprintf('True vehicle position:\n');
    disp(truePosition);

    fprintf('MAP estimated position:\n');
    disp(MAPestimate);

    fprintf('Minimum MAP objective: %.6f\n',minObjective);
    fprintf('Position estimation error: %.6f\n',positionError);

    % AI-assisted: Evaluate MAP objective over grid
    objectiveValues = zeros(size(X));

    for i = 1:size(X,1)
        for j = 1:size(X,2)

            candidatePosition = [X(i,j);Y(i,j)];

            objectiveValues(i,j) = ...
                MAPobjective(candidatePosition);

        end
    end

    figure(K), clf

    contour(X,Y,objectiveValues,contourLevels, ...
        'LineWidth',1);

    hold on

    plot(truePosition(1),truePosition(2), ...
        'r+','MarkerSize',12,'LineWidth',2.5);

    plot(landmarks(1,:),landmarks(2,:), ...
        'ko','MarkerSize',9,'LineWidth',2);

    plot(MAPestimate(1),MAPestimate(2), ...
        'bx','MarkerSize',11,'LineWidth',2.5);

    xlabel('x');
    ylabel('y');

    title(sprintf('K = %d Landmarks',K));

    legend('Contours','True Position','Landmarks', ...
        'MAP Estimate','Location','southoutside');

    axis equal
    axis([-2 2 -2 2]);
    grid on
    hold off

    set(gcf,'Position',[100 100 900 650]);

    filename = sprintf('Q3_MAP_K%d.png',K);
    exportgraphics(gcf,filename,'Resolution',200);

end