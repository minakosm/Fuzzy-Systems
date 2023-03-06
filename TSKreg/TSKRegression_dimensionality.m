%% Clear workspace, command window and close windows
clc
clear 
close all 

%% Load Data 
data = importdata('Datasets/superconduct.csv');
preproc = 1; 
[trnData, valData, testData] = split_scale(data,preproc);

dataTarget = data(:,end);
trnTarget = trnData(:,end);
valTarget = valData(:,end);
testTarget = testData(:,end);

%% Evaluation functions
MSE = @(ypred, y) mse(ypred,y);
RMSE = @(ypred, y) sqrt(MSE(ypred,y));
R2 = @(ypred,y) 1-sum((y-ypred).^2)/sum((y-mean(y)).^2);
NMSE = @(ypred, y) 1 - R2(ypred,y);
NDEI = @(ypred,y) sqrt(NMSE(ypred,y));

%% Grid Search
[idx, weight] = relieff(data(:,1:end-1),data(:,end),10, 'method','regression');

%Plot the weights of the most significant predictors
bar(weight(idx));
ylabel("Weights");

featNum = 5:2:11;
cRadius = 0.3:0.3:0.9;
k_folds = 5 ; 

gridErrors = zeros(length(featNum), length(cRadius), 5);
gridRulesNum = zeros(length(featNum), length(cRadius));

for f=featNum
    Dtrn = [trnData(:,idx(1:f)) trnTarget];
    Dtst = [testData(:,idx(1:f)) testTarget];
    for r=cRadius
        crossVal = cvpartition(Dtrn(:,end), "KFold",k_folds);
        CVerror = zeros(5, k_folds);

        for k=1:k_folds
            fprintf('FEATURES = %d\n RADIUS = %d\n K_FOLD = %d\n', f, r, k);
            DtrnCV = Dtrn(training(crossVal,k),:);
            DvalCV = Dtrn(test(crossVal,k),:);
            
            fisOpt = genfisOptions("SubtractiveClustering","ClusterInfluenceRange",r);
            fis = genfis(DtrnCV(:,1:end-1), DtrnCV(:,end), fisOpt);

            %If check because if nr of Rules are less than 2 then genfis
            %produces an error.
            if (size(fis.Rules,2) < 2)
                continue;
            end

            ANFISopt = anfisOptions;
            ANFISopt.InitialFIS = fis;
            ANFISopt.EpochNumber = 75;
            ANFISopt.ValidationData = DvalCV;
            
            [trnFis,trnError,stepSize,valFis,valError] = anfis(DtrnCV, ANFISopt);

            ypred = evalfis(valFis,Dtst(:,1:end-1));
            CVerror(:, k) = [MSE(ypred, testTarget)
                            RMSE(ypred, testTarget)
                            NMSE(ypred, testTarget)
                            NDEI(ypred, testTarget) 
                            R2(ypred,testTarget)];
        end
        indexes = [find(featNum == f) find(cRadius == r)];

        gridMSE(indexes(1),indexes(2))= mean(CVerror(1,:));
        gridRMSE(indexes(1),indexes(2))= mean(CVerror(2,:));
        gridNMSE(indexes(1),indexes(2))= mean(CVerror(3,:));
        gridNDEI(indexes(1),indexes(2)) = mean(CVerror(4,:));
        gridR2(indexes(1),indexes(2))= mean(CVerror(5,:));

        gridRulesNum(indexes(1),indexes(2)) = size(valFis.Rules,2);
    end

end

%% Plots
errorNames= ["MSE" "RMSE" "NMSE" "NDEI" "R2"];

gridErrors(:,:,1) = gridMSE;
gridErrors(:,:,2) = gridRMSE;
gridErrors(:,:,3) = gridNMSE;
gridErrors(:,:,4) = gridNDEI;
gridErrors(:,:,5) = gridR2;

% Errors / NumOfRules
figure();
scatter(reshape(gridRulesNum,1,[]), reshape(gridErrors(:,:,2),1,[]), "r*");
hold on;
grid on;
xlabel("Number of Rules");
ylabel("Error");
legend(errorNames(2));
title("RMSE relevant to Number of Rules");

% Errors / Features
[x,y]=meshgrid(featNum, cRadius);
figure();
surf(x,y,gridErrors(:,:,2)');
xlabel("Number of Features");
ylabel("Cluster Radius");
zlabel("RMSE");
title("RMSE to Grid Charachteristics");

%% Find Best Model and Train
gridRMSE(gridRMSE==0)=100;
[row, col] = find(gridRMSE == min(gridRMSE(:)));
optFeat = featNum(row);
optRad = cRadius(col);
fprintf("Best Model Charactheristics from OA(f=%d, r=%d)\n",optFeat,optRad);

trnOpt = [trnData(:,idx(1:optFeat)) trnTarget];
valOpt = [valData(:,idx(1:optFeat)) valTarget];
testOpt = [testData(:, idx(1:optFeat)) testTarget];

opt_fisOpt = genfisOptions("SubtractiveClustering","ClusterInfluenceRange",optRad);
opt_fis = genfis(trnOpt(:,1:end-1),trnTarget,opt_fisOpt);

opt_ANFISoptions = anfisOptions;
opt_ANFISoptions.EpochNumber = 100;
opt_ANFISoptions.ValidationData = valOpt;
opt_ANFISoptions.InitialFIS = opt_fis;

[opt_trnFis,opt_trnError,opt_stepSize,opt_valFis,opt_valError] = anfis(trnOpt, opt_ANFISoptions);

opt_ypred = evalfis(opt_valFis, testOpt(:,1:end-1));
%% Optimal Model Plots
%Plot MF
for l = 1:length(opt_trnFis.input)
   figure;
   [xmf, ymf] = plotmf(opt_fis, 'input', l);
   plot(xmf, ymf);
   xlabel('Input');
   ylabel('Degree of membership');
   title(['Input #' num2str(l) "Before Training"]);
   figure;
   [xmf, ymf] = plotmf(opt_trnFis, 'input', l);
   plot(xmf, ymf);
   xlabel('Input');
   ylabel('Degree of membership');
   title(['Input' num2str(l) "After Training"]);
end

%Learning Curve
figure();
plot([opt_trnError, opt_valError], "LineWidth",2); grid on;
legend("Training Error", "Validation Error");
xlabel("# of Epochs");
ylabel("Error");
title("Optimal Model Learning Curve");

%Predictions
figure();
hold on;
title('Prediction');
xlabel('Test Dataset Sample');
ylabel("Value");
plot(1:length(opt_ypred), opt_ypred, 'x','Color','red');

figure();
hold on;
title('Real Values');
xlabel('Test Dataset Sample');
ylabel('Value');
plot(1:length(opt_ypred), testOpt(:,end), 'o','Color','blue');

figure();
hold on;
title("Prediction Errors");
xlabel('Test Dataset Sample');
ylabel('Error');
plot(1:length(opt_ypred), opt_ypred-testOpt(:,end));

%Calculate Metrics 
opt_metrics = [MSE(opt_ypred, testOpt(:,end))
               RMSE(opt_ypred, testOpt(:,end))
               NMSE(opt_ypred, testOpt(:,end))
               NDEI(opt_ypred, testOpt(:,end))
               R2(opt_ypred, testOpt(:,end))];

opt_metrics = array2table(opt_metrics,'VariableNames',"Optimal Model",'Rownames',errorNames);
disp(opt_metrics);