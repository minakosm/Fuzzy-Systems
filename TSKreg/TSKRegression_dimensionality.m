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
featNum = 1:2:9;
cRadius = 0.3:0.3:0.9;
k_folds = 5 ; 

gridErrors = zeros(length(featNum), length(cRadius), 4);
rulesNum = zeros(length(featNum), length(cRadius));

%[idx, weight] = fsrmrmr(data(:,1:end-1), dataTarget);
[idx, weight] = relieff(data(:,1:end-1),data(:,end),10, 'method','regression');

%Plot the weights of the most significant predictors
bar(weight(idx));
ylabel("Weights");

for f=featNum
    Dtrn = [trnData(:,idx(1:f)) trnTarget];
    Dtst = [testData(:,idx(1:f)) testTarget];
    for r=cRadius
        crossVal = cvpartition(Dtrn(:,end), "KFold",k_folds);
        CVerror = zeros(4, k_folds);

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
            ANFISopt.EpochNumber = 100;
            ANFISopt.ValidationData = DvalCV;
            
            [trnFis,trnError,stepSize,valFis,valError] = anfis(DtrnCV, ANFISopt);

            ypred = evalfis(valFis,Dtst(:,1:end-1));
            CVerror(:, k) = [RMSE(ypred, testTarget) 
                            NMSE(ypred, testTarget) 
                            NDEI(ypred, testTarget) 
                            R2(ypred,testTarget)];
        end
        indexes = [find(featNum == f) find(cRadius == r)];
        for i=1:4
            gridErrors(indexes(1),indexes(2),i) = mean(CVerror(i,:));
        end
        rulesNum(indexes(1),indexes(2)) = size(fis.Rules,2);
    end
end

%% Plot Results 

% Error relative to number of rules and number of features
figure('Position',[75 70 1400 680]);
% Flatten numRules and gridMSE
numRulesFlat = reshape(rulesNum,1,[]);
gridMSEFlat = reshape(gridMSE,1,[]);
subplot(1,2,1),scatter(numRulesFlat,sqrt(gridMSEFlat)), hold on;
xlabel('Number of Rules');
ylabel('RMSE');
title('Error Relative to Number of Rules');
subplot(1,2,2),boxplot(gridMSE',featNum);
xlabel('Number of Features');
ylabel('MSE');
title('Error Relative to Number of Features');