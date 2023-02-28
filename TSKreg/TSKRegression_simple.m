%% Clear workspace 
clc
clear 
close all

%% Load and Split Data
data = importdata('Datasets/airfoil_self_noise.dat');
preproc = 1;
[trnData, valData, testData] = split_scale(data,preproc);

trnTarget = trnData(:, end);
valTarget = valData(:, end);
testTarget = testData(:, end);

nrModels = 4;
metrics=zeros(nrModels,4);

%% Evaluation functions
MSE = @(ypred, y) mse(ypred,y);
RMSE = @(ypred, y) sqrt(MSE(ypred,y));
R2 = @(ypred,y) 1-sum((y-ypred).^2)/sum((y-mean(y)).^2);
NMSE = @(ypred, y) 1 - R2(ypred,y);
NDEI = @(ypred,y) sqrt(NMSE(ypred,y));

%% FIS with grid partition
%setup options
fisOpt = genfisOptions('GridPartition');

ANFISoptions = anfisOptions;
ANFISoptions.EpochNumber = 100;
ANFISoptions.ValidationData = valData;

for model=1:nrModels
    fisOpt.NumMembershipFunctions = 3 - mod(model,2);
    fisOpt.InputMembershipFunctionType = 'gbellmf';
    switch model
        case {1, 2}
            fisOpt.OutputMembershipFunctionType = 'constant';
        case {3, 4}
            fisOpt.OutputMembershipFunctionType = 'linear';
    end
    fis(model) = genfis(trnData(:,1:end-1),trnTarget,fisOpt);

    % Plot Input Membership Functions
    figure(model);
    [x, mf]= plotmf(fis(model), 'input', 1);
    subplot(2,1,1);
    plot(x,mf);
    xlabel("input1 (gbellmf)");
    subplot(2,1,2);
    [x,mf] = plotmf(fis(model), 'input',2);
    plot(x,mf);
    xlabel("input2 (gbellmf)");
    
    %ANFIS
    ANFISoptions.InitialFIS = fis(model);
    [trnFis,trnError,stepSize,valFis,valError] = anfis(trnData, ANFISoptions);
    
    %Plot Output Membership Functions
    for i=1:size(trnData,2)-1
        figure(10*model + i);
        plotmf(valFis, 'input',i);
        title(['TSK Model' int2str(model) ' input' int2str(i)]);
    end

    figure(101*model);
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    title(['TSK Model' int2str(model) ' learning curve']);

    ypred=evalfis(valFis,testData(:,1:end-1));
    metrics(:,model) = [RMSE(ypred, testTarget) 
        NMSE(ypred, testTarget) 
        NDEI(ypred, testTarget) 
        R2(ypred,testTarget)].';

    %Plot Prediction Error
    predError = testTarget - ypred;
    figure(111*model);
    plot(predError,'LineWidth',2); grid on;
    xlabel('input'); ylabel('absolute error');
    title(['TSK Model' int2str(model) ' prediction error'])
    
end

%% Results
varnames={'TSK_model_1','TSK_model_2','TSK_model_3','TSK_model_4'};
rownames = {'RMSE','NMSE','NDEI','R2'};
metrics= array2table(metrics,'VariableNames', varnames, 'Rownames',rownames);

