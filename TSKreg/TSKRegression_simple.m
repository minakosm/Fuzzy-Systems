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
    figure();
    for i=1:size(data,2)-2
        [x, mf]= plotmf(fis(model), 'input', i);
        subplot(3,2,i);
        plot(x,mf);
        xlabel("input" + int2str(i)+ "(gbellmf)");
    end
        [x, mf]= plotmf(fis(model), 'input', 5);
        subplot(3,2,[5 6]);
        plot(x,mf);
        xlabel("input5 (gbellmf)");
    
    %ANFIS
    ANFISoptions.InitialFIS = fis(model);
    [trnFis,trnError,stepSize,valFis,valError] = anfis(trnData, ANFISoptions);
    
    %Plot Membership Functions Of Input Variables
    for i=1:size(trnData,2)-1
        figure();
        plotmf(valFis, 'input',i);
        title(['TSK Model' int2str(model) ' input' int2str(i)]);
    end

    figure();
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    title(['TSK Model' int2str(model) ' Learning Curve']);

    ypred=evalfis(valFis,testData(:,1:end-1));
    metrics(:,model) = [MSE(ypred, testTarget)
                        RMSE(ypred, testTarget) 
                        NMSE(ypred, testTarget) 
                        NDEI(ypred, testTarget) 
                        R2(ypred,testTarget)]';

    numOfRules(model) = size(valFis.Rules,2);

    %Plot Prediction Error
    predError = ypred-testTarget;
    figure();
    plot(predError,'LineWidth',2); grid on;
    xlabel('Input'); ylabel('absolute error');
    title(['TSK Model' int2str(model) ' Prediction Error'])
    
end

%% Results
varnames={'TSK_model_1','TSK_model_2','TSK_model_3','TSK_model_4'};
rownames = {'MSE', 'RMSE','NMSE','NDEI','R2'};
metrics= array2table(metrics,'VariableNames', varnames, 'Rownames',rownames);
disp(metrics);

numOfRules = array2table(numOfRules,"VariableNames",varnames, 'RowNames',"Number of Rules");
disp(numOfRules);