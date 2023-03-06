%% Clear workspace 
clc
clear 
close all

%% Load and Split Data
data = importdata('Datasets/haberman.data');
preproc = 1;
[trnData, valData, testData] = split_scale(data,preproc);

ra = [0.2 0.9];

numOfModels = 4;
OA = zeros(numOfModels,1);
PA = zeros(numOfModels,2);
UA = zeros(numOfModels,2);
kHat = zeros(numOfModels,1);
errorMatrix = zeros(2,2,numOfModels);
modelRules = zeros(numOfModels,1); 

fisOpt = genfisOptions('SubtractiveClustering');
%fisOpt.OutputMembershipFunctionType = 'constant';
%fisOpt.FISType = 'sugeno';

ANFISopt = anfisOptions;
ANFISopt.EpochNumber = 100;
ANFISopt.ValidationData = valData;

for radius=ra

    idx = [1+(find(ra==radius)-1)*2, 2+(find(ra==radius)-1)*2];
    %Clustering Per Class
    [c1,sig1]=subclust(trnData(trnData(:,end)==1,:),radius);
    [c2,sig2]=subclust(trnData(trnData(:,end)==2,:),radius);
    numOfRules=size(c1,1)+size(c2,1);
    
    %Build FIS from scratch
    fis1 = sugfis;

    %Add Input-Output
    for i=1:size(trnData,2)-1
        name_in = "in" + int2str(i);
        fis1 = addInput(fis1,[0,1], "Name",name_in);
    end
    fis1 = addOutput(fis1, [0,1], "Name", "out1");

    %Add Iput Membership Functions
    for i=1:size(trnData,2)-1
        var_name = "in" + int2str(i);
        for j=1:size(c1,1)    
            fis1 = addMF(fis1, var_name, "gaussmf", [sig1(i) c1(j,i)]);
        end
        for j=1:size(c2,1)
            fis1 = addMF(fis1, var_name, "gaussmf", [sig2(i) c2(j,i)]);
        end
    end

    %Add Output Membership Functions 
    params = [zeros(1, size(c1,1)) ones(1,size(c2,1))];
    for i=1:numOfRules
        fis1 = addMF(fis1, "out1", 'constant', params(i));
    end

    %Add FIS RuleBase
    ruleList = zeros(numOfRules, size(trnData,2));
    for i=1:size(ruleList,1)
        ruleList(i,:) = i;
    end
    ruleList = [ruleList, ones(numOfRules,2)];
    fis1 = addrule(fis1, ruleList);

    %Train - Evaluate ANFIS
    ANFISopt.InitialFIS = fis1;
    [trnFis,trnError,stepSize,valFis,valError] = anfis(trnData, ANFISopt);
    
    %Plot Training-Validation Errors
    figure();
    plot([trnError,valError], 'LineWidth',2); grid on;
    legend('Training Error', 'Validation Error');
    xlabel("# of Epochs");
    ylabel("Error");
    title("Class Dependent SC Training Error, r=" + num2str(radius));

    ypred = evalfis(valFis, testData(:,1:end-1));
    ypred = round(ypred);
    ypred = min(max(1,ypred), 2);

    %Plot Membership Functions
    for i=1:size(trnData,2)-1
        figure();
        plotmf(trnFis, "input", i);
        title("TSK Class Dependent r=" +num2str(radius)+ ", mf after training for input " + int2str(i));
    end
    
    errorMatrix(:,:,idx(1)) = confusionmat(testData(:,end), ypred);
    n = sum(errorMatrix(:,:,idx(1)), "all");
    TP = errorMatrix(1,1,idx(1));
    FP = errorMatrix(1,2,idx(1));
    TN = errorMatrix(2,2,idx(1));
    FN = errorMatrix(2,1,idx(1));

    OA(idx(1)) = (TP + TN) / (TP+FP+TN+FN);
    PA(idx(1),1) = TP/(FN + TP);
    PA(idx(1),2) = TN/(TN + FP);
    UA(idx(1),1) = TP/(TP + FP);
    UA(idx(1),2) = TN/(FN + TN);
    
    m = ((TP+FP)*(TP+FN) + (TN+FN)*(FP+TN));
    kHat(idx(1)) = (n*(TP+TN)-m)/(n^2-m);

    modelRules(idx(1)) = size(valFis.Rules,2);

    %Compare with Class-Independent SP
    
    fisOpt.ClusterInfluenceRange = radius;
    fis2 = genfis(trnData(:,1:end-1), trnData(:,end), fisOpt);
    
    ANFISopt.InitialFIS = fis2;
    [trnFis,trnError,~,valFis,valError] = anfis(trnData, ANFISopt);
    
    %Plot Training-Validation Errors
    figure();
    plot([trnError valError], 'LineWidth', 2); grid on;
    legend("Training Error", "Validation Error");
    xlabel("# of Epochs");
    ylabel("Error");
    title("Class Independent SC Training Error, r=" + num2str(radius));

    ypred = evalfis(valFis, testData(:,1:end-1));
    ypred = round(ypred);
    ypred = min(max(1,ypred), 2);

    %Plot Membership Functions
    for i=1:size(trnData,2)-1
        figure();
        plotmf(trnFis, "input", i);
        title("TSK Class Independent r=" +num2str(radius)+ ", mf after training for input " + int2str(i));
    end

    errorMatrix(:,:,idx(2)) = confusionmat(testData(:,end), ypred);
    n = sum(errorMatrix(:,:,idx(2)), "all");
    TP = errorMatrix(1,1,idx(2));
    FP = errorMatrix(1,2,idx(2));
    TN = errorMatrix(2,2,idx(2));
    FN = errorMatrix(2,1,idx(2));

    OA(idx(2)) = (TP + TN) / (TP+FP+TN+FN);
    PA(idx(2),1) = TP/(FN + TP);
    PA(idx(2),2) = TN/(TN + FP);
    UA(idx(2),1) = TP/(TP + FP);
    UA(idx(2),2) = TN/(FN + TN);
    
    m = ((TP+FP)*(TP+FN) + (TN+FN)*(FP+TN));
    kHat(idx(2)) = (n*(TP+TN)-m)/(n^2-m);

    modelRules(idx(2)) = size(valFis.Rules,2);
end

disp(errorMatrix);

