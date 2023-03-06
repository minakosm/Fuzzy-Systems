%% Clear workspace 
clc
clear 
close all

%% Load Data
data = importdata('Datasets/epileptic_seizure_data.csv');
data = data.data;
preproc = 1;
[trnData, valData, testData] = split_scale(data,preproc);

trnTarget = trnData(:,end);
valTarget = valData(:,end);
testTarget = testData(:,end);

valData(valData>1)=1;
valData(valData<0)=0;

testData(testData>1)=1;
testData(testData<0)=0;

ANFISopt = anfisOptions;
ANFISopt.EpochNumber = 50;

%% Grid Search
[idx, weight] = relieff(data(:,1:end-1),data(:,end),10, 'method','classification');

%Plot the weights of the most significant predictors
bar(weight(idx));
ylabel("Weights");

featNum = 5:3:15;
cRadius = 0.3:0.2:0.9;
k_folds = 5 ; 

gridOA = zeros(length(featNum), length(cRadius));
gridError = zeros(length(featNum), length(cRadius));
gridRulesNum = zeros(length(featNum), length(cRadius));

for f=featNum
    Dtrn = [trnData(:,idx(1:f)) trnTarget];
    Dtst = [testData(:,idx(1:f)) testTarget];
    for r=cRadius
        indexes = [find(featNum == f) find(cRadius == r)];
        crossVal = cvpartition(Dtrn(:,end), "KFold",k_folds); 
        OA_k = zeros(k_folds,1);
        CVerror = zeros(k_folds,1);
        rulesNum_k = zeros(k_folds,1);
        for k=1:k_folds
            DtrnCV = Dtrn(training(crossVal,k),:);
            DvalCV = Dtrn(test(crossVal,k),:);
            
            [c1,sig1]=subclust(Dtrn(Dtrn(:,end)==1,:),r);
            [c2,sig2]=subclust(Dtrn(Dtrn(:,end)==2,:),r);
            [c3,sig3]=subclust(Dtrn(Dtrn(:,end)==3,:),r);
            [c4,sig4]=subclust(Dtrn(Dtrn(:,end)==4,:),r);
            [c5,sig5]=subclust(Dtrn(Dtrn(:,end)==5,:),r);
            numOfRules = size(c1,1) + size(c2,1) + size(c3,1) + size(c4,1) + size(c5,1);

            %Build FIS from scratch
            fis = sugfis;
            %Add Input-Output Vaiables and Membership Functions
            for i=1:size(Dtrn,2)-1
                name_in = "in" + int2str(i);
                fis = addInput(fis,[0,1], "Name",name_in);
                for j=1:size(c1,1)    
                    fis = addMF(fis, name_in, "gaussmf", [sig1(i) c1(j,i)]);
                end
                for j=1:size(c2,1)
                    fis = addMF(fis, name_in, "gaussmf", [sig2(i) c2(j,i)]);
                end
                for j=1:size(c3,1)
                    fis = addMF(fis, name_in, "gaussmf", [sig3(i) c3(j,i)]);
                end
                for j=1:size(c4,1)
                    fis = addMF(fis, name_in, "gaussmf", [sig4(i) c4(j,i)]);
                end
                for j=1:size(c5,1)
                    fis = addMF(fis, name_in, "gaussmf", [sig5(i) c5(j,i)]);
                end
            end
            fis = addOutput(fis, [0,1], "Name", "out1");

            %Add Output Membership Variables
            params = [zeros(1,size(c1,1)) 0.25*ones(1,size(c2,1)) 0.5*ones(1,size(c3,1)) 0.75*ones(1,size(c4,1)) ones(1,size(c5,1))];
            for i=1:numOfRules
                fis = addMF(fis, "out1", 'constant', params(i));
            end
            %Add FIS RuleBase
            ruleList = zeros(numOfRules, size(DtrnCV,2));
            for i=1:size(ruleList,1)
                ruleList(i,:) = i;
            end
            ruleList = [ruleList, ones(numOfRules,2)];
            fis = addrule(fis, ruleList);

            %Train and Evaluate ANFIS
            ANFISopt.InitialFIS = fis;
            ANFISopt.ValidationData = DvalCV;
            fprintf("\nStart %dfold Training for %d Features and %d radius\n", k, f, r);
            [trnFis,trnError,stepSize,valFis,valError] = anfis(DtrnCV, ANFISopt);
            fprintf("END OF TRAINING\n");
        
            ypred = evalfis(valFis, Dtst(:, 1:end-1));
            ypred = round(ypred);
            ypred(ypred<1)=1;
            ypred(ypred>5)=5;

            errorMatrix = confusionmat(Dtst(:,end), ypred);
            OA_k(k) = trace(errorMatrix) / sum(errorMatrix, "all");
            rulesNum_k(k) = size(valFis.Rules,2);
            CVerror(k) = sqrt(mse(valError));
        end
        gridOA(indexes(1),indexes(2)) = mean(OA_k);
        gridError(indexes(1), indexes(2)) = mean(CVerror);
        gridRulesNum(indexes(1),indexes(2)) = mean(rulesNum_k);

    end
end

%% Plots
% OA / NumOfRules
figure();
hold on; grid on;
scatter(reshape(gridRulesNum,1,[]), reshape(gridOA,1,[]), "r*");
xlabel("Overall Accuracy");
ylabel("Number of Rules");
title("Overall Accuracy relevant to Number of Rules");

% OA / (numOfFeatures, cRadius)
figure();
[x,y] = meshgrid(featNum, cRadius); 
surf(x,y,gridOA');
xlabel("Number of Features");
ylabel("Cluster Radius");
zlabel("Overall Accuracy");
title("Overall Accuracy to Grid Charachteristics");

%% Find Best Model and Train
[optFeat, optRad] = find(gridOA == max(gridOA(:)));
optFeat = featNum(optFeat);
optRad = cRadius(optRad);

[optFeatError, optRadError] = find(gridError == min(gridError(:)));
optFeatError = featNum(optFeatError);
optRadError = cRadius(optRadError);

fprintf("Best Model Charactheristics from OA(f=%d, r=%d)\n",optFeat,optRad);
fprintf("Best Model Charactheristics from Mean Error (f=%d, r=%d)\n",optFeat,optRad);

trnOpt = [trnData(:, idx(1:optFeat)) trnTarget];
valOpt = [valData(:,idx(1:optFeat)) valTarget];
testOpt = [testData(:, idx(:,1:optFeat)) testTarget];

[opt_c1,opt_sig1] = subclust(trnOpt(trnOpt(:,end)==1,:), optRad);
[opt_c2,opt_sig2] = subclust(trnOpt(trnOpt(:,end)==2,:), optRad);
[opt_c3,opt_sig3] = subclust(trnOpt(trnOpt(:,end)==3,:), optRad);
[opt_c4,opt_sig4] = subclust(trnOpt(trnOpt(:,end)==4,:), optRad);
[opt_c5,opt_sig5] = subclust(trnOpt(trnOpt(:,end)==5,:), optRad);
opt_numOfRules = size(opt_c1,1) + size(opt_c2,1) + size(opt_c3,1) + size(opt_c4,1) + size(opt_c5,1);

opt_fis = sugfis;
opt_ANFISopt = anfisOptions;
opt_ANFISopt.EpochNumber = 100;
opt_ANFISopt.ValidationData = valOpt;

%Add Input-Output Vaiables and Membership Functions
 for i=1:size(trnOpt,2)-1
    name_in = "opt_in" + int2str(i);
    opt_fis = addInput(opt_fis,[0,1], "Name",name_in);
    for j=1:size(opt_c1,1)    
        opt_fis = addMF(opt_fis, name_in, "gaussmf", [opt_sig1(i) opt_c1(j,i)]);
    end
    for j=1:size(opt_c2,1)
        opt_fis = addMF(opt_fis, name_in, "gaussmf", [opt_sig2(i) opt_c2(j,i)]);
    end
    for j=1:size(opt_c3,1)
        opt_fis = addMF(opt_fis, name_in, "gaussmf", [opt_sig3(i) opt_c3(j,i)]);
    end
    for j=1:size(opt_c4,1)
        opt_fis = addMF(opt_fis, name_in, "gaussmf", [opt_sig4(i) opt_c4(j,i)]);
    end
    for j=1:size(opt_c5,1)
        opt_fis = addMF(opt_fis, name_in, "gaussmf", [opt_sig5(i) opt_c5(j,i)]);
    end
end
opt_fis = addOutput(opt_fis, [0,1], "Name", "opt_out");

%Add Output Membership Variables
opt_params = [zeros(1,size(opt_c1,1)) 0.25*ones(1,size(opt_c2,1)) 0.5*ones(1,size(opt_c3,1)) 0.75*ones(1,size(opt_c4,1)) ones(1,size(opt_c5,1))];
for i=1:opt_numOfRules
    opt_fis = addMF(opt_fis, "opt_out", 'constant', opt_params(i));
end
%Add FIS RuleBase
ruleList = zeros(opt_numOfRules, size(trnOpt,2));
for i=1:size(ruleList,1)
    ruleList(i,:) = i;
end
ruleList = [ruleList, ones(opt_numOfRules,2)];
opt_fis = addrule(opt_fis, ruleList);

%Train and Evaluate optANFIS
fprintf("\nTRAINING OPTIMAL MODEL\n");
opt_ANFISopt.InitialFIS = opt_fis;
[opt_trnFis,opt_trnError,opt_stepSize,opt_valFis,opt_valError] = anfis(trnOpt, opt_ANFISopt);
fprintf("\nEND OF TRAINING\n");

opt_ypred = evalfis(opt_valFis, testOpt(:,1:end-1));
opt_ypred = round(opt_ypred);
opt_ypred(opt_ypred<1)=1;
opt_ypred(opt_ypred>5)=5;

opt_errorMatrix = confusionmat(opt_ypred, testOpt(:,end));
opt_OA = trace(opt_errorMatrix)/sum(opt_errorMatrix,"all");
sumAct = sum(opt_errorMatrix);
sumPred = sum(opt_errorMatrix');
for i=1:5
    opt_PA(i) = opt_errorMatrix(i,i)/sumAct(i);
    opt_UA(i) = opt_errorMatrix(i,i)/sumPred(i);
end

opt_kHat = (sum(opt_errorMatrix,"all")*trace(opt_errorMatrix) - sum(sumPred.*sumAct) ) / (sum(opt_errorMatrix,"all")^2 - sum(sumPred.*sumAct) );

%% Opt Model Plots
%Plot MF
for l = 1:length(opt_trnFis.input)
   figure;
   [xmf, ymf] = plotmf(opt_fis, 'input', l);
   plot(xmf, ymf);
   xlabel('Input');
   ylabel('Degree of membership');
   title(['Input' num2str(l) "Before Training"]);
   figure;
   [xmf, ymf] = plotmf(opt_valFis, 'input', l);
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

