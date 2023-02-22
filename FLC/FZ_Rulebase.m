%% Clear command window and close all windows
clc
close all 

%% Create Mamdani Fuzzy Inference System
boundaries = [-1 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1];
mf_names = ["NV", "NL", "NM", "NS", "ZR", "PS", "PM", "PL", "PV"];
var_names = ["E", "dE", "dU"];

fis = mamfis;
fis = addInput(fis, [-1, 1], "Name", var_names(1));
fis = addInput(fis, [-1, 1], "Name", var_names(2));
fis = addOutput(fis, [-1, 1], "Name", var_names(3));

for i = var_names
    for j = 1:1:length(mf_names)
        fis = addMF(fis, i, "trimf", [boundaries(j) boundaries(j+1) boundaries(j+2)], "Name", mf_names(j));
    end
end

%% Create Rule List 
%We correspond a number with the names of the member functions with (1)
%being NV and (9) being PV
c = [5 4 3 2 1 1 1 1 1];
r = [5 6 7 8 9 9 9 9 9];
ruleBase = toeplitz(c, r);
ruleBaseStr = strings();  %String representation of the RuleBase
ruleList=[];

for i=1:length(ruleBase(1,:))
    for j=1:length(ruleBase(:,1))
        ruleList = [ruleList ; [i j ruleBase(length(ruleBase)+1-i, j) 1 1]];
        ruleBaseStr(i,j) = mf_names(ruleBase(i,j));
    end
end
fis = addRule(fis, ruleList);
%writeFIS(fis);