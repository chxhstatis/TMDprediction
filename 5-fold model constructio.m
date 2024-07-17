clc;
clear;
%% load data
pathToRecordingsFolder = fullfile('tmj5');
location = pathToRecordingsFolder;
ads = audioDatastore(location);
ads.Labels = helpergenLabels(ads);
summary(ads.Labels); 

sf = waveletScattering('SignalLength',800,'SamplingFrequency',50);

%cross validatation
numFolds = 5;
sensitivities = zeros(numFolds, 1);
specificities = zeros(numFolds, 1);
FPR = zeros(numFolds, 1);
testAccuracy = zeros(numFolds, 1);
AUC = zeros(numFolds, 1);
score=zeros(31, numFolds)
label=zeros(31, numFolds)

cv = cvpartition(numel(ads.Files), 'KFold', numFolds);

fold=1
    rng default;
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    adsTrain = subset(ads, trainIdx);
    adsTest = subset(ads, testIdx);
    countEachLabel(adsTrain)
    countEachLabel(adsTest)
%% normalization
Xtrain = [];
scatds_Train = transform(adsTrain,@(x)helperReadSPData(x));
while hasdata(scatds_Train)
    smat = read(scatds_Train);
    Xtrain = cat(2,Xtrain,smat);
end

Xtest = [];
scatds_Test = transform(adsTest,@(x)helperReadSPData(x));
while hasdata(scatds_Test)
    smat = read(scatds_Test);
    Xtest = cat(2,Xtest,smat);
end

% Wavelet scattering transform
Strain = sf.featureMatrix(Xtrain);
Stest = sf.featureMatrix(Xtest);
TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(mean(TrainFeatures,2));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(mean(TestFeatures,2));

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(num2cell(TrainFeatures,[1 2]));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(num2cell(TestFeatures, [1 2]));

% 构造一个包含 512 个隐含层的简单 LSTM 网络
[inputSize, ~] = size(TrainFeatures{1});
YTrain = adsTrain.Labels;

numHiddenUnits = 512;
numClasses = numel(unique(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% hyperparameters
maxEpochs = 300;
miniBatchSize = 14;

options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Shuffle','every-epoch',...
    'Verbose', false, ...
    'Plots','training-progress');
%% train LSTM network
net = trainNetwork(TrainFeatures,YTrain,layers,options);

%% the testAccuracy of LSTM
predLabels = classify(net,TestFeatures);
testAccuracy(fold) = sum(predLabels==adsTest.Labels)/numel(predLabels)*100

predLabels_1 = double(predLabels); %tmj4为40，tmj3为29，tmj2与tmj1为14
true_Labels = double(adsTest.Labels); 

C = confusionmat(true_Labels, predLabels_1);
% extract True Positive, False Negative, False Positive and True Negative
TP = C(1,1); % True Positive
FN = C(1,2); % False Negative
FP = C(2,1); % False Positive
TN = C(2,2); % True Negative
% sensitivities and specificities
sensitivities(fold)  = TP / (TP + FN); % 灵敏度，TPR
specificities(fold) = TN / (TN + FP); % 特异度，1-FPR
FPR(fold) = 1 - specificities(fold)

out1=predict(net,TestFeatures)

score(:,fold)=out1(1:31,2)
label(:,fold)=adsTest.Labels(1:31)

[X1,Y1,T1,AUC(fold)]=perfcurve(adsTest.Labels,out1(:,2),"1")

%
fold=2
    rng default;
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    adsTrain = subset(ads, trainIdx);
    adsTest = subset(ads, testIdx);
    countEachLabel(adsTrain)
    countEachLabel(adsTest)

Xtrain = [];
scatds_Train = transform(adsTrain,@(x)helperReadSPData(x));
while hasdata(scatds_Train)
    smat = read(scatds_Train);
    Xtrain = cat(2,Xtrain,smat);
end

Xtest = [];
scatds_Test = transform(adsTest,@(x)helperReadSPData(x));
while hasdata(scatds_Test)
    smat = read(scatds_Test);
    Xtest = cat(2,Xtest,smat);
end


Strain = sf.featureMatrix(Xtrain);
Stest = sf.featureMatrix(Xtest);

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(mean(TrainFeatures,2));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(mean(TestFeatures,2));

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(num2cell(TrainFeatures,[1 2]));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(num2cell(TestFeatures, [1 2]));

[inputSize, ~] = size(TrainFeatures{1});
YTrain = adsTrain.Labels;

numHiddenUnits = 512;
numClasses = numel(unique(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


maxEpochs = 300;
miniBatchSize = 14;

options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Shuffle','every-epoch',...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(TrainFeatures,YTrain,layers,options);


predLabels = classify(net,TestFeatures);
testAccuracy(fold) = sum(predLabels==adsTest.Labels)/numel(predLabels)*100

predLabels_1 = double(predLabels); %tmj4为40，tmj3为29，tmj2与tmj1为14
true_Labels = double(adsTest.Labels); 

C = confusionmat(true_Labels, predLabels_1);

TP = C(1,1); % True Positive
FN = C(1,2); % False Negative
FP = C(2,1); % False Positive
TN = C(2,2); % True Negative

sensitivities(fold)  = TP / (TP + FN); % 灵敏度，TPR
specificities(fold) = TN / (TN + FP); % 特异度，1-FPR
FPR(fold) = 1 - specificities(fold)

out1=predict(net,TestFeatures)
score(:,fold)=out1(1:31,2)
label(:,fold)=adsTest.Labels(1:31)

[X2,Y2,T2,AUC(fold)]=perfcurve(adsTest.Labels,out1(:,2),"1")



fold=3

    rng default;
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    adsTrain = subset(ads, trainIdx);
    adsTest = subset(ads, testIdx);
    countEachLabel(adsTrain)
    countEachLabel(adsTest)

Xtrain = [];
scatds_Train = transform(adsTrain,@(x)helperReadSPData(x));
while hasdata(scatds_Train)
    smat = read(scatds_Train);
    Xtrain = cat(2,Xtrain,smat);
end

Xtest = [];
scatds_Test = transform(adsTest,@(x)helperReadSPData(x));
while hasdata(scatds_Test)
    smat = read(scatds_Test);
    Xtest = cat(2,Xtest,smat);
end


Strain = sf.featureMatrix(Xtrain);
Stest = sf.featureMatrix(Xtest);

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(mean(TrainFeatures,2));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(mean(TestFeatures,2));

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(num2cell(TrainFeatures,[1 2]));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(num2cell(TestFeatures, [1 2]));

[inputSize, ~] = size(TrainFeatures{1});
YTrain = adsTrain.Labels;

numHiddenUnits = 512;
numClasses = numel(unique(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


maxEpochs = 300;
miniBatchSize = 14;

options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Shuffle','every-epoch',...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(TrainFeatures,YTrain,layers,options);


predLabels = classify(net,TestFeatures);
testAccuracy(fold) = sum(predLabels==adsTest.Labels)/numel(predLabels)*100

predLabels_1 = double(predLabels); %tmj4为40，tmj3为29，tmj2与tmj1为14
true_Labels = double(adsTest.Labels); 

C = confusionmat(true_Labels, predLabels_1);

TP = C(1,1); % True Positive
FN = C(1,2); % False Negative
FP = C(2,1); % False Positive
TN = C(2,2); % True Negative

sensitivities(fold)  = TP / (TP + FN); % 灵敏度，TPR
specificities(fold) = TN / (TN + FP); % 特异度，1-FPR
FPR(fold) = 1 - specificities(fold)

out1=predict(net,TestFeatures)
score(:,fold)=out1(1:31,2)
label(:,fold)=adsTest.Labels(1:31)
[X3,Y3,T3,AUC(fold)]=perfcurve(adsTest.Labels,out1(:,2),"1")



fold=4
    rng default;
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    adsTrain = subset(ads, trainIdx);
    adsTest = subset(ads, testIdx);
    countEachLabel(adsTrain)
    countEachLabel(adsTest)

Xtrain = [];
scatds_Train = transform(adsTrain,@(x)helperReadSPData(x));
while hasdata(scatds_Train)
    smat = read(scatds_Train);
    Xtrain = cat(2,Xtrain,smat);
end

Xtest = [];
scatds_Test = transform(adsTest,@(x)helperReadSPData(x));
while hasdata(scatds_Test)
    smat = read(scatds_Test);
    Xtest = cat(2,Xtest,smat);
end


Strain = sf.featureMatrix(Xtrain);
Stest = sf.featureMatrix(Xtest);

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(mean(TrainFeatures,2));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(mean(TestFeatures,2));

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(num2cell(TrainFeatures,[1 2]));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(num2cell(TestFeatures, [1 2]));


[inputSize, ~] = size(TrainFeatures{1});
YTrain = adsTrain.Labels;

numHiddenUnits = 512;
numClasses = numel(unique(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


maxEpochs = 300;
miniBatchSize = 14;

options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Shuffle','every-epoch',...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(TrainFeatures,YTrain,layers,options);

predLabels = classify(net,TestFeatures);
testAccuracy(fold) = sum(predLabels==adsTest.Labels)/numel(predLabels)*100

predLabels_1 = double(predLabels); %tmj4为40，tmj3为29，tmj2与tmj1为14
true_Labels = double(adsTest.Labels); 

C = confusionmat(true_Labels, predLabels_1);

TP = C(1,1); % True Positive
FN = C(1,2); % False Negative
FP = C(2,1); % False Positive
TN = C(2,2); % True Negative

sensitivities(fold)  = TP / (TP + FN); % 灵敏度，TPR
specificities(fold) = TN / (TN + FP); % 特异度，1-FPR
FPR(fold) = 1 - specificities(fold)

out1=predict(net,TestFeatures)

score(:,fold)=out1(1:31,2)
label(:,fold)=adsTest.Labels(1:31)
[X4,Y4,T4,AUC(fold)]=perfcurve(adsTest.Labels,out1(:,2),"1")



fold=5

    rng default;
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    adsTrain = subset(ads, trainIdx);
    adsTest = subset(ads, testIdx);
    countEachLabel(adsTrain)
    countEachLabel(adsTest)

Xtrain = [];
scatds_Train = transform(adsTrain,@(x)helperReadSPData(x));
while hasdata(scatds_Train)
    smat = read(scatds_Train);
    Xtrain = cat(2,Xtrain,smat);
end

Xtest = [];
scatds_Test = transform(adsTest,@(x)helperReadSPData(x));
while hasdata(scatds_Test)
    smat = read(scatds_Test);
    Xtest = cat(2,Xtest,smat);
end


Strain = sf.featureMatrix(Xtrain);
Stest = sf.featureMatrix(Xtest);

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(mean(TrainFeatures,2));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(mean(TestFeatures,2));

TrainFeatures = Strain(2:end,:,:);
TrainFeatures = squeeze(num2cell(TrainFeatures,[1 2]));
TestFeatures = Stest(2:end,:,:);
TestFeatures = squeeze(num2cell(TestFeatures, [1 2]));

[inputSize, ~] = size(TrainFeatures{1});
YTrain = adsTrain.Labels;

numHiddenUnits = 512;
numClasses = numel(unique(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


maxEpochs = 300;
miniBatchSize = 14;

options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','shortest', ...
    'Shuffle','every-epoch',...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(TrainFeatures,YTrain,layers,options);


predLabels = classify(net,TestFeatures);
testAccuracy(fold) = sum(predLabels==adsTest.Labels)/numel(predLabels)*100

predLabels_1 = double(predLabels); %tmj4为40，tmj3为29，tmj2与tmj1为14
true_Labels = double(adsTest.Labels); 

C = confusionmat(true_Labels, predLabels_1);

TP = C(1,1); % True Positive
FN = C(1,2); % False Negative
FP = C(2,1); % False Positive
TN = C(2,2); % True Negative

sensitivities(fold)  = TP / (TP + FN); % 灵敏度，TPR
specificities(fold) = TN / (TN + FP); % 特异度，1-FPR
FPR(fold) = 1 - specificities(fold)

out1=predict(net,TestFeatures)
score(:,fold)=out1(1:31,2)
label(:,fold)=adsTest.Labels(1:31)
[X5,Y5,T5,AUC(fold)]=perfcurve(adsTest.Labels,out1(:,2),"1")


csvwrite('result/5f_sensitivities.csv', sensitivities);
csvwrite('result/5f_specificities.csv', specificities);
csvwrite('result/5f_testAccuracy.csv', testAccuracy);
csvwrite('result/5f_AUC.csv', AUC);

% calculate the mean testAccuracy
meanSensitivity = mean(sensitivities);
meanSpecificity = mean(specificities);
meanFPR = mean(FPR);
meantestAccuracy=mean(testAccuracy)
meanAUC=mean(AUC)
stdErrorSensitivity = std(sensitivities) / sqrt(numFolds);
stdErrorSpecificity = std(specificities) / sqrt(numFolds);
stdErrortestAccuracy=std(testAccuracy) / sqrt(numFolds);
stdErrorAUC=std(AUC) / sqrt(numFolds);

sensitivityCI = [meanSensitivity - 1.96 * stdErrorSensitivity, meanSensitivity + 1.96 * stdErrorSensitivity];
specificityCI = [meanSpecificity - 1.96 * stdErrorSpecificity, meanSpecificity + 1.96 * stdErrorSpecificity];
testAccuracyCI = [meantestAccuracy - 1.96 * stdErrortestAccuracy, meantestAccuracy + 1.96 * stdErrortestAccuracy];
AUCCI = [meanAUC - 1.96 * stdErrorAUC, meanAUC + 1.96 * stdErrorAUC];



disp(['Mean Sensitivity: ', num2str(meanSensitivity)]);
disp(['95% CI for Sensitivity: [', num2str(sensitivityCI(1)), ', ', num2str(sensitivityCI(2)), ']']);
disp(['Mean Specificity: ', num2str(meanSpecificity)]);
disp(['95% CI for Specificity: [', num2str(specificityCI(1)), ', ', num2str(specificityCI(2)), ']']);
disp(['Mean testAccuracy: ', num2str(meantestAccuracy)]);
disp(['95% CI for testAccuracy: [', num2str(testAccuracyCI(1)), ', ', num2str(testAccuracyCI(2)), ']']);
disp(['Mean AUC: ', num2str(meanAUC)]);
disp(['95% CI for AUC: [', num2str(AUCCI(1)), ', ', num2str(AUCCI(2)), ']']);

%for ROC
nx=min([numel(X1),numel(X2),numel(X3),numel(X4),numel(X5)])
ny=min([numel(Y1),numel(Y2),numel(Y3),numel(Y4),numel(Y5)])
nt=min([numel(T1),numel(T2),numel(T3),numel(T4),numel(T5)])

X_all=[X1(1:nx),X2(1:nx),X3(1:nx),X4(1:nx),X5(1:nx)]
Y_all=[Y1(1:nx),Y2(1:nx),Y3(1:nx),Y4(1:nx),Y5(1:nx)]
T_all=[T1(1:nx),T2(1:nx),T3(1:nx),T4(1:nx),T5(1:nx)]

csvwrite('result/X_all_5f14.csv', X_all);
csvwrite('result/Y_all_5f14.csv', Y_all);
csvwrite('result/T_all_5f14.csv', T_all);
csvwrite('result/label_all_5f14.csv', label);
csvwrite('result/score_all_5f14.csv', score);


%%%%
plot(X1,Y1)
hold on
plot(X2,Y2)
plot(X3,Y3)
plot(X4,Y4)
plot(X5,Y5)
legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification')
hold off


