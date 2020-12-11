
% load the annotations File

% Import the annotations.csv using import data in matlab. and import column
% V as column vector and also column Z as column vector.

% Get the image indices for training data

trainimgs_idx = CharacteristicsIndividual(1:1144,1 ) ; % Column V data

% Get the image indices for test data

testimgs_idx = CharacteristicsIndividual(1145:end,1);  % Column V data

% Get the patient indices for training data 

trainpatient_idx = unique(trainimgs_idx);

% Get the patient indices for testing data

testpatient_idx = unique(testimgs_idx);

% Make logical labels

for i=1:size(ExperimentalConditionDiagnosis,1)
    
    if i<=1144
        if ExperimentalConditionDiagnosis(i,1) == "chronic heart failure"
            trainlabel(i,1)=1;
        else
            trainlabel(i,1)=0;
        end
    else
        if ExperimentalConditionDiagnosis(i,1) == "chronic heart failure"
            testlabel(i-1144,1)=1;
        else
            testlabel(i-1144,1)=0;
        end
    end
end

% Split for validation data

indices = false(1144,1);   indices(1:165,1) = 1;  indices(551:715,1)=1; 

valimgs_idx = trainimgs_idx(indices,1);vallabel = trainlabel(indices,1);
valpatient_idx = [trainpatient_idx(1:15,1);trainpatient_idx(51:65,1)]  ;
    
% Now remaining data is for training

trainimgs_idx = trainimgs_idx(~indices,1); trainlabel = trainlabel(~indices,1);
trainpatient_idx = [trainpatient_idx(16:50,1);trainpatient_idx(66:104,1)];

% Accumulate all at one place

traindata = [trainimgs_idx, trainlabel];
testdata  = [testimgs_idx , testlabel ];valldata  = [valimgs_idx  , vallabel  ];



save('MetaData.mat','traindata','testdata','valldata','trainpatient_idx','valpatient_idx','testpatient_idx');







