
data = load('MetaData.mat');


direc = './Training Data/';
train_images = dir(fullfile(direc,'*.png'));      
nfiles = length(train_images);   
idx1=1;
idx2=1;
for i=1:nfiles
   if i<=165 || (i>=551 && i<=715) 
       currentfilename = train_images(i).name;
       file = fullfile(direc,currentfilename);currentimage = imread(file);valimgs(idx1,:,:,:)= currentimage;
       idx1=idx1+1;
   else
       currentfilename = train_images(i).name;
       file = fullfile(direc,currentfilename);currentimage = imread(file);trainimgs(idx2,:,:,:)= currentimage;
       idx2=idx2+1;
   end
       
end
direc = './Testing Data/';
test_images = dir(fullfile(direc,'*.png'));      
nfiles = length(test_images);   
for i=1:nfiles
   currentfilename = test_images(i).name;
   file = fullfile(direc,currentfilename);currentimage = imread(file);testimgs(i,:,:,:)= currentimage;
end

trainlabel = data.traindata(:,2);testlabel = data.testdata(:,2);vallabel = data.valldata(:,2);
save('dataset250.mat','trainimgs','testimgs','valimgs','trainlabel','testlabel','vallabel');
