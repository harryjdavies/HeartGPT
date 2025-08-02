% Specify the names of your main folders here

% download .mat files from CinC challenge 2020
% https://physionet.org/content/challenge-2020/1.0.2/

mainFolders = {'cpsc_2018', 'cpsc_2018_extra', 'georgia','ptb-xl'}; % replace with your folder names

ecg_store = zeros(42437,1000);
m = 0;
% Loop over each main folder
for i = 1:length(mainFolders)
    mainFolder = mainFolders{i};
    disp(mainFolder)
    
    % Get a list of all subfolders within the main folder
    subFolders = dir(fullfile(mainFolder, '*'));
    % Remove '.' and '..' directories
    subFolders = subFolders(~ismember({subFolders.name}, {'.', '..'}));
    
    % Loop over each subfolder
    for j = 1:length(subFolders)
        subFolder = subFolders(j).name;
        
        % Get a list of all .mat files within the subfolder
        matFiles = dir(fullfile(mainFolder, subFolder, '*.mat'));
        
        % Loop over each .mat file
        for k = 1:length(matFiles)
            matFile = matFiles(k).name;
            
            % Load the .mat file
            load(fullfile(mainFolder, subFolder, matFile));

            if length(val) > 4999
                data_temp = val(1,1:5000);

                ecg_sig_example = resample(data_temp,100,500);
                ecg_sig_example = round(100*rescale(ecg_sig_example),0);
                m = m + 1;
                ecg_store(m,:) = ecg_sig_example;

                disp(m)

            end
        end
    end
end