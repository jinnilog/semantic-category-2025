
eeglab
%%
file = loadcnt('230413_PSH_Overt.cnt');
Data = file.data;

%% Data plot
fprintf('Bad channel rejection...\n');
for i = 1:size(Data,1) %107 channels
    plot(Data(i,:));
    title(num2str(i));
    pause;
end

%% Channel regection
Data([1,20,53:74,107],:) = [];
ChanList = 1:107;
ChanList([1,20,53:74,107]) = [];
fprintf('Done...\n'); 

%% Common Average Reference
fprintf('Common Average Reference...');
CA = mean(Data,1);
for i = 1:size(Data,1)
    Data(i,:) = Data(i,:) - CA;
end
fprintf('Done...\n');

%% Filtering
srate = 2000;

fprintf('Filtering..\n');
[B, A] = butter(4, 0.001, 'high');
fdata = zeros(size(Data));
for i = 1:size(Data,1)
    fdata(i,:) = filtfilt(B, A, double(Data(i,:)));
    fprintf('.');
end
fdata = NotchFilter_ryun(fdata,srate,60);
fdata = NotchFilter_ryun(fdata,srate,120);
fprintf('Notch');
fdata = NotchFilter_ryun(fdata,srate,180);
fprintf('Done..\n');

%% Epoching (channel, time, trial)
fprintf('Epoching...');
WinSize_bs = [-1 2]; % baseline
WinSize_wd = [1 4]; % word
WinSize_tot = [-1 5]; % total

load('PSH_sem_overt_onset', 'onset');
EpochData_bs = zeros(size(fdata,1),(WinSize_bs(2)-WinSize_bs(1))*srate+1,27);
EpochData_wd = zeros(size(fdata,1),(WinSize_wd(2)-WinSize_wd(1))*srate+1,27);
EpochData_tot = zeros(size(fdata,1),(WinSize_tot(2)-WinSize_tot(1))*srate+1,27);
%channel, sample number*srate+1 (including current time), trial

sf=2000;
Esample_bs = [WinSize_bs(1).*sf, WinSize_bs(2).*sf];
Esample_wd = [WinSize_wd(1).*sf, WinSize_wd(2).*sf];
Esample_tot = [WinSize_tot(1).*sf, WinSize_tot(2).*sf];

t = 1; % for total
a = 1; % for baseline
b = 1; % for word

for i=1:size(onset,1)
    if onset(i,2) == 1 && onset(i,4) == 1
        EpochData_tot(:,:,t) = fdata(:,onset(i,1)+Esample_tot(1):onset(i,1)+Esample_tot(2));
        t = t + 1;
    end
    if onset(i,2) == 1 && onset(i,4) == 1
        EpochData_bs(:,:,a) = fdata(:,onset(i,1)+Esample_bs(1):onset(i,1)+Esample_bs(2));
        a = a + 1;
    elseif onset(i,2) == 2 && onset(i,4) == 1
        EpochData_wd(:,:,b) = fdata(:,onset(i,1)+Esample_wd(1):onset(i,1)+Esample_wd(2));
        b = b + 1;
    end
end

fprintf('Done...\n');

save('ChanList','EpochData_tot','EpochData_bs','EpochData_wd','srate');

%% Wavelet Transform
BaseTime = [1.3 2];
TFSize = 170;
Trial = size(EpochData_tot,3);
TFData_cut = zeros(170,8000);
pvalue = [];

for ch = 1 : size(EpochData_tot,1)
    TFData = zeros(TFSize,(WinSize_tot(2)-WinSize_tot(1))*srate+1);
    TFcat =[];
    for i = 1:Trial
        Tmp = squeeze(EpochData_tot(ch,:,i));
        Tmp = abs(cwt(Tmp,srate./(1:TFSize), 'cmor2.48-1'));
        
        %baseline correction
        BaseMean = mean(Tmp(:,BaseTime(1)*srate+1:BaseTime(2)*srate), 2); %frequency,time(baseline)
        BaseStd = std(Tmp(:,BaseTime(1)*srate+1:BaseTime(2)*srate),[],2);
        for k = 1:TFSize
            Tmp(k,:) = (Tmp(k,:) - BaseMean(k))/BaseStd(k);
        end
        TFcat = cat(3,TFcat,Tmp);
        
        TFData = TFData + Tmp;
        %fprintf('.')
    end
    TFcat_cut = TFcat(:,2001:10000,:);
    %Baseline
    Base_HG1 = TFcat_cut(70:170,601:2000,:); % Baseline HG (70Hz-170Hz, 0.3~1sec, trials)
    %Overt Speech
    Speech_HG1 = TFcat_cut(70:170,2001:8000,:); % Overt speech HG (70Hz-170Hz, 1~4sec, trials)
    
    %Mean
    Mean_B1 = squeeze(mean(Base_HG1,1));
    Mean_S1 = squeeze(mean(Speech_HG1,1));

    Mean_B1 = squeeze(mean(Mean_B1,1));
    Mean_S1 = squeeze(mean(Mean_S1,1));
    
    Mean_B1 = Mean_B1';
    Mean_S1 = Mean_S1';

    %ttest
    [P,H] = signrank(Mean_B1,Mean_S1); 
    pvalue = cat(2,pvalue,[P,H]);
    
    %Box plot
    group = [ones(size(Mean_B1)); 2 * ones(size(Mean_S1))];
    boxplot([Mean_B1,Mean_S1],group,'Labels',{'Basline HG','Speech HG'})
    title('Ch', ChanList(ch))
    xlabel('Baseline vs. Overt speech')
    ylabel('Amplitude(mean)')
    fname = ['G:\Boxplot\230413_PSH\B_predicate\Ch',num2str(ChanList(ch)),'.png'];
    saveas(gca, fname, 'png')

    TFData = TFData/Trial;
    %fprintf('\n');

    TFData_cut = TFData(:,2001:10000);

    % For simple moving average
    for i = 1:size(TFData_cut, 1)
        TFData_cut(i, :) = smoothdata(TFData_cut(i, :), 'movmean', [50]);
    end

    % Plotting
    figure;
    pcolor(linspace(-1,3, 4*srate),1:TFSize,TFData_cut); shading interp; colormap('jet'); caxis([-1 1])
    set(gca, 'YScale', 'log') % Change the y-axis to a logarithmic scale
    
    xline(0, '--', "Word presentation");
    xline(-0.7, '--', "Baseline");
    
    % Adding horizontal lines at specified frequencies
    yline([4, 8, 12, 30, 50, 70, 110, 170], ':'); % Adds dotted horizontal lines at the specified frequencies
    
    % Set custom y-ticks
    yticks([0.5, 4, 8, 12, 30, 50, 70, 110, 170]);
    yticklabels({'0.5', '4', '8', '12', '30', '50', '70', '110', '170'});
    
    title('Ch', ChanList(ch))
    xlabel('Time (s)')
    ylabel('Frequency (Hz)')
    
    colorbar; % Adds a colorbar to the plot
    
    fname = ['G:\CWT\230413_PSH\B_predicate\Ch',num2str(ChanList(ch)),'.png'];
    saveas(gca, fname, 'png')
    close all  
    %pause;
end
save('TFData_cut');
