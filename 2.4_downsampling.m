%% Wavelet Transform
BaseTime = [1.3 2];
TFSize = 170;
Trial = size(EpochData_tot,3);
TFData_cut = zeros(170,8000);
pvalue = [];
channel = size(EpochData_tot,1);

Con_TFData_cut = zeros(channel, 170, length(4001:20:6000), Trial); 

for ch = 1 : size(EpochData_tot,1)
    TFData = zeros(TFSize,(WinSize_tot(2)-WinSize_tot(1))*srate+1);
    TFcat =[];

    for i = 1:Trial

        % Skip the trials
        if i == 2 || i == 10 || i == 15 || i == 23 || i == 24
            continue;
        end

        Tmp = squeeze(EpochData_tot(ch,:,i));
        Tmp = abs(cwt(Tmp,srate./(1:TFSize), 'cmor2.48-1'));
        
        %baseline correction
        BaseMean = mean(Tmp(:,BaseTime(1)*srate+1:BaseTime(2)*srate), 2); %frequency,time(baseline)
        BaseStd = std(Tmp(:,BaseTime(1)*srate+1:BaseTime(2)*srate),[],2);
        for k = 1:TFSize
            Tmp(k,:) = (Tmp(k,:) - BaseMean(k))/BaseStd(k);
        end
        % Apply simple moving average smoothing to Tmp
        for k = 1:TFSize
            Tmp(k,:) = smoothdata(Tmp(k,:), 'movmean', [50]);
        end

        TFcat = cat(3,TFcat,Tmp);
        TFData = TFData + Tmp;

        % Downsampling
        downsampled_data = downsample(Tmp', 20)'; % Transpose to downsample along time dimension

        % Select the time range after downsampling
        selected_time_range = downsampled_data(:, 201:300); % Adjusted for 100 Hz sampling rate

        % Store in the final matrix
        Con_TFData_cut(ch, :, :, i) = selected_time_range;

    end

    TFData = TFData/Trial;
    TFData_cut = TFData(:,2001:10000);


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
    
    close all  
    %pause;
end
%save('TFData_cut');
% Save the final data matrix
save('B1_predicate_220412_KLN.mat', 'Con_TFData_cut');
