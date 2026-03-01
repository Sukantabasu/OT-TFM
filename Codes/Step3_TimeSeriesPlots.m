% Copyright (c) 2026 Sukanta Basu
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

% File: Step3_TimeSeriesPlots.m
% ==============================
% Author: Sukanta Basu (University at Albany)
% Date: March 1, 2026
% Description: Time-series visualization of observed and predicted Cn2
%     at 15 m height during the July 2006 test period, overlaying
%     TabPFNv2 and TabDPT ensemble median predictions across four
%     consecutive weekly subplots. Reproduces Figure 1 of the paper.
%
% Associated Publication:
%     S. Basu, "Leveraging deep learning-based foundation models for
%     optical turbulence (Cn2) estimation under data scarcity,"
%     Applied Optics, https://doi.org/10.1364/AO.585045
%
% Inputs:
%     predictions_TabPFN.csv  - CSV output from Step2_TFM.py with
%                               optMod = 0. Must contain columns:
%                               date, observed, pred_18days.
%     predictions_TabDPT.csv  - CSV output from Step2_TFM.py with
%                               optMod = 1. Must contain the same
%                               column structure.
%     Both files are read from DATA_DIR (FinalResults/).
%
% Processing:
%     - Converts log10(Cn2) predictions and observations back to
%       linear Cn2 values for physical interpretability.
%     - Splits the July test period into four consecutive 7-day
%       windows and plots each as a separate subplot.
%     - Y-axis is displayed on a logarithmic scale spanning
%       1e-17 to 1e-12 m^(-2/3), consistent with the paper figures.
%
% Output:
%     HawaiiCn2_TimeSeries.eps - 4-panel time-series figure saved
%                                to FIG_DIR as an EPS file.
%
% AI Assistance: Claude AI (Anthropic) was used for documentation,
%     code restructuring, and performance optimization.

clear; close all; clc;

% Input & output directories

ROOT_DIR = '/Users/sukantabasu/Dropbox/Priority/Projects/JDETO/Papers/2025_HawaiiCn2_TabPFN/';
DATA_DIR = [ROOT_DIR 'FinalResults/'];
FIG_DIR = [ROOT_DIR 'Figures/'];

% Read the data
TabPFN = readtable([DATA_DIR 'predictions_TabPFN.csv']);
TabDPT = readtable([DATA_DIR 'predictions_TabDPT.csv']);

% Convert date strings to datetime
TabPFN.date = datetime(TabPFN.date, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

% Select 4 consecutive weeks of data
% Starting from the first date in the dataset
start_date = TabPFN.date(1);
week_duration = days(7);

% Create figure with 4 subplots (one for each week)
figure('Position', [100, 100, 1400, 900]);

for week = 1:4
    % Define the time range for this week
    week_start = start_date + (week-1)*week_duration;
    week_end = week_start + week_duration;
    
    % Extract data for this week
    week_mask = (TabPFN.date >= week_start) & (TabPFN.date < week_end);
    week_TabPFN = TabPFN(week_mask, :);
    week_TabDPT = TabDPT(week_mask, :);
    
    % Convert from log10(Cn2) to Cn2
    cn2_OBS = 10.^(week_TabPFN.observed);
    cn2_TabPFN = 10.^(week_TabPFN.pred_18days);
    cn2_TabDPT = 10.^(week_TabDPT.pred_18days);
    
    % Create subplot
    subplot(2, 2, week);
    hold on; grid on;
    
    % Plot observations using markers only (circles)
    plot(week_TabPFN.date, cn2_OBS, 'ko', 'MarkerSize', 2, ...
         'MarkerFaceColor', 'k', 'DisplayName', 'Observed');
    
    % Plot TabPFN & TabDPT predictions
    plot(week_TabPFN.date, cn2_TabPFN, 'rs', 'MarkerSize', 2, ...
         'MarkerFaceColor', 'r', 'DisplayName', 'TabPFNv2');
    plot(week_TabPFN.date, cn2_TabDPT, 'bp', 'MarkerSize', 2, ...
         'MarkerFaceColor', 'b', 'DisplayName', 'TabDPT');
    
    % Set y-axis to logarithmic scale
    set(gca, 'YScale', 'log');
    
    % Formatting
    xlabel('Date', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('C_n^2 [m^{-2/3}]', 'FontSize', 11, 'FontWeight', 'bold');
    % title(sprintf('Week %d: %s to %s', week, ...
    %               datestr(week_start, 'yyyy-mm-dd'), ...
    %               datestr(week_end, 'yyyy-mm-dd')), ...
    %       'FontSize', 12, 'FontWeight', 'bold');
    
    % Add legend
    legend('Location', 'best', 'FontSize', 10);
    
    % Improve datetime axis formatting
    xtickformat('MMM-dd HH:mm');
    xtickangle(45);
    
    % Set y-axis limits and ticks at powers of 10
    ylim([1e-17, 1e-12]);
    yticks([1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12]);
    
    hold off;
end

% Adjust spacing between subplots
set(gcf, 'Color', 'w');

% Save the figure
print(gcf, [FIG_DIR 'HawaiiCn2_TimeSeries.eps'], '-depsc');