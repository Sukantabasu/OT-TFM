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

% File: Step4_ScatterPlots.m
% ===========================
% Author: Sukanta Basu (University at Albany)
% Date: March 1, 2026
% Description: Density-colored scatter plots of observed vs. predicted
%     log10(Cn2) for either TabPFNv2 or TabDPT, at a selected training
%     sample size (3 or 18 days). Reproduces Figure 2 of the paper.
%
% Associated Publication:
%     S. Basu, "Leveraging deep learning-based foundation models for
%     optical turbulence (Cn2) estimation under data scarcity,"
%     Applied Optics, https://doi.org/10.1364/AO.585045
%
% Configuration:
%     optMod  - 0 = TabPFNv2, 1 = TabDPT
%     nDays   - 3 or 18; selects the pred_3days or pred_18days column
%               from the predictions CSV, corresponding to the top and
%               bottom panels of Figure 2 in the paper, respectively.
%
% Inputs:
%     predictions_TabPFN.csv  - CSV output from Step2_TFM.py with
%                               optMod = 0. Must contain columns:
%                               observed, pred_3days, pred_18days.
%     predictions_TabDPT.csv  - CSV output from Step2_TFM.py with
%                               optMod = 1. Must contain the same
%                               column structure.
%     Both files are read from DATA_DIR (FinalResults/).
%
% Processing:
%     - Computes a 2D histogram (50 bins) of observed vs. predicted
%       log10(Cn2) values and assigns each scatter point a density
%       color derived from its 2D bin count.
%     - Overlays a 1:1 reference line and annotates R² in the plot.
%     - Axis limits are fixed to [-17, -12] on both axes, consistent
%       with the dynamic range reported in the paper.
%     - Colorbar is capped at a density of 100 for visual clarity.
%
% Output (written to FIG_DIR):
%     HawaiiCn2_Scatter_TabPFN_3d.eps   - TabPFNv2, 3-day training
%     HawaiiCn2_Scatter_TabPFN_18d.eps  - TabPFNv2, 18-day training
%     HawaiiCn2_Scatter_TabDPT_3d.eps   - TabDPT, 3-day training
%     HawaiiCn2_Scatter_TabDPT_18d.eps  - TabDPT, 18-day training
%
% AI Assistance: Claude AI (Anthropic) was used for documentation,
%     code restructuring, and performance optimization.

clear; close all; clc;

% Input & output directories

ROOT_DIR = '/Users/sukantabasu/Dropbox/Priority/Projects/JDETO/Papers/2025_HawaiiCn2_TabPFN/';
DATA_DIR = [ROOT_DIR 'FinalResults/'];
FIG_DIR = [ROOT_DIR 'Figures/'];

optMod = 1; 
nDays = 3;

% Read the data
if optMod == 0
    data = readtable([DATA_DIR 'predictions_TabPFN.csv']);
else
    data = readtable([DATA_DIR 'predictions_TabDPT.csv']);
end

% Extract observed and predicted values (in log10 space)
cn2_Obs = data.observed;
if nDays == 18
    cn2_Pred = data.pred_18days;
elseif nDays == 3
    cn2_Pred = data.pred_3days;
end

% Remove NaN values
valid_idx = ~isnan(cn2_Obs) & ~isnan(cn2_Pred);
cn2_Obs = cn2_Obs(valid_idx);
cn2_Pred = cn2_Pred(valid_idx);

% Create figure for TabPFN/TabDPT
figure('Position', [100, 100, 800, 700]);

% Create hexbin plot via 2D histogram for density
[N, Xedges, Yedges] = histcounts2(cn2_Obs, cn2_Pred, 50);

% Get bin indices for each point
[~, ~, binX] = histcounts(cn2_Obs, Xedges);
[~, ~, binY] = histcounts(cn2_Pred, Yedges);

% Get density for each point
density = zeros(size(cn2_Obs));
for i = 1:length(cn2_Obs)
    if binX(i) > 0 && binX(i) <= size(N,1) && binY(i) > 0 && binY(i) <= size(N,2)
        density(i) = N(binX(i), binY(i));
    end
end

% Create scatter plot colored by density
scatter(cn2_Obs, cn2_Pred, 10, density, 'filled');
colormap('hot');
c = colorbar; 
c.Limits = [0 100];
ylabel(c, 'Density', 'FontSize', 12, 'FontWeight', 'bold');

hold on;

% Add 1:1 line
min_val = min([min(cn2_Obs), min(cn2_Pred)]);
max_val = max([max(cn2_Obs), max(cn2_Pred)]);
plot([min_val, max_val], [min_val, max_val], 'b--', 'LineWidth', 2, 'DisplayName', '1:1 Line');

% Calculate R²
SS_res = sum((cn2_Obs - cn2_Pred).^2);
SS_tot = sum((cn2_Obs - mean(cn2_Obs)).^2);
R2 = 1 - (SS_res / SS_tot);

% Add R² text
text(0.05, 0.95, sprintf('R^2 = %.2f', R2), ...
    'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black', 'VerticalAlignment', 'top');

% Formatting
xlabel('Observed log_{10}(C_n^2) [m^{-2/3}]', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Predicted log_{10}(C_n^2) [m^{-2/3}]', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
axis equal;

set(gca,'FontSize',14);
set(gca,'XLim',[-17, -12]);
set(gca,'YLim',[-17, -12]);
set(gca,'XTick',[-17:1:-12],'XTickLabel',num2str([-17:1:-12].'));
set(gca,'YTick',[-17:1:-12],'YTickLabel',num2str([-17:1:-12].'));

hold off;
set(gcf, 'Color', 'w');

if optMod == 0
    if nDays == 18
        print(gcf, [FIG_DIR 'HawaiiCn2_Scatter_TabPFN_18d.eps'], '-depsc');
    elseif nDays == 3
        print(gcf, [FIG_DIR 'HawaiiCn2_Scatter_TabPFN_3d.eps'], '-depsc'); 
    end
else
    if nDays == 18
        print(gcf, [FIG_DIR 'HawaiiCn2_Scatter_TabDPT_18d.eps'], '-depsc');
    elseif nDays == 3
        print(gcf, [FIG_DIR 'HawaiiCn2_Scatter_TabDPT_3d.eps'], '-depsc'); 
    end
end    
