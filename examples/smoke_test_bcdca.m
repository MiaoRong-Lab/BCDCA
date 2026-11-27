function smoke_test_bcdca
% Small integration check for the PlatEMO implementation.
releaseRoot = fileparts(fileparts(mfilename('fullpath')));
sourceRoot = fullfile(releaseRoot, 'src');
addpath(genpath(sourceRoot));

if exist('MW1', 'class') ~= 8
    error('BCDCA:PlatEMONotOnPath', ...
        'Add the PlatEMO root and its subfolders to the MATLAB path first.');
end
assert(contains(which('BCDCA'), sourceRoot), ...
    'MATLAB resolved BCDCA outside this package.');
sourceText = fileread(which('BCDCA'));
firstLine = regexp(sourceText, '\r?\n', 'split');
assert(contains(firstLine{1}, '< ALGORITHM'), ...
    'PlatEMO GUI cannot identify the BCDCA entry class.');

runCase(MW1('N', 30, 'maxFE', 300), 'low');
runCase(LIRCMOP1('N', 30, 'maxFE', 300), 'soft');
runCase(SDC1('N', 30, 'maxFE', 300), 'hard');
fprintf('BCDCA smoke tests passed for all three search modes.\n');
end

function runCase(problem, expectedMode)
rng(1, 'twister');
algorithm = BCDCA( ...
    'save', 0, 'outputFcn', @quietOutput);
algorithm.Solve(problem);

trace = algorithm.auditTrace;
assert(problem.FE >= problem.maxFE, 'The run stopped before maxFE.');
assert(strcmp(algorithm.modeCode, expectedMode), ...
    'Expected mode %s, got %s.', expectedMode, algorithm.modeCode);
assert(~isempty(trace), 'The algorithm produced no allocation trace.');
assert(all([trace.scarcity] == 1 - [trace.feasibleRate]), ...
    'The allocation state does not match the measured feasible rate.');
assert(all([trace.constraintCount] + [trace.relaxCount] == [trace.budget]), ...
    'A controlled offspring budget was not conserved.');

fprintf('%s: mode=%s, FE=%d, generations=%d.\n', ...
    class(problem), algorithm.modeCode, problem.FE, numel(trace));
end

function quietOutput(~, ~)
end
