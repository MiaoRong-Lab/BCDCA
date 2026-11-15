function smoke_test_bcdca
% Small integration check for the PlatEMO implementation.
releaseRoot = fileparts(fileparts(mfilename('fullpath')));
sourceRoot = fullfile(releaseRoot, 'src');
addpath(genpath(sourceRoot));

if exist('MW1', 'class') ~= 8
    error('BCDCA:PlatEMONotOnPath', ...
        'Add the PlatEMO root and its subfolders to the MATLAB path first.');
end
assert(contains(which('BCDCOA_Revision'), sourceRoot), ...
    'MATLAB resolved BCDCOA_Revision outside this package.');
requiredClasses = {'FSBBR_Revision','FSBBR_Controller','BRACE','DRIFT', ...
    'RIFT','SIFT','TIDE'};
for i = 1:numel(requiredClasses)
    resolvedPath = which(requiredClasses{i});
    assert(~isempty(resolvedPath) && contains(resolvedPath, sourceRoot), ...
        'MATLAB did not resolve %s from this package.', requiredClasses{i});
end
sourceText = fileread(which('BCDCOA_Revision'));
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
algorithm = BCDCOA_Revision('parameter', {0.30, 2, 0.25}, ...
    'save', 0, 'outputFcn', @quietOutput);
algorithm.Solve(problem);

trace = algorithm.auditTrace;
assert(problem.FE >= problem.maxFE, 'The run stopped before maxFE.');
assert(strcmp(algorithm.modeCode, expectedMode), ...
    'Expected mode %s, got %s.', expectedMode, algorithm.modeCode);
assert(~isempty(trace), 'The algorithm produced no allocation trace.');
assert(all([trace.variantCode] == 2), 'The run used an unexpected variant.');
assert(all([trace.q] == [trace.scarcity]), ...
    'Variant 2 did not use feasibility scarcity.');
assert(all([trace.constraintCount] + [trace.relaxCount] == [trace.budget]), ...
    'A controlled offspring budget was not conserved.');

fprintf('%s: mode=%s, FE=%d, generations=%d.\n', ...
    class(problem), algorithm.modeCode, problem.FE, numel(trace));
end

function quietOutput(~, ~)
end
