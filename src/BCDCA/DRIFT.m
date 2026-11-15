classdef DRIFT < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Feasible archive and region-transfer routines used by BCDCA
% archiveRatio --- 0.25 --- Near-feasible archive ratio
% refineRate   --- 0.10 --- Decision-region refinement ratio
% relaxPower   --- 2.5  --- Decay exponent of the promising-region threshold

    methods
        function main(Algorithm, Problem)
            [archiveRatio, refineRate, relaxPower] = Algorithm.ParameterSet(0.25, 0.10, 2.5);
            archiveRatio = min(0.70, max(0.10, archiveRatio));
            refineRate   = min(0.35, max(0.00, refineRate));
            relaxPower   = max(1.0, relaxPower);

            PopCon = Problem.Initialization();
            PopObj = PopCon;
            initCV = RIFT.fullCV(PopCon);
            initFeasRate = mean(initCV <= 1e-12);
            initMedianCV = median(initCV);
            consCount = size(PopCon.cons, 2);
            hardConstraintMode = Problem.D >= 30 && ...
                (consCount > Problem.M || ...
                (consCount >= Problem.M && initFeasRate < 0.20 && initMedianCV > 50));
            if hardConstraintMode
                DRIFT.regionTransferMode(Algorithm, Problem, PopCon, archiveRatio, refineRate, relaxPower);
                return;
            end
            PopReg = PopCon;
            PopFit = PopReg;
            Archive = PopCon(1:0);
            FeasArchive = PopCon(1:0);

            initCV = RIFT.fullCV([PopCon, PopReg, PopFit]);
            cv0 = max(initCV);
            if isempty(cv0) || cv0 <= 0
                cv0 = 1;
            end
            archiveSize = max(8, ceil(archiveRatio * Problem.N));
            iter = 0;

            while Algorithm.NotTerminated(PopCon)
                iter = iter + 1;
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                alpha = 2 ./ (1 + exp(-10 * phase)) - 1;
                para  = ceil(Problem.maxFE / Problem.N) / 2 - ceil(Problem.FE / Problem.N);
                epsilon = RIFT.relaxThreshold([PopCon, Archive], cv0, phase, relaxPower);

                if iter == 1 || mod(iter, 2) == 0
                    Archive = RIFT.updateArchive([PopCon, PopObj, PopReg, PopFit, Archive], archiveSize, epsilon, phase);
                end

                PopCon = PopCon(randperm(numel(PopCon)));
                PopObj = PopObj(randperm(numel(PopObj)));
                overlap = ismember(PopObj.objs, PopCon.objs, 'rows');
                gamma = 1 - sum(overlap) / max(1, Problem.N);

                lowFeasMode = hardConstraintMode && RIFT.feasibleRate(PopCon) < 0.20;
                [PopCon, RankCon] = DRIFT.selectionConstrained(PopCon, Problem.N, alpha, lowFeasMode);
                [PopObj, RankObj] = DRIFT.selectionObjective(PopObj, Problem.N, alpha, gamma, para);

                feasRate = RIFT.feasibleRate(PopCon);
                if hardConstraintMode && feasRate < 0.20
                    mainPool = RIFT.safePool([PopCon, PopFit, PopReg, RIFT.lowCVSample(Archive, ceil(0.20 * Problem.N))], PopCon);
                    [rankMain, crowdMain] = RIFT.matingRank(mainPool, 'main', 0, phase);
                    Offspring = RIFT.generateOffspring(Problem, mainPool, rankMain, crowdMain, ...
                        Problem.N, 0.16, 0.35);
                else
                    Pool = [PopCon, PopObj];
                    Rank = [RankCon, RankObj];
                    MatingPool = TournamentSelection(2, 2 * Problem.N, Rank);
                    Offspring = OperatorGAhalf(Problem, Pool(MatingPool));
                end

                if hardConstraintMode && (phase < 0.85 || feasRate < 0.60)
                    regPool = RIFT.safePool([PopReg, Archive, RIFT.lowCVSample(PopCon, ceil(0.25 * Problem.N)), ...
                        RIFT.lowCVSample(PopObj, ceil(0.15 * Problem.N))], PopReg);
                    [rankReg, crowdReg] = RIFT.matingRank(regPool, 'region', epsilon, phase);
                    if feasRate < 0.10
                        regCount = ceil(0.60 * Problem.N);
                    else
                        regCount = ceil(0.38 * Problem.N);
                    end
                    OffReg = RIFT.generateOffspring(Problem, regPool, rankReg, crowdReg, ...
                        regCount, 0.18, 0.12);
                else
                    OffReg = PopCon(1:0);
                end

                if hardConstraintMode && (phase < 0.90 || feasRate < 0.80)
                    fitPool = RIFT.safePool([PopFit, PopReg, Archive, RIFT.lowCVSample(PopCon, ceil(0.20 * Problem.N))], PopCon);
                    [rankFit, crowdFit] = RIFT.matingRank(fitPool, 'main', 0, phase);
                    if feasRate < 0.10
                        fitCount = ceil(0.55 * Problem.N);
                    else
                        fitCount = ceil(0.28 * Problem.N);
                    end
                    OffFit = RIFT.generateOffspring(Problem, fitPool, rankFit, crowdFit, ...
                        fitCount, 0.16, 0.18);
                else
                    OffFit = PopCon(1:0);
                end

                if phase < 0.70 || feasRate < 0.88
                    if hardConstraintMode && feasRate < 0.20
                        transferCount = ceil(0.24 * Problem.N);
                    else
                        transferCount = ceil((0.04 + 0.12 * max(0, 0.88 - feasRate)) * Problem.N);
                    end
                    transferPool = RIFT.safePool([RIFT.lowCVSample(Archive, ceil(0.35 * Problem.N)), ...
                        RIFT.lowCVSample(PopFit, ceil(0.25 * Problem.N)), ...
                        RIFT.lowCVSample(PopReg, ceil(0.30 * Problem.N)), PopCon, PopObj], PopCon);
                    [rankTransfer, crowdTransfer] = RIFT.matingRank(transferPool, 'transfer', 0.50 * epsilon, phase);
                    OffTransfer = RIFT.generateOffspring(Problem, transferPool, rankTransfer, crowdTransfer, ...
                        transferCount, 0.25, 0.08);
                else
                    OffTransfer = PopCon(1:0);
                end

                if phase > 0.20 && feasRate < 0.78
                    refineCount = ceil(refineRate * Problem.N * (0.78 - feasRate) / 0.78);
                    if hardConstraintMode && feasRate < 0.20
                        refineCount = max(refineCount, ceil(0.20 * Problem.N));
                    end
                else
                    refineCount = 0;
                end
                OffRefine = RIFT.generateRegionRefinement(Problem, ...
                    RIFT.safePool([Archive, PopFit, PopReg, PopCon, PopObj], PopCon), refineCount, phase);

                if hardConstraintMode && feasRate < 0.20
                    OffCVRescue = DRIFT.generateCVRescue(Problem, ...
                        RIFT.safePool([Archive, PopFit, PopReg, PopCon, PopObj], PopCon), ceil(0.20 * Problem.N), phase);
                    OffBlock = DRIFT.generateBlockRescue(Problem, ...
                        RIFT.safePool([Archive, PopFit, PopReg, PopCon, PopObj], PopCon), ceil(0.30 * Problem.N), phase);
                else
                    OffCVRescue = PopCon(1:0);
                    OffBlock = PopCon(1:0);
                end

                if hardConstraintMode && phase > 0.55 && feasRate < 0.10
                    polishPool = RIFT.safePool([PopFit, PopReg, Archive, PopCon], PopCon);
                    bestPolishCV = min(RIFT.fullCV(polishPool));
                    if phase > 0.75 && bestPolishCV < 0.05
                        polishCount = ceil(1.50 * Problem.N);
                    elseif phase > 0.75 && bestPolishCV < 0.20
                        polishCount = ceil(0.60 * Problem.N);
                    else
                        polishCount = ceil(0.32 * Problem.N);
                    end
                    OffPolish = DRIFT.generateCoordinatePolish(Problem, polishPool, polishCount, phase);
                else
                    OffPolish = PopCon(1:0);
                end

                if hardConstraintMode && phase > 0.35
                    highDPool = RIFT.safePool([PopCon, PopFit, PopReg, RIFT.lowCVSample(Archive, ceil(0.35 * Problem.N)), PopObj], PopCon);
                    [rankHighD, crowdHighD] = RIFT.matingRank(highDPool, 'main', 0, phase);
                    OffHighD = RIFT.generateOffspring(Problem, highDPool, rankHighD, crowdHighD, ...
                        ceil(0.32 * Problem.N), 0.18, 0.22);
                else
                    OffHighD = PopCon(1:0);
                end

                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive, PopCon, PopFit, PopReg, ...
                    Offspring, OffFit, OffReg, OffTransfer, OffRefine, OffCVRescue, OffBlock, ...
                    OffPolish, OffHighD, Archive], archiveSize, phase);

                archiveInject = RIFT.lowCVSample(Archive, ceil(0.18 * Problem.N));
                feasibleInject = RIFT.objectiveSample(FeasArchive, ceil(0.25 * Problem.N));
                conPool = [PopCon, PopFit, PopReg, Offspring, OffFit, OffReg, OffTransfer, OffRefine, ...
                    OffCVRescue, OffBlock, OffPolish, OffHighD, archiveInject, feasibleInject];
                if hardConstraintMode && phase > 0.25
                    lowFeasMode = RIFT.feasibleRate(conPool) < 0.30;
                    [driftCore, ~] = DRIFT.selectionConstrained(conPool, ceil(0.78 * Problem.N), alpha, lowFeasMode);
                    if lowFeasMode
                        fitCore = DRIFT.feasibilityBackboneSelection(conPool, ceil(0.72 * Problem.N), phase);
                        regCore = RIFT.regionSelection(conPool, ceil(0.45 * Problem.N), epsilon, phase);
                        PopCon  = DRIFT.feasibilityBackboneSelection([driftCore, fitCore, regCore], Problem.N, phase);
                    else
                        riftCore = RIFT.mainSelection(conPool, ceil(0.32 * Problem.N), phase);
                        [PopCon, ~] = DRIFT.selectionConstrained([driftCore, riftCore], Problem.N, alpha, lowFeasMode);
                    end
                else
                    [PopCon, ~] = DRIFT.selectionConstrained(conPool, Problem.N, alpha);
                end
                if hardConstraintMode
                    PopFit = DRIFT.feasibilityBackboneSelection([PopFit, OffFit, OffReg, OffTransfer, OffRefine, OffCVRescue, OffBlock, ...
                        OffPolish, OffHighD, FeasArchive, Archive, RIFT.lowCVSample(PopReg, ceil(0.25 * Problem.N)), ...
                        RIFT.lowCVSample(PopCon, ceil(0.20 * Problem.N))], Problem.N, phase);
                    PopReg = RIFT.regionSelection([PopReg, OffReg, OffTransfer, OffRefine, OffCVRescue, ...
                        OffBlock, OffPolish, OffFit, OffHighD, PopFit, FeasArchive, Archive, RIFT.lowCVSample(PopCon, ceil(0.25 * Problem.N))], ...
                        Problem.N, epsilon, phase);
                else
                    PopReg = PopCon;
                    PopFit = PopCon;
                end
                [PopObj, ~] = DRIFT.selectionObjective([PopObj, Offspring, OffTransfer, OffRefine, OffCVRescue, OffHighD, ...
                    RIFT.objectiveSample(FeasArchive, ceil(0.15 * Problem.N)), ...
                    RIFT.objectiveSample(Archive, ceil(0.18 * Problem.N))], Problem.N, alpha, gamma, para);
            end
        end
    end

    methods(Static)
        function regionTransferMode(Algorithm, Problem, PopMain, archiveRatio, ~, relaxPower)
            transferRate = 0.25;
            archiveRatio = max(0.55, min(0.80, archiveRatio));
            pBestRate = 0.15;

            PopRegion = Problem.Initialization();
            PopObj    = Problem.Initialization();
            Archive   = PopMain(1:0);
            FeasArchive = PopMain(1:0);

            initCV = RIFT.fullCV([PopMain, PopRegion, PopObj]);
            cv0 = max(initCV);
            if isempty(cv0) || cv0 <= 0
                cv0 = 1;
            end
            archiveSize = max(8, ceil(archiveRatio * Problem.N));
            PopOutput = PopMain;

            while Algorithm.NotTerminated(PopOutput)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                epsilon = RIFT.relaxThreshold([PopMain, PopRegion, Archive], cv0, phase, relaxPower);

                Archive = RIFT.updateArchive([PopMain, PopRegion, PopObj, Archive], archiveSize, epsilon, phase);

                mainPool = RIFT.safePool([PopMain, Archive], PopMain);
                [rankMain, crowdMain] = RIFT.matingRank(mainPool, 'main', 0, phase);
                OffMain = RIFT.generateOffspring(Problem, mainPool, rankMain, crowdMain, Problem.N, pBestRate, 0.20);

                mainFeasRate = RIFT.feasibleRate(PopMain);
                regionPool = RIFT.safePool([PopRegion, Archive, PopMain], PopRegion);
                [rankRegion, crowdRegion] = RIFT.matingRank(regionPool, 'region', epsilon, phase);
                OffRegion = RIFT.generateOffspring(Problem, regionPool, rankRegion, crowdRegion, ...
                    ceil(0.65 * Problem.N), pBestRate, 0.15);

                objPool = RIFT.safePool([PopObj, RIFT.lowCVSample(Archive, ceil(0.25 * Problem.N)), ...
                    RIFT.lowCVSample(PopMain, ceil(0.15 * Problem.N))], PopObj);
                [rankObj, crowdObj] = RIFT.matingRank(objPool, 'objective', inf, phase);
                OffObj = RIFT.generateOffspring(Problem, objPool, rankObj, crowdObj, ...
                    ceil(0.45 * Problem.N), pBestRate, 0.50);

                objBridge = RIFT.lowCVSample([PopObj, OffObj, Archive], ceil(0.35 * Problem.N));
                if phase > 0.30 && mainFeasRate < 0.65
                    refineCount = min(ceil(0.22 * Problem.N), ...
                        ceil((0.08 + 0.20 * (0.65 - mainFeasRate)) * Problem.N));
                else
                    refineCount = 0;
                end
                OffRefine = RIFT.generateRegionRefinement(Problem, ...
                    RIFT.safePool([Archive, objBridge, PopRegion, PopMain], PopMain), refineCount, phase);

                OffBlock = PopMain(1:0);
                OffPolish = PopMain(1:0);

                transferCount = max(2, ceil(transferRate * Problem.N * (1.15 - 0.65 * mainFeasRate)));
                transferCount = min(ceil(0.35 * Problem.N), transferCount);
                transferPool = RIFT.safePool([Archive, objBridge, OffRefine, OffPolish, OffBlock, PopRegion, PopMain], PopMain);
                [rankTransfer, crowdTransfer] = RIFT.matingRank(transferPool, 'transfer', 0.50 * epsilon, phase);
                OffTransfer = RIFT.generateOffspring(Problem, transferPool, rankTransfer, crowdTransfer, ...
                    transferCount, min(0.35, pBestRate + 0.10), 0.05);
                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive, PopMain, PopRegion, PopObj, ...
                    OffMain, OffRegion, OffObj, OffTransfer, OffRefine, OffPolish, Archive], archiveSize, phase);

                mainCandidates = [PopMain, OffMain, OffTransfer, OffRefine, OffPolish, OffBlock, Archive, ...
                    objBridge, RIFT.lowCVSample(OffRegion, ceil(0.25 * Problem.N))];
                PopMain = RIFT.mainSelection(mainCandidates, Problem.N, phase);

                PopRegion = RIFT.regionSelection([PopRegion, OffRegion, OffTransfer, OffRefine, OffPolish, OffBlock, OffMain, ...
                    objBridge, PopMain, Archive], ...
                    Problem.N, epsilon, phase);

                PopObj = RIFT.objectiveSelection([PopObj, OffObj, RIFT.objectiveSample(OffMain, ceil(0.20 * Problem.N)), ...
                    RIFT.objectiveSample(OffRegion, ceil(0.20 * Problem.N))], Problem.N, phase);
                PopOutput = RIFT.mainSelection([PopMain, PopRegion, PopObj, Archive, FeasArchive, ...
                    OffMain, OffRegion, OffTransfer, OffRefine, OffPolish, OffBlock], Problem.N, phase);
            end
        end

        function [Population, RankSolution] = selectionConstrained(Population, N, alpha, lowFeasMode)
            if nargin < 4
                lowFeasMode = false;
            end
            if isempty(Population)
                RankSolution = [];
                return;
            end
            if numel(Population) < N
                RankSolution = 1:numel(Population);
                return;
            end

            PopObj = Population.objs;
            PopCon = Population.cons;
            CV = sum(max(0, PopCon), 2);
            feasible = CV <= 1e-12;
            if lowFeasMode && sum(feasible) < N
                balancedCV = DRIFT.balancedCV(Population);
                [Population, RankSolution] = DRIFT.lowFeasibilitySelection(Population, N, balancedCV, alpha, feasible);
                return;
            end

            if sum(feasible) >= N
                Population = RIFT.selectByObjective(Population(feasible), N, alpha);
                RankSolution = 1:numel(Population);
                return;
            end

            if any(feasible)
                chosen = find(feasible);
                remain = find(~feasible);
                slots = N - numel(chosen);
                score = RIFT.scaleToUnit(CV(remain)) + ...
                    0.20 * RIFT.scaleToUnit(sum(RIFT.normalizeColumns(PopObj(remain, :)), 2)) - ...
                    0.04 * RIFT.scaleToUnit(RIFT.decisionCrowding(Population(remain).decs));
                [~, order] = sort(score, 'ascend');
                chosen = [chosen(:); remain(order(1:min(slots, numel(order))))];
                Population = Population(chosen(1:min(N, numel(chosen))));
                RankSolution = 1:numel(Population);
                return;
            end

            AugObj = [RIFT.normalizeColumns(PopObj), 0.85 * RIFT.scaleToUnit(CV(:))];
            Population = DRIFT.selectByAugmentedFront(Population, AugObj, N, 0.10);
            RankSolution = 1:numel(Population);
        end

        function [Population, RankSolution] = selectionObjective(Population, N, alpha, gamma, para)
            if isempty(Population)
                RankSolution = [];
                return;
            end
            if numel(Population) < N
                RankSolution = 1:numel(Population);
                return;
            end

            PopObj = Population.objs;
            PopCon = Population.cons;
            CV = sum(max(0, PopCon), 2);
            gamma = max(gamma, 1e-6);
            penaltyPower = min(4, max(0.5, exp(min(2, para)) / gamma));
            penalty = RIFT.scaleToUnit(CV(:)) .^ penaltyPower;
            AugObj = RIFT.normalizeColumns(PopObj) + (0.15 + 0.35 * alpha) * penalty;
            Population = DRIFT.selectByAugmentedFront(Population, AugObj, N, 0.06);
            RankSolution = 1:numel(Population);
        end

        function Population = selectByAugmentedFront(Population, AugObj, N, decWeight)
            if isempty(Population) || numel(Population) <= N
                return;
            end

            [FrontNo, MaxFNo] = NDSort(AugObj, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            if sum(Next) < N && ~isempty(Last)
                crowd = CrowdingDistance(AugObj(Last, :), FrontNo(Last));
                decCrowd = RIFT.decisionCrowding(Population(Last).decs);
                score = (1 - decWeight) * RIFT.scaleToUnit(crowd(:)) + ...
                    decWeight * RIFT.scaleToUnit(decCrowd(:));
                [~, order] = sort(score, 'descend');
                Next(Last(order(1:N - sum(Next)))) = true;
            end
            if sum(Next) < N
                rest = find(~Next);
                Next(rest(1:min(numel(rest), N - sum(Next)))) = true;
            end
            Population = Population(Next);
        end

        function [Population, RankSolution] = lowFeasibilitySelection(Population, N, CV, phaseProxy, feasible)
            if nargin < 5
                feasible = CV <= 1e-12;
            end
            if isempty(Population)
                RankSolution = [];
                return;
            end
            if numel(Population) <= N
                RankSolution = 1:numel(Population);
                return;
            end

            Next = false(1, numel(Population));
            if any(feasible)
                Next(find(feasible)) = true;
            end

            slots = N - sum(Next);
            if slots > 0
                idx = find(~Next);
                cvScore = RIFT.scaleToUnit(CV(idx));
                decDensity = 1 - RIFT.scaleToUnit(RIFT.decisionCrowding(Population(idx).decs));
                score = 0.94 * cvScore(:) + 0.06 * decDensity(:);
                [~, order] = sort(score, 'ascend');
                candCount = min(numel(order), max(slots, ceil(0.55 * numel(order))));
                cand = idx(order(1:candCount));

                ObjSpace = RIFT.normalizeColumns(Population(cand).objs);
                CVSpace  = RIFT.scaleToUnit(CV(cand));
                DecSpace = RIFT.normalizeColumns(Population(cand).decs);
                decWeight = 0.18 / sqrt(max(1, size(DecSpace, 2)));
                diversitySpace = [0.55 * ObjSpace, 0.85 * CVSpace(:), decWeight * DecSpace];
                pick = RIFT.greedyDiversePick(diversitySpace, score(order(1:candCount)), slots);
                Next(cand(pick)) = true;
            end

            selected = find(Next);
            rankCV = RIFT.scaleToUnit(CV(selected));
            rankDecDensity = 1 - RIFT.scaleToUnit(RIFT.decisionCrowding(Population(selected).decs));
            rankScore = 0.96 * rankCV(:) + 0.04 * rankDecDensity(:);
            [~, order] = sort(rankScore, 'ascend');
            Population = Population(selected(order(1:min(N, numel(order)))));
            RankSolution = 1:numel(Population);
        end

        function Population = feasibilityBackboneSelection(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end

            rawCV = RIFT.fullCV(Population);
            feasible = rawCV <= 1e-12;
            if sum(feasible) >= N
                Population = RIFT.selectByObjective(Population(feasible), N, phase);
                return;
            end

            Next = false(1, numel(Population));
            if any(feasible)
                Next(find(feasible)) = true;
            end

            slots = N - sum(Next);
            idx = find(~Next);
            if slots > 0 && ~isempty(idx)
                balCV = DRIFT.balancedCV(Population);
                objScore = RIFT.scaleToUnit(sum(RIFT.normalizeColumns(Population.objs), 2));
                crowd = RIFT.decisionCrowding(Population.decs);
                scoreAll = 0.54 * RIFT.scaleToUnit(balCV(:)) + ...
                           0.36 * RIFT.scaleToUnit(rawCV(:)) + ...
                           0.08 * objScore(:) - ...
                           0.02 * RIFT.scaleToUnit(crowd(:));
                [~, order] = sort(scoreAll(idx), 'ascend');
                candCount = min(numel(order), max(slots, ceil(0.60 * numel(order))));
                cand = idx(order(1:candCount));

                ConsSpace = RIFT.normalizeColumns(max(Population(cand).cons, 0));
                ObjSpace  = RIFT.normalizeColumns(Population(cand).objs);
                DecSpace  = RIFT.normalizeColumns(Population(cand).decs);
                decWeight = 0.14 / sqrt(max(1, size(DecSpace, 2)));
                pickSpace = [0.90 * ConsSpace, 0.30 * ObjSpace, decWeight * DecSpace];
                pick = RIFT.greedyDiversePick(pickSpace, scoreAll(cand), slots);
                Next(cand(pick)) = true;
            end

            selected = find(Next);
            rawCV = RIFT.fullCV(Population(selected));
            if any(rawCV <= 1e-12)
                feasible = rawCV <= 1e-12;
                feasibleIdx = selected(feasible);
                infeasibleIdx = selected(~feasible);
                if numel(feasibleIdx) >= N
                    Population = RIFT.selectByObjective(Population(feasibleIdx), N, phase);
                    return;
                end
                if ~isempty(infeasibleIdx)
                    balCV = DRIFT.balancedCV(Population(infeasibleIdx));
                    [~, ordInf] = sortrows([RIFT.scaleToUnit(balCV(:)), RIFT.scaleToUnit(RIFT.fullCV(Population(infeasibleIdx)))], [1 2]);
                    orderedInfeasible = infeasibleIdx(ordInf(:));
                    selected = [feasibleIdx(:); orderedInfeasible(:)];
                else
                    selected = feasibleIdx(:);
                end
            end
            Population = Population(selected(1:min(N, numel(selected))));
        end

        function Archive = updateFeasibleArchive(Pool, K, phase)
            if isempty(Pool) || K <= 0
                Archive = Pool(1:0);
                return;
            end
            feasible = RIFT.fullCV(Pool) <= 1e-12;
            if ~any(feasible)
                Archive = Pool(1:0);
                return;
            end
            Feas = Pool(feasible);
            if numel(Feas) <= K
                Archive = Feas;
            else
                Archive = RIFT.selectByObjective(Feas, K, phase);
            end
        end

        function Del = truncation(PopObj, K)
            distance = pdist2(PopObj, PopObj);
            distance(logical(eye(size(distance, 1)))) = inf;
            Del = false(1, size(PopObj, 1));
            while sum(Del) < K
                remain = find(~Del);
                temp = sort(distance(remain, remain), 2);
                [~, rank] = sortrows(temp);
                Del(remain(rank(1))) = true;
            end
        end

        function Offspring = generateCVRescue(Problem, Source, count, phase)
            Empty = Source(1:0);
            if isempty(Source) || count <= 0 || numel(Source) < 4
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            CV = DRIFT.balancedCV(Source);
            [~, order] = sort(CV, 'ascend');
            eliteCount = max(3, min(numel(Source), ceil(0.16 * numel(Source))));
            baseRange  = max(eliteCount, min(numel(Source), ceil(0.62 * numel(Source))));
            Dec = Source.decs;
            D = size(Dec, 2);

            baseIdx  = order(randi(baseRange, 1, count));
            bestIdx  = order(randi(eliteCount, 1, count));
            donorA   = order(randi(baseRange, 1, count));
            donorB   = randi(numel(Source), 1, count);

            span  = repmat(Problem.upper - Problem.lower, count, 1);
            lower = repmat(Problem.lower, count, 1);
            upper = repmat(Problem.upper, count, 1);
            F = 0.55 + 0.45 * rand(count, 1);
            jitter = (0.020 * (1 - phase) + 0.004) .* randn(count, D) .* span;

            OffDec = Dec(baseIdx, :) + repmat(F, 1, D) .* (Dec(bestIdx, :) - Dec(baseIdx, :)) + ...
                     0.18 * (Dec(donorA, :) - Dec(donorB, :)) + jitter;
            OffDec = min(max(OffDec, lower), upper);
            Offspring = Problem.Evaluation(OffDec);
        end

        function Offspring = generateBlockRescue(Problem, Source, count, phase)
            Empty = Source(1:0);
            if isempty(Source) || count <= 0 || numel(Source) < 4
                Offspring = Empty;
                return;
            end

            count = max(1, round(count));
            Dec = Source.decs;
            [~, D] = size(Dec);
            headD = max(1, min(D - 1, Problem.M));
            tail = (headD + 1):D;
            if isempty(tail)
                Offspring = Empty;
                return;
            end

            rawCV = RIFT.fullCV(Source);
            balCV = DRIFT.balancedCV(Source);
            [~, rawOrder] = sort(rawCV, 'ascend');
            [~, balOrder] = sort(balCV, 'ascend');
            eliteRaw = max(3, min(numel(Source), ceil(0.18 * numel(Source))));
            eliteBal = max(3, min(numel(Source), ceil(0.18 * numel(Source))));

            headIdx = balOrder(randi(eliteBal, 1, count));
            tailIdx = rawOrder(randi(eliteRaw, 1, count));
            donorA  = rawOrder(randi(max(eliteRaw, ceil(0.45 * numel(Source))), 1, count));
            donorB  = randi(numel(Source), 1, count);

            OffDec = Dec(headIdx, :);
            OffDec(:, tail) = Dec(tailIdx, tail);

            F = 0.18 + 0.35 * rand(count, 1);
            OffDec(:, tail) = OffDec(:, tail) + repmat(F, 1, numel(tail)) .* ...
                (Dec(donorA, tail) - Dec(donorB, tail));

            resetRate = 0.04 + 0.10 * max(0, 0.75 - phase);
            resetMask = rand(count, numel(tail)) < resetRate;
            if any(resetMask(:))
                randomTail = rand(count, numel(tail)) .* ...
                    repmat(Problem.upper(tail) - Problem.lower(tail), count, 1) + ...
                    repmat(Problem.lower(tail), count, 1);
                temp = OffDec(:, tail);
                temp(resetMask) = randomTail(resetMask);
                OffDec(:, tail) = temp;
            end

            span = repmat(Problem.upper - Problem.lower, count, 1);
            OffDec = OffDec + (0.006 * (1 - phase) + 0.001) .* randn(count, D) .* span;
            OffDec = min(max(OffDec, repmat(Problem.lower, count, 1)), repmat(Problem.upper, count, 1));
            Offspring = Problem.Evaluation(OffDec);
        end

        function Offspring = generateCoordinatePolish(Problem, Source, count, phase)
            Empty = Source(1:0);
            if isempty(Source) || count <= 0 || numel(Source) < 2 || Problem.FE >= Problem.maxFE
                Offspring = Empty;
                return;
            end

            count = max(1, round(count));
            rawCV = RIFT.fullCV(Source);
            [~, order] = sort(rawCV, 'ascend');
            eliteCount = max(1, min(numel(Source), min(5, ceil(0.08 * numel(Source)))));
            localBudget = max(1, ceil(count / eliteCount));
            Dec = Source.decs;
            D = size(Dec, 2);
            span = Problem.upper - Problem.lower;
            lower = Problem.lower;
            upper = Problem.upper;
            step = (0.018 * (1 - phase) + 0.0015) .* span;
            maskRate = min(0.25, max(1 / max(D, 1), 0.08 + 0.08 * (1 - phase)));
            varWeight = RIFT.decisionRegionWeights(Source, order(1:eliteCount));
            varWeight = max(varWeight(:)', 0.05);
            varProb = varWeight ./ sum(varWeight);
            varCdf = cumsum(varProb);

            Offspring = Empty;
            produced = 0;
            for e = 1:eliteCount
                x = Dec(order(e), :);
                bestCV = rawCV(order(e));
                nearScale = min(1, max(0.04, 3 * sqrt(max(bestCV, 1e-10))));
                localStep = step .* nearScale;
                for t = 1:localBudget
                    if produced >= count || Problem.FE >= Problem.maxFE
                        return;
                    end
                    improved = false;
                    if rand < 0.72
                        j = find(rand <= varCdf, 1, 'first');
                        dirs = [-1, 1];
                        dirs = dirs(randperm(2));
                        for s = dirs
                            if produced >= count || Problem.FE >= Problem.maxFE
                                return;
                            end
                            y = x;
                            y(j) = y(j) + s * (0.55 + 0.45 * rand) * localStep(j);
                            y = min(max(y, lower), upper);
                            trial = Problem.Evaluation(y);
                            trialCV = RIFT.fullCV(trial);
                            Offspring = [Offspring, trial]; %#ok<AGROW>
                            produced = produced + 1;
                            if trialCV <= bestCV
                                x = trial.decs;
                                bestCV = trialCV;
                                improved = true;
                            end
                        end
                    else
                        mask = rand(1, D) < min(0.60, maskRate * D * varProb);
                        if ~any(mask)
                            mask(find(rand <= varCdf, 1, 'first')) = true;
                        end
                        y = x;
                        y(mask) = y(mask) + randn(1, sum(mask)) .* localStep(mask);
                        y = min(max(y, lower), upper);
                        trial = Problem.Evaluation(y);
                        trialCV = RIFT.fullCV(trial);
                        Offspring = [Offspring, trial]; %#ok<AGROW>
                        produced = produced + 1;
                        if trialCV <= bestCV
                            x = trial.decs;
                            bestCV = trialCV;
                            improved = true;
                        end
                    end
                    if improved
                        localStep = min(step * 1.50, localStep * 1.08);
                    else
                        localStep = max(step * 0.002, localStep * 0.62);
                    end
                end
            end
        end

        function CV = balancedCV(Population)
            if isempty(Population) || isempty(Population.cons)
                CV = zeros(numel(Population), 1);
                return;
            end
            V = max(Population.cons, 0);
            scale = max(V, [], 1);
            scale(scale <= eps) = 1;
            CV = sum(V ./ scale, 2);
        end

        function X = safeShift(X)
            if isempty(X)
                return;
            end
            zeroRows = sqrt(sum(X.^2, 2)) <= eps;
            if any(zeroRows)
                X(zeroRows, :) = eps;
            end
        end
    end
end
