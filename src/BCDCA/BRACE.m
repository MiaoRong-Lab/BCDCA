classdef BRACE < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Constraint-aware selection and offspring routines used by BCDCA
% archiveRatio --- 0.30 --- Ratio of near-feasible and feasible archives
% activeRatio  --- 0.55 --- Offspring ratio of the active-constraint route
% bridgeRatio  --- 0.22 --- Offspring ratio of the bridge route

    methods
        function main(Algorithm, Problem)
            [archiveRatio, activeRatio, bridgeRatio] = Algorithm.ParameterSet(0.30, 0.55, 0.22);
            archiveRatio = min(0.70, max(0.10, archiveRatio));
            activeRatio  = min(0.90, max(0.20, activeRatio));
            bridgeRatio  = min(0.45, max(0.05, bridgeRatio));

            PopHard   = Problem.Initialization();
            PopActive = Problem.Initialization();
            PopBridge = Problem.Initialization();
            EmptyPop  = PopHard(1:0);

            if BRACE.isHardDecisionMode([PopHard, PopActive, PopBridge], Problem)
                BRACE.hardDecisionMode(Algorithm, Problem, PopHard, PopActive, PopBridge, archiveRatio);
                return;
            end
            if Problem.D >= 30
                BRACE.softHighDimMode(Algorithm, Problem, PopHard, PopActive, archiveRatio);
                return;
            end
            BRACE.lowDimCompatibilityMode(Algorithm, Problem, PopHard, PopActive, archiveRatio);
            return;

            FeasArchive = EmptyPop;
            NearArchive = EmptyPop;
            archiveSize = max(8, ceil(archiveRatio * Problem.N));

            initCV = BRACE.totalCV([PopHard, PopActive, PopBridge]);
            cv0 = max(initCV);
            if isempty(cv0) || cv0 <= 0
                cv0 = 1;
            end

            PopOutput = BRACE.outputSelection([PopHard, PopActive, PopBridge], Problem.N, 0);
            while Algorithm.NotTerminated(PopOutput)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                allMemory = [PopHard, PopActive, PopBridge, FeasArchive, NearArchive];
                activeCons = BRACE.activeConstraintSubset(allMemory, phase);
                epsilon = BRACE.relaxThreshold(allMemory, cv0, phase);

                [rankHard, crowdHard] = BRACE.matingRank(PopHard, 'hard', activeCons, epsilon, phase);
                OffHard = BRACE.generateOffspring(Problem, PopHard, rankHard, crowdHard, ...
                    Problem.N, 0.28, 0.15);

                activeSource = BRACE.safePool([PopActive, NearArchive, BRACE.lowCVSample(PopHard, ceil(0.25 * Problem.N))], PopActive);
                [rankActive, crowdActive] = BRACE.matingRank(activeSource, 'active', activeCons, epsilon, phase);
                activeCount = ceil(activeRatio * Problem.N * (1.05 - 0.40 * phase));
                OffActive = BRACE.generateOffspring(Problem, activeSource, rankActive, crowdActive, ...
                    activeCount, 0.38, 0.05);

                bridgeSource = BRACE.safePool([PopBridge, FeasArchive, NearArchive, ...
                    BRACE.lowCVSample(PopActive, ceil(0.25 * Problem.N)), BRACE.objectiveSample(PopHard, ceil(0.20 * Problem.N))], ...
                    PopBridge);
                [rankBridge, crowdBridge] = BRACE.matingRank(bridgeSource, 'bridge', activeCons, epsilon, phase);
                bridgeCount = ceil(bridgeRatio * Problem.N * (0.65 + 0.75 * phase));
                OffBridge = BRACE.generateOffspring(Problem, bridgeSource, rankBridge, crowdBridge, ...
                    bridgeCount, 0.22, 0.45);

                candidatePool = [PopHard, PopActive, PopBridge, OffHard, OffActive, OffBridge, FeasArchive, NearArchive];
                FeasArchive = BRACE.updateFeasibleArchive(candidatePool, archiveSize, phase);
                NearArchive = BRACE.updateNearArchive(candidatePool, archiveSize, activeCons, epsilon, phase);

                PopHard = BRACE.environmentalSelection([PopHard, OffHard, OffActive, OffBridge, ...
                    FeasArchive, BRACE.lowCVSample(NearArchive, ceil(0.30 * Problem.N))], ...
                    Problem.N, 'hard', activeCons, epsilon, phase);

                PopActive = BRACE.environmentalSelection([PopActive, OffActive, OffHard, OffBridge, ...
                    NearArchive, BRACE.lowCVSample(PopHard, ceil(0.25 * Problem.N))], ...
                    Problem.N, 'active', activeCons, epsilon, phase);

                PopBridge = BRACE.environmentalSelection([PopBridge, OffBridge, OffHard, OffActive, ...
                    FeasArchive, NearArchive, BRACE.objectiveSample(PopHard, ceil(0.25 * Problem.N))], ...
                    Problem.N, 'bridge', activeCons, epsilon, phase);

                PopOutput = BRACE.outputSelection([PopHard, PopActive, PopBridge, FeasArchive, NearArchive], ...
                    Problem.N, phase);
            end
        end
    end

    methods(Static)
        function hardDecisionMode(Algorithm, Problem, PopMain, PopAux, PopTrace, archiveRatio)
            FeasArchive = PopMain(1:0);
            archiveSize = max(8, ceil(archiveRatio * Problem.N));

            FitnessMain = TIDE.calFitness(PopMain.objs, PopMain.cons);
            FitnessAux  = TIDE.calFitness(PopAux.objs,  PopAux.cons);
            FitnessTrace = TIDE.violationFitness(PopTrace);
            ZminMain = min(PopMain.objs, [], 1);
            ZminAux  = min(PopAux.objs,  [], 1);

            initCons = [PopMain.cons; PopAux.cons; PopTrace.cons];
            initCons(initCons < 0) = 0;
            VAR0 = max(sum(initCons, 2));
            if isempty(VAR0) || VAR0 <= 0
                VAR0 = 1;
            end
            cp = (-log(VAR0) - 6) / log(1 - 0.5);
            X = 0;
            PopOutput = TIDE.outputSelection([PopMain, PopAux, PopTrace, FeasArchive], Problem.N, 0);

            while Algorithm.NotTerminated(PopOutput)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                VAR = VAR0 * max(0, 1 - X) ^ cp;
                halfN = floor(Problem.N / 2);
                restN = Problem.N - halfN;

                MatingMain = PopMain(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingMain, ZminMain);
                OffMain(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));

                refineSource = SIFT.feasibleRefinementSource([FeasArchive, PopMain, PopAux, PopTrace]);
                refineN = SIFT.refinementBudget(restN, phase, refineSource);
                pbestN = restN - refineN;
                if pbestN > 0
                    OffMain(halfN+1:halfN+pbestN) = TIDE.operatorPBestDE(Problem, PopMain, FitnessMain, pbestN, 0.10);
                end
                if refineN > 0
                    refineFitness = SIFT.objectiveRefinementFitness(refineSource, phase);
                    OffMain(halfN+pbestN+1:Problem.N) = TIDE.operatorPBestDE(Problem, refineSource, refineFitness, refineN, 0.12);
                end

                MatingAux = PopAux(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingAux, ZminAux);
                OffAux(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));
                OffAux(halfN+1:Problem.N) = TIDE.operatorPBestDE(Problem, PopAux, FitnessAux, restN, 0.10);

                tracePool = [PopTrace, PopMain, PopAux, FeasArchive];
                traceFitness = TIDE.violationFitness(tracePool);
                traceFeasRate = BRACE.feasibleRate(tracePool);
                if phase > 0.25 && traceFeasRate < 0.80
                    traceCount = ceil((0.35 + 0.35 * max(0, 0.80 - traceFeasRate)) * Problem.N);
                else
                    traceCount = ceil(0.12 * Problem.N);
                end
                polishCount = 0;
                if phase > 0.45 && traceFeasRate < 0.35
                    bestTraceCV = min(TIDE.totalCV(tracePool));
                    if bestTraceCV < 0.10
                        polishCount = ceil(0.30 * Problem.N);
                    else
                        polishCount = ceil(0.16 * Problem.N);
                    end
                end
                OffTrace = TIDE.operatorCVPBestDE(Problem, tracePool, traceFitness, traceCount, 0.12, phase);
                OffPolish = TIDE.operatorCVLocalSearch(Problem, tracePool, traceFitness, polishCount, phase);

                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive, PopMain, PopAux, PopTrace, ...
                    OffMain, OffAux, OffTrace, OffPolish], archiveSize, phase);

                ZminMain = min([ZminMain; OffMain.objs], [], 1);
                ZminAux  = min([ZminAux;  OffAux.objs],  [], 1);

                PopTrace = TIDE.manifoldSelection([PopTrace, OffTrace, OffPolish, PopMain, PopAux, ...
                    OffMain, OffAux, FeasArchive], Problem.N, phase);
                FitnessTrace = TIDE.violationFitness(PopTrace); %#ok<NASGU>

                [PopMain, FitnessMain] = TIDE.mainTaskSelection([PopMain, OffMain, OffAux, ...
                    OffTrace, OffPolish, PopTrace, FeasArchive], Problem.N, true);
                [PopAux, FitnessAux] = TIDE.auxTaskSelection([PopAux, OffAux, OffMain, ...
                    OffTrace, OffPolish, PopTrace, FeasArchive], Problem.N, VAR);
                PopOutput = TIDE.outputSelection([PopMain, PopAux, PopTrace, FeasArchive], Problem.N, phase);
                X = X + 1 / (Problem.maxFE / Problem.N);
            end
        end

        function softHighDimMode(Algorithm, Problem, PopMain, PopAux, archiveRatio)
            PopFront = PopMain;
            PopScout = PopAux;
            PopFree  = PopAux;
            FeasArchive = PopMain(1:0);
            archiveSize = max(8, ceil(archiveRatio * Problem.N));

            FitnessMain = TIDE.calFitness(PopMain.objs, PopMain.cons);
            FitnessAux  = TIDE.calFitness(PopAux.objs,  PopAux.cons);
            ZminMain = min(PopMain.objs, [], 1);
            ZminAux  = min(PopAux.objs,  [], 1);

            initCons = [PopMain.cons; PopAux.cons];
            initCons(initCons < 0) = 0;
            VAR0 = max(sum(initCons, 2));
            if isempty(VAR0) || VAR0 <= 0
                VAR0 = 1;
            end
            cp = (-log(VAR0) - 6) / log(1 - 0.5);
            X = 0;
            alpha = BRACE.lowDimAlpha(Problem);
            [PopFront, RankFront] = BRACE.lowDimDPEAConSelection(PopFront, Problem.N, alpha);
            [PopScout, RankScout] = BRACE.lowDimDPEAObjSelection(PopScout, Problem.N, alpha, 1, 0);
            PopOutput = TIDE.outputSelection([PopMain, PopAux, PopFront, PopFree, FeasArchive], Problem.N, 0);

            while Algorithm.NotTerminated(PopOutput)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                VAR = VAR0 * max(0, 1 - X) ^ cp;
                gen = max(1, ceil(Problem.FE / Problem.N));
                halfN = floor(Problem.N / 2);
                restN = Problem.N - halfN;
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE / Problem.N) / 2 - ceil(Problem.FE / Problem.N);
                gamma = BRACE.lowDimPopulationGap(PopFront, PopScout);

                MatingMain = PopMain(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingMain, ZminMain);
                OffMain(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));
                OffMain(halfN+1:Problem.N) = TIDE.operatorPBestDE(Problem, PopMain, FitnessMain, restN, 0.10);

                MatingAux = PopAux(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingAux, ZminAux);
                OffAux(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));
                OffAux(halfN+1:Problem.N) = TIDE.operatorPBestDE(Problem, PopAux, FitnessAux, restN, 0.10);

                refineSource = SIFT.feasibleRefinementSource([FeasArchive, PopMain, PopAux]);
                if phase > 0.38 && ~isempty(refineSource) && numel(refineSource) >= 6
                    refineCount = ceil(0.22 * Problem.N);
                    refineFitness = SIFT.objectiveRefinementFitness(refineSource, phase);
                    OffRefine = TIDE.operatorPBestDE(Problem, refineSource, refineFitness, refineCount, 0.12);
                else
                    OffRefine = PopMain(1:0);
                end

                if Problem.M == 2 && phase < 0.92
                    freePool = BRACE.safePool([PopFree, PopAux, FeasArchive], PopFree);
                    [rankFree, crowdFree] = BRACE.objectiveMatingRank(freePool);
                    freeCount = ceil((0.30 - 0.16 * phase) * Problem.N);
                    OffFree = BRACE.generateOffspring(Problem, freePool, rankFree, crowdFree, ...
                        freeCount, 0.18, 0.65);
                else
                    OffFree = PopMain(1:0);
                end

                useFrontRoute = Problem.M >= 3 && (mod(gen, 3) == 0 || (phase > 0.78 && mod(gen, 2) == 0));
                if useFrontRoute
                    frontArchive = BRACE.objectiveSample(FeasArchive, ceil(0.25 * Problem.N));
                    [PopFront, RankFront] = BRACE.lowDimDPEAConSelection([PopFront, frontArchive], Problem.N, alpha);
                    [PopScout, RankScout] = BRACE.lowDimDPEAObjSelection([PopScout, frontArchive, ...
                        BRACE.lowCVSample(PopFront, ceil(0.12 * Problem.N))], Problem.N, alpha, gamma, para);
                    if Problem.M >= 3
                        frontRatio = 0.30 - 0.10 * phase;
                    else
                        frontRatio = 0.22 - 0.08 * phase;
                    end
                    frontCount = ceil(frontRatio * Problem.N);
                    OffFront = BRACE.rankedGAOffspring(Problem, [PopFront, PopScout], [RankFront, RankScout], frontCount);
                else
                    OffFront = PopMain(1:0);
                end

                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive, PopMain, PopAux, PopFront, PopScout, PopFree, ...
                    OffMain, OffAux, OffRefine, OffFront, OffFree], ...
                    archiveSize, phase);
                ZminMain = min([ZminMain; OffMain.objs], [], 1);
                ZminAux  = min([ZminAux;  OffAux.objs],  [], 1);

                [PopMain, FitnessMain] = TIDE.mainTaskSelection([PopMain, OffMain, OffAux, OffRefine, OffFree, FeasArchive], ...
                    Problem.N, true);
                [PopAux,  FitnessAux] = TIDE.auxTaskSelection([PopAux, OffAux, OffMain, OffRefine, OffFree, FeasArchive], ...
                    Problem.N, VAR);
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE / Problem.N) / 2 - ceil(Problem.FE / Problem.N);
                if useFrontRoute
                    frontMain = BRACE.objectiveSample(OffMain, ceil(0.16 * Problem.N));
                    frontAux  = BRACE.objectiveSample(OffAux,  ceil(0.16 * Problem.N));
                    frontArchive = BRACE.objectiveSample(FeasArchive, ceil(0.25 * Problem.N));
                    [PopFront, RankFront] = BRACE.lowDimDPEAConSelection([PopFront, OffFront, frontMain, frontAux, frontArchive], ...
                        Problem.N, alpha);
                    [PopScout, RankScout] = BRACE.lowDimDPEAObjSelection([PopScout, OffFront, frontMain, frontAux, frontArchive], ...
                        Problem.N, alpha, gamma, para);
                end
                if Problem.M == 2
                    PopFree = BRACE.selectByObjective([PopFree, OffFree, BRACE.objectiveSample(OffMain, ceil(0.12 * Problem.N)), ...
                        BRACE.objectiveSample(OffAux, ceil(0.12 * Problem.N)), FeasArchive], Problem.N, phase);
                end
                PopOutput = TIDE.outputSelection([PopMain, PopAux, PopFront, PopScout, PopFree, FeasArchive], Problem.N, phase);
                X = X + 1 / (Problem.maxFE / Problem.N);
            end
        end

        function lowDimCompatibilityMode(Algorithm, Problem, PopCon, PopObj, archiveRatio)
            PopObj = PopCon;
            FeasArchive = PopCon(1:0);
            archiveSize = max(8, ceil(archiveRatio * Problem.N));
            alpha = BRACE.lowDimAlpha(Problem);
            [PopCon, RankCon] = BRACE.lowDimDPEAConSelection(PopCon, Problem.N, alpha);
            [PopObj, RankObj] = BRACE.lowDimDPEAObjSelection(PopObj, Problem.N, alpha, 1, 0);
            PopOutput = PopCon;

            while Algorithm.NotTerminated(PopOutput)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                PopCon = PopCon(randperm(numel(PopCon)));
                PopObj = PopObj(randperm(numel(PopObj)));
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE / Problem.N) / 2 - ceil(Problem.FE / Problem.N);
                gamma = BRACE.lowDimPopulationGap(PopCon, PopObj);

                [PopCon, RankCon] = BRACE.lowDimDPEAConSelection(PopCon, Problem.N, alpha);
                [PopObj, RankObj] = BRACE.lowDimDPEAObjSelection(PopObj, Problem.N, alpha, gamma, para);

                matingPool = [PopCon, PopObj];
                rankPool = [RankCon, RankObj];
                MatingPool = TournamentSelection(2, 2 * Problem.N, rankPool);
                Offspring = OperatorGAhalf(Problem, matingPool(MatingPool));

                FeasArchive = BRACE.updateFeasibleArchive([FeasArchive, PopCon, PopObj, Offspring], archiveSize, phase);
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE / Problem.N) / 2 - ceil(Problem.FE / Problem.N);
                [PopCon, RankCon] = BRACE.lowDimDPEAConSelection([PopCon, Offspring], ...
                    Problem.N, alpha);
                [PopObj, RankObj] = BRACE.lowDimDPEAObjSelection([PopObj, Offspring], ...
                    Problem.N, alpha, gamma, para);
                PopOutput = PopCon;
            end
        end

        function alpha = lowDimAlpha(Problem)
            alpha = 2 ./ (1 + exp(-Problem.FE * 10 / max(Problem.maxFE, 1))) - 1;
            alpha = min(1, max(0, alpha));
        end

        function Offspring = rankedGAOffspring(Problem, Population, rank, count)
            Empty = Population(1:0);
            if isempty(Population) || count <= 0
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            rank = rank(:)';
            if numel(rank) ~= numel(Population)
                rank = 1:numel(Population);
            end
            MatingPool = TournamentSelection(2, 2 * count, rank);
            Offspring = OperatorGAhalf(Problem, Population(MatingPool));
            if numel(Offspring) > count
                Offspring = Offspring(1:count);
            end
        end

        function [rank, crowd] = objectiveMatingRank(Population)
            if isempty(Population)
                rank = [];
                crowd = [];
                return;
            end
            Obj = Population.objs;
            try
                front = NDSort(Obj, inf);
                objCrowd = CrowdingDistance(Obj, front);
            catch
                front = ones(numel(Population), 1);
                objCrowd = ones(numel(Population), 1);
            end
            rank = front(:)';
            crowd = BRACE.scaleToUnit(objCrowd(:)) + ...
                0.10 * BRACE.scaleToUnit(BRACE.decisionCrowding(Population.decs));
        end

        function gamma = lowDimPopulationGap(PopCon, PopObj)
            if isempty(PopCon) || isempty(PopObj)
                gamma = 1;
                return;
            end
            try
                shared = ismember(PopObj.objs, PopCon.objs, 'rows');
                gamma = 1 - sum(shared) / max(1, min(numel(PopCon), numel(PopObj)));
            catch
                gamma = 1;
            end
            gamma = min(1, max(1e-6, gamma));
        end

        function [Population, RankSolution] = lowDimDPEAConSelection(Population, N, alpha)
            if isempty(Population)
                RankSolution = [];
                return;
            end
            if numel(Population) <= N
                RankSolution = 1:numel(Population);
                return;
            end

            popSize = numel(Population);
            PopObj = Population.objs;
            PopCon = BRACE.consMatrix(Population);
            CV = sum(max(0, PopCon), 2);
            z = min(PopObj, [], 1);
            n = max(PopObj, [], 1);
            M = size(PopObj, 2);
            [W, ~] = UniformPoint(N, M);
            Region = BRACE.lowDimAssignRegion(PopObj, z, W);

            PopObj2 = PopObj;
            infeasible = CV > 1e-12;
            if any(infeasible)
                FDir = n - z;
                if norm(FDir) <= 1e-12
                    regionFmax = 1;
                else
                    [~, regionFmax] = min(pdist2(FDir, W, 'cosine'), [], 2);
                end
                normW = max(norm(W(regionFmax, :)), 1e-12);
                PopObj2(infeasible, :) = repmat(n, sum(infeasible), 1) + ...
                    CV(infeasible) .* W(regionFmax, :) ./ normW;
            end

            FrontNo = BRACE.lowDimConstrainedFront(PopObj, CV);
            crowd = BRACE.lowDimKthCrowding(PopObj);
            rankConvg = BRACE.lowDimConvergenceRank(PopObj, FrontNo, crowd, N);

            frontD = BRACE.lowDimRegionRanks(PopObj2, z, W, Region);
            frontD = frontD + infeasible * popSize;
            rankDivs = BRACE.rankByFrontCrowd(frontD, crowd);

            rankAll = alpha * rankConvg + (1 - alpha) * rankDivs;
            [~, order] = sort(rankAll, 'ascend');
            Population = Population(order(1:N));
            RankSolution = 1:N;
        end

        function [Population, RankSolution] = lowDimDPEAObjSelection(Population, N, alpha, gamma, para)
            if isempty(Population)
                RankSolution = [];
                return;
            end
            if numel(Population) <= N
                RankSolution = 1:numel(Population);
                return;
            end

            popSize = numel(Population);
            PopObj = Population.objs;
            PopCon = BRACE.consMatrix(Population);
            CV = sum(max(0, PopCon), 2);
            z = min(PopObj, [], 1);
            M = size(PopObj, 2);
            [W, ~] = UniformPoint(N, M);
            Region = BRACE.lowDimAssignRegion(PopObj, z, W);
            PopObj2 = PopObj;
            phiMax = max(CV);
            if isempty(phiMax) || phiMax <= 1e-12
                phiMax = 1;
            end
            pressure = exp(min(max(para, -30), 30)) / max(gamma, 1e-6);
            pressure = min(max(pressure, 1e-6), 1e6);

            for i = 1:size(W, 1)
                index = find(Region == i);
                if isempty(index)
                    continue;
                end
                localCV = CV(index);
                infeasible = localCV > 1e-12;
                if any(infeasible)
                    objsTemp = PopObj2(index, :);
                    FMax = max(objsTemp, [], 1);
                    scale = (localCV(infeasible) ./ phiMax) .^ pressure;
                    scale = min(1, max(0, scale));
                    PopObj2(index(infeasible), :) = objsTemp(infeasible, :) + ...
                        scale .* (FMax - objsTemp(infeasible, :));
                end
            end

            FrontNo = BRACE.lowDimParetoFront(PopObj2);
            crowd = BRACE.lowDimKthCrowding(PopObj);
            rankConvg = BRACE.lowDimConvergenceRank(PopObj, FrontNo, crowd, N);

            frontD = BRACE.lowDimRegionRanks(PopObj2, z, W, Region);
            rankDivs = BRACE.rankByFrontCrowd(frontD, crowd);

            rankAll = alpha * rankConvg + (1 - alpha) * rankDivs;
            [~, order] = sort(rankAll, 'ascend');
            Population = Population(order(1:N));
            RankSolution = 1:N;
        end

        function Region = lowDimAssignRegion(PopObj, z, W)
            Shifted = PopObj - z;
            rowNorm = sqrt(sum(Shifted.^2, 2));
            wNorm = sqrt(sum(W.^2, 2))';
            wNorm(wNorm <= 1e-12) = 1;
            Dist = inf(size(PopObj, 1), size(W, 1));
            valid = rowNorm > 1e-12;
            if any(valid)
                Dist(valid, :) = 1 - (Shifted(valid, :) * W') ./ (rowNorm(valid) * wNorm);
            end
            Dist(~isfinite(Dist)) = inf;
            bad = all(isinf(Dist), 2);
            if any(bad)
                Dist(bad, :) = pdist2(BRACE.normalizeColumns(PopObj(bad, :)), W, 'euclidean');
            end
            [~, Region] = min(Dist, [], 2);
        end

        function FrontNo = lowDimConstrainedFront(PopObj, CV)
            popSize = size(PopObj, 1);
            Dominate = false(popSize);
            for i = 1:popSize-1
                for j = i+1:popSize
                    if CV(i) < CV(j)
                        Dominate(i, j) = true;
                    elseif CV(i) > CV(j)
                        Dominate(j, i) = true;
                    else
                        k = any(PopObj(i, :) < PopObj(j, :)) - any(PopObj(i, :) > PopObj(j, :));
                        if k == 1
                            Dominate(i, j) = true;
                        elseif k == -1
                            Dominate(j, i) = true;
                        end
                    end
                end
            end
            S = sum(Dominate, 2);
            R = zeros(1, popSize);
            for i = 1:popSize
                R(i) = sum(S(Dominate(:, i)));
            end
            FrontNo = R(:) + 1;
        end

        function FrontNo = lowDimParetoFront(PopObj)
            popSize = size(PopObj, 1);
            Dominate = false(popSize);
            for i = 1:popSize-1
                for j = i+1:popSize
                    k = any(PopObj(i, :) < PopObj(j, :)) - any(PopObj(i, :) > PopObj(j, :));
                    if k == 1
                        Dominate(i, j) = true;
                    elseif k == -1
                        Dominate(j, i) = true;
                    end
                end
            end
            S = sum(Dominate, 2);
            R = zeros(1, popSize);
            for i = 1:popSize
                R(i) = sum(S(Dominate(:, i)));
            end
            FrontNo = R(:) + 1;
        end

        function frontD = lowDimRegionRanks(PopObj, z, W, Region)
            popSize = size(PopObj, 1);
            frontD = ones(popSize, 1);
            for i = 1:size(W, 1)
                index = find(Region == i);
                if isempty(index)
                    continue;
                end
                g = sum((PopObj(index, :) - z) .* W(i, :), 2);
                [~, order] = sort(g, 'ascend');
                frontD(index(order)) = (1:numel(index))';
            end
        end

        function rank = rankByFrontCrowd(frontNo, crowd)
            [~, order] = sortrows([frontNo(:), -crowd(:)]);
            rank = zeros(1, numel(frontNo));
            rank(order) = 1:numel(frontNo);
        end

        function rank = lowDimConvergenceRank(PopObj, frontNo, crowd, N)
            middle = zeros(numel(frontNo), 1);
            firstFront = frontNo(:) == 1;
            if sum(firstFront) > N
                del = BRACE.lowDimTruncation(PopObj(firstFront, :), sum(firstFront) - N);
                temp = find(firstFront);
                middle(temp(del)) = 1;
                [~, order] = sortrows([frontNo(:), middle(:), -crowd(:)]);
            else
                [~, order] = sortrows([frontNo(:), -crowd(:)]);
            end
            rank = zeros(1, numel(frontNo));
            rank(order) = 1:numel(frontNo);
        end

        function del = lowDimTruncation(PopObj, K)
            del = false(1, size(PopObj, 1));
            if K <= 0 || isempty(PopObj)
                return;
            end
            Dist = pdist2(PopObj, PopObj);
            Dist(logical(eye(size(Dist, 1)))) = inf;
            while sum(del) < K
                remain = find(~del);
                temp = sort(Dist(remain, remain), 2);
                [~, rank] = sortrows(temp);
                del(remain(rank(1))) = true;
            end
        end

        function crowd = lowDimKthCrowding(PopObj)
            popSize = size(PopObj, 1);
            if popSize <= 2
                crowd = inf(popSize, 1);
                return;
            end
            Dist = pdist2(PopObj, PopObj);
            Dist(logical(eye(popSize))) = inf;
            Dist = sort(Dist, 2);
            kth = max(1, min(size(Dist, 2), floor(sqrt(popSize))));
            crowd = Dist(:, kth);
            if any(~isfinite(crowd))
                finiteCrowd = crowd(isfinite(crowd));
                if isempty(finiteCrowd)
                    crowd(:) = 1;
                else
                    crowd(~isfinite(crowd)) = max(finiteCrowd);
                end
            end
        end

        function [rank, crowd] = lowDimMatingRank(Population, phase)
            CV = BRACE.totalCV(Population);
            Obj = BRACE.normalizeColumns(Population.objs);
            penalty = BRACE.scaleToUnit(CV(:));
            AugObj = Obj + (0.08 + 0.30 * phase) * penalty;
            try
                front = NDSort(AugObj, inf);
                objCrowd = CrowdingDistance(AugObj, front);
            catch
                front = ones(numel(Population), 1);
                objCrowd = ones(numel(Population), 1);
            end
            rank = front(:) + 0.20 * penalty(:);
            crowd = 0.82 * BRACE.scaleToUnit(objCrowd(:)) + ...
                    0.18 * BRACE.scaleToUnit(BRACE.decisionCrowding(Population.decs));
        end

        function Population = lowDimConSelection(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            CV = BRACE.totalCV(Population);
            feasible = CV <= 1e-12;
            if sum(feasible) >= N
                Population = BRACE.selectByObjective(Population(feasible), N, phase);
                return;
            end

            Next = false(1, numel(Population));
            if any(feasible)
                Next(find(feasible)) = true; %#ok<FNDSB>
            end
            slots = N - sum(Next);
            rest = find(~Next);
            if slots > 0 && ~isempty(rest)
                Obj = BRACE.normalizeColumns(Population(rest).objs);
                score = 0.70 * BRACE.scaleToUnit(CV(rest)) + ...
                        0.22 * BRACE.scaleToUnit(sum(Obj, 2)) - ...
                        0.08 * BRACE.scaleToUnit(BRACE.decisionCrowding(Population(rest).decs));
                [~, order] = sort(score, 'ascend');
                Next(rest(order(1:min(slots, numel(order))))) = true;
            end
            Population = Population(Next);
            if numel(Population) > N
                Population = Population(1:N);
            end
        end

        function Population = lowDimObjSelection(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            CV = BRACE.totalCV(Population);
            Obj = BRACE.normalizeColumns(Population.objs);
            penalty = BRACE.scaleToUnit(CV(:)).^(1 + 1.5 * phase);
            AugObj = Obj + (0.04 + 0.34 * phase) * penalty;
            Population = BRACE.selectByAugmentedObjective(Population, AugObj, N, 0.08);
        end

        function Population = selectByAugmentedObjective(Population, AugObj, N, decWeight)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            [FrontNo, MaxFNo] = NDSort(AugObj, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            if sum(Next) < N && ~isempty(Last)
                crowd = CrowdingDistance(AugObj(Last, :), FrontNo(Last));
                decCrowd = BRACE.decisionCrowding(Population(Last).decs);
                score = (1 - decWeight) * BRACE.scaleToUnit(crowd(:)) + ...
                        decWeight * BRACE.scaleToUnit(decCrowd(:));
                [~, order] = sort(score, 'descend');
                Next(Last(order(1:N - sum(Next)))) = true;
            end
            if sum(Next) < N
                rest = find(~Next);
                Next(rest(1:min(numel(rest), N - sum(Next)))) = true;
            end
            Population = Population(Next);
        end

        function tf = isHardDecisionMode(Population, Problem)
            CV = BRACE.totalCV(Population);
            consCount = size(BRACE.consMatrix(Population), 2);
            initFeasRate = mean(CV <= 1e-12);
            initMedianCV = median(CV);
            tf = Problem.D >= 30 && ...
                (consCount > Problem.M || ...
                (consCount >= Problem.M && initFeasRate < 0.20 && initMedianCV > 50));
        end

        function Offspring = generateOffspring(Problem, Source, rank, crowd, count, pBestRate, objBias)
            Empty = Source(1:0);
            if isempty(Source) || count <= 0
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            if numel(Source) < 4
                parents = randi(numel(Source), 1, 2 * count);
                Offspring = OperatorGAhalf(Problem, Source(parents));
                if numel(Offspring) > count
                    Offspring = Offspring(1:count);
                end
                return;
            end

            gaCount = max(1, ceil((0.55 - 0.25 * objBias) * count));
            deCount = max(0, count - gaCount);

            parents = TournamentSelection(2, 2 * gaCount, rank(:), -crowd(:));
            OffGA = OperatorGAhalf(Problem, Source(parents));

            if deCount > 0
                [~, order] = sort(rank(:), 'ascend');
                eliteCount = max(1, min(numel(Source), ceil(pBestRate * numel(Source))));
                base = TournamentSelection(2, deCount, rank(:), -crowd(:));
                guide = order(randi(eliteCount, 1, deCount));
                rand1 = randi(numel(Source), 1, deCount);
                OffDE = OperatorDE(Problem, Source(base), Source(guide), Source(rand1), ...
                    {1, 0.5 + 0.3 * objBias, 1, 20});
            else
                OffDE = Empty;
            end

            Offspring = [OffGA, OffDE];
            if numel(Offspring) > count
                Offspring = Offspring(1:count);
            end
        end

        function Population = environmentalSelection(Pool, N, mode, activeCons, epsilon, phase)
            Pool = BRACE.safePool(Pool, Pool);
            if isempty(Pool) || numel(Pool) <= N
                Population = Pool;
                return;
            end

            fullCV = BRACE.totalCV(Pool);
            activeCV = BRACE.taskCV(Pool, activeCons);
            switch mode
                case 'hard'
                    effCV = fullCV;
                    relaxed = effCV <= 1e-12;
                case 'active'
                    effCV = activeCV;
                    relaxed = effCV <= max(epsilon, 1e-12);
                otherwise
                    effCV = 0.65 * fullCV + 0.35 * activeCV;
                    relaxed = fullCV <= max(0.50 * epsilon, 1e-12);
            end

            if sum(relaxed) >= N
                Population = BRACE.selectByObjective(Pool(relaxed), N, phase);
                return;
            end

            Next = false(1, numel(Pool));
            if any(relaxed)
                Next(find(relaxed)) = true; %#ok<FNDSB>
            end
            remainSlots = N - sum(Next);
            if remainSlots > 0
                rest = find(~Next);
                score = BRACE.selectionScore(Pool(rest), mode, activeCons, epsilon, phase);
                [~, order] = sort(score, 'ascend');
                choose = rest(order(1:min(remainSlots, numel(order))));
                Next(choose) = true;
            end
            if sum(Next) > N
                chosen = find(Next);
                score = BRACE.selectionScore(Pool(chosen), mode, activeCons, epsilon, phase);
                [~, order] = sort(score, 'ascend');
                chosen = chosen(order(1:N));
                Next(:) = false;
                Next(chosen) = true;
            end
            Population = Pool(Next);
        end

        function score = selectionScore(Population, mode, activeCons, epsilon, phase)
            fullCV = BRACE.totalCV(Population);
            activeCV = BRACE.taskCV(Population, activeCons);
            Obj = BRACE.normalizeColumns(Population.objs);
            objScore = sum(Obj, 2);
            decCrowd = BRACE.decisionCrowding(Population.decs);
            activePenalty = max(0, activeCV - max(epsilon, 0));

            switch mode
                case 'hard'
                    score = 0.72 * BRACE.scaleToUnit(fullCV) + ...
                            0.23 * BRACE.scaleToUnit(objScore) - ...
                            0.05 * BRACE.scaleToUnit(decCrowd);
                case 'active'
                    score = 0.48 * BRACE.scaleToUnit(activePenalty) + ...
                            0.27 * BRACE.scaleToUnit(fullCV) + ...
                            0.20 * BRACE.scaleToUnit(objScore) - ...
                            0.05 * BRACE.scaleToUnit(decCrowd);
                otherwise
                    score = 0.38 * BRACE.scaleToUnit(fullCV) + ...
                            0.20 * BRACE.scaleToUnit(activePenalty) + ...
                            0.34 * BRACE.scaleToUnit(objScore) - ...
                            0.08 * BRACE.scaleToUnit(decCrowd);
            end
        end

        function [rank, crowd] = matingRank(Population, mode, activeCons, epsilon, phase)
            if isempty(Population)
                rank = [];
                crowd = [];
                return;
            end
            score = BRACE.selectionScore(Population, mode, activeCons, epsilon, phase);
            Obj = BRACE.normalizeColumns(Population.objs);
            try
                front = NDSort(Obj, inf);
                objCrowd = CrowdingDistance(Obj, front);
            catch
                objCrowd = ones(numel(Population), 1);
            end
            rank = score(:);
            crowd = 0.65 * BRACE.scaleToUnit(objCrowd(:)) + ...
                    0.35 * BRACE.scaleToUnit(BRACE.decisionCrowding(Population.decs));
        end

        function Archive = updateFeasibleArchive(Pool, K, phase)
            if isempty(Pool)
                Archive = Pool;
                return;
            end
            feasible = BRACE.totalCV(Pool) <= 1e-12;
            if ~any(feasible)
                Archive = Pool(1:0);
                return;
            end
            Archive = BRACE.selectByObjective(Pool(feasible), min(K, sum(feasible)), phase);
        end

        function Archive = updateNearArchive(Pool, K, activeCons, epsilon, phase)
            if isempty(Pool)
                Archive = Pool;
                return;
            end
            fullCV = BRACE.totalCV(Pool);
            activeCV = BRACE.taskCV(Pool, activeCons);
            sortedCV = sort(fullCV, 'ascend');
            cut = sortedCV(max(1, min(numel(sortedCV), ceil((0.35 + 0.25 * (1 - phase)) * numel(sortedCV)))));
            near = fullCV <= max(cut, 0.50 * max(epsilon, 0));
            if sum(near) < min(K, numel(Pool))
                [~, order] = sort(fullCV, 'ascend');
                near(order(1:min(numel(order), max(K, sum(near))))) = true;
            end
            Cand = Pool(near);
            Obj = BRACE.normalizeColumns(Cand.objs);
            score = 0.55 * BRACE.scaleToUnit(fullCV(near)) + ...
                    0.25 * BRACE.scaleToUnit(activeCV(near)) + ...
                    0.20 * BRACE.scaleToUnit(sum(Obj, 2));
            pick = BRACE.pickDiverse(Obj, score, min(K, numel(Cand)));
            Archive = Cand(pick);
        end

        function Population = outputSelection(Pool, N, phase)
            if isempty(Pool)
                Population = Pool;
                return;
            end
            feasible = BRACE.totalCV(Pool) <= 1e-12;
            if any(feasible)
                Population = BRACE.selectByObjective(Pool(feasible), min(N, sum(feasible)), phase);
                if numel(Population) < N
                    fill = BRACE.lowCVSample(Pool(~feasible), N - numel(Population));
                    Population = [Population, fill];
                end
            else
                Population = BRACE.environmentalSelection(Pool, N, 'hard', [], 0, phase);
            end
        end

        function Population = selectByObjective(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            Obj = Population.objs;
            [FrontNo, MaxFNo] = NDSort(Obj, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            if sum(Next) < N && ~isempty(Last)
                crowd = CrowdingDistance(Obj(Last, :), FrontNo(Last));
                decCrowd = BRACE.decisionCrowding(Population(Last).decs);
                mixCrowd = (0.80 + 0.12 * phase) * BRACE.scaleToUnit(crowd(:)) + ...
                           (0.20 - 0.12 * phase) * BRACE.scaleToUnit(decCrowd(:));
                [~, order] = sort(mixCrowd, 'descend');
                Next(Last(order(1:N - sum(Next)))) = true;
            end
            Population = Population(Next);
        end

        function Sample = lowCVSample(Population, K)
            if isempty(Population) || K <= 0
                Sample = Population(1:0);
                return;
            end
            [~, order] = sort(BRACE.totalCV(Population), 'ascend');
            Sample = Population(order(1:min(K, numel(order))));
        end

        function Sample = objectiveSample(Population, K)
            if isempty(Population) || K <= 0
                Sample = Population(1:0);
                return;
            end
            score = sum(BRACE.normalizeColumns(Population.objs), 2) + ...
                    0.15 * BRACE.scaleToUnit(BRACE.totalCV(Population));
            [~, order] = sort(score, 'ascend');
            Sample = Population(order(1:min(K, numel(order))));
        end

        function activeCons = activeConstraintSubset(Population, phase)
            C = BRACE.consMatrix(Population);
            if isempty(C)
                activeCons = [];
                return;
            end
            V = max(C, 0);
            nCons = size(V, 2);
            if nCons == 0
                activeCons = [];
                return;
            end
            severity = mean(V, 1) + 0.50 * std(V, 0, 1);
            activity = mean(V > 1e-12, 1);
            score = BRACE.scaleToUnit(severity(:))' + 0.80 * BRACE.scaleToUnit(activity(:))';
            if all(score <= 1e-12)
                activeCons = 1:nCons;
                return;
            end
            keepRatio = 0.35 + 0.55 * min(1, max(0, phase));
            keep = max(1, min(nCons, ceil(keepRatio * nCons)));
            [~, order] = sort(score, 'descend');
            activeCons = sort(order(1:keep));
        end

        function epsilon = relaxThreshold(Population, cv0, phase)
            base = cv0 * max(0, 1 - phase) ^ 2.8;
            CV = BRACE.totalCV(Population);
            if isempty(CV)
                epsilon = base;
                return;
            end
            CV = sort(CV, 'ascend');
            q = CV(max(1, min(numel(CV), ceil(0.25 * numel(CV)))));
            epsilon = max(base, q * max(0, 1 - phase));
            if phase > 0.88
                epsilon = 0;
            end
        end

        function CV = taskCV(Population, activeCons)
            C = BRACE.consMatrix(Population);
            if isempty(C)
                CV = zeros(numel(Population), 1);
                return;
            end
            if isempty(activeCons)
                CV = sum(max(C, 0), 2);
                return;
            end
            activeCons = activeCons(activeCons >= 1 & activeCons <= size(C, 2));
            if isempty(activeCons)
                CV = sum(max(C, 0), 2);
            else
                CV = sum(max(C(:, activeCons), 0), 2);
            end
        end

        function CV = totalCV(Population)
            C = BRACE.consMatrix(Population);
            if isempty(C)
                CV = zeros(numel(Population), 1);
            else
                CV = sum(max(C, 0), 2);
            end
        end

        function rate = feasibleRate(Population)
            if isempty(Population)
                rate = 0;
            else
                rate = mean(BRACE.totalCV(Population) <= 1e-12);
            end
        end

        function C = consMatrix(Population)
            if isempty(Population)
                C = [];
            else
                C = Population.cons;
                if isempty(C)
                    C = zeros(numel(Population), 0);
                end
            end
        end

        function Pool = safePool(Pool, Fallback)
            if isempty(Pool)
                Pool = Fallback;
            end
        end

        function pick = pickDiverse(Obj, score, K)
            if isempty(Obj) || K <= 0
                pick = [];
                return;
            end
            K = min(K, size(Obj, 1));
            score = score(:);
            [~, first] = min(score);
            pick = first;
            remain = setdiff(1:size(Obj, 1), pick);
            while numel(pick) < K && ~isempty(remain)
                dist = pdist2(Obj(remain, :), Obj(pick, :), 'euclidean');
                minDist = min(dist, [], 2);
                merit = BRACE.scaleToUnit(minDist) - 0.35 * BRACE.scaleToUnit(score(remain));
                [~, best] = max(merit);
                pick(end+1) = remain(best); %#ok<AGROW>
                remain(best) = [];
            end
        end

        function DecCrowd = decisionCrowding(Dec)
            if isempty(Dec)
                DecCrowd = [];
                return;
            end
            N = size(Dec, 1);
            if N <= 2
                DecCrowd = ones(N, 1);
                return;
            end
            D = pdist2(Dec, Dec, 'euclidean');
            D(1:N+1:end) = inf;
            DecCrowd = min(D, [], 2);
        end

        function Obj = normalizeColumns(Obj)
            if isempty(Obj)
                return;
            end
            lo = min(Obj, [], 1);
            hi = max(Obj, [], 1);
            span = hi - lo;
            span(span < 1e-12) = 1;
            Obj = (Obj - lo) ./ span;
        end

        function y = scaleToUnit(x)
            x = x(:);
            if isempty(x)
                y = x;
                return;
            end
            finite = isfinite(x);
            if ~any(finite)
                y = zeros(size(x));
                return;
            end
            x(~finite) = max(x(finite));
            lo = min(x);
            hi = max(x);
            if hi - lo < 1e-12
                y = zeros(size(x));
            else
                y = (x - lo) ./ (hi - lo);
            end
        end
    end
end
