classdef RIFT < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Region-informed infeasible-feasible transfer
% transferRate --- 0.25 --- Boundary transfer offspring ratio
% archiveRatio  --- 0.45 --- Region archive ratio
% relaxPower    --- 2.5  --- Decay exponent of the promising-region threshold
% pBestRate     --- 0.15 --- Elite ratio used by DE guidance

    methods
        function main(Algorithm, Problem)
            [transferRate, archiveRatio, relaxPower, pBestRate] = Algorithm.ParameterSet(0.25, 0.45, 2.5, 0.15);
            transferRate = min(0.50, max(0.05, transferRate));
            archiveRatio  = min(0.80, max(0.10, archiveRatio));
            relaxPower    = max(1.0, relaxPower);
            pBestRate     = min(0.40, max(0.05, pBestRate));

            PopMain   = Problem.Initialization();
            PopRegion = Problem.Initialization();
            PopObj    = Problem.Initialization();
            Archive   = PopMain(1:0);

            initCV = RIFT.fullCV([PopMain, PopRegion, PopObj]);
            cv0 = max(initCV);
            if isempty(cv0) || cv0 <= 0
                cv0 = 1;
            end
            archiveSize = max(8, ceil(archiveRatio * Problem.N));

            while Algorithm.NotTerminated(PopMain)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                epsilon = RIFT.relaxThreshold([PopMain, PopRegion, Archive], cv0, phase, relaxPower);

                Archive = RIFT.updateArchive([PopMain, PopRegion, PopObj, Archive], archiveSize, epsilon, phase);

                mainPool = RIFT.safePool([PopMain, Archive], PopMain);
                [rankMain, crowdMain] = RIFT.matingRank(mainPool, 'main', 0, phase);
                OffMain = RIFT.generateOffspring(Problem, mainPool, rankMain, crowdMain, Problem.N, pBestRate, 0.20);

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
                mainFeasRate = RIFT.feasibleRate(PopMain);
                if phase > 0.30 && mainFeasRate < 0.65
                    refineCount = min(ceil(0.22 * Problem.N), ...
                        ceil((0.08 + 0.20 * (0.65 - mainFeasRate)) * Problem.N));
                else
                    refineCount = 0;
                end
                OffRefine = RIFT.generateRegionRefinement(Problem, ...
                    RIFT.safePool([Archive, objBridge, PopRegion, PopMain], PopMain), refineCount, phase);

                transferCount = max(2, ceil(transferRate * Problem.N * (1.15 - 0.65 * RIFT.feasibleRate(PopMain))));
                transferCount = min(ceil(0.35 * Problem.N), transferCount);
                transferPool = RIFT.safePool([Archive, objBridge, OffRefine, PopRegion, PopMain], PopMain);
                [rankTransfer, crowdTransfer] = RIFT.matingRank(transferPool, 'transfer', 0.50 * epsilon, phase);
                OffTransfer = RIFT.generateOffspring(Problem, transferPool, rankTransfer, crowdTransfer, ...
                    transferCount, min(0.35, pBestRate + 0.10), 0.05);

                PopMain = RIFT.mainSelection([PopMain, OffMain, OffTransfer, OffRefine, Archive, ...
                    objBridge, RIFT.lowCVSample(OffRegion, ceil(0.25 * Problem.N))], Problem.N, phase);

                PopRegion = RIFT.regionSelection([PopRegion, OffRegion, OffTransfer, OffRefine, OffMain, ...
                    objBridge, PopMain, Archive], ...
                    Problem.N, epsilon, phase);

                PopObj = RIFT.objectiveSelection([PopObj, OffObj, RIFT.objectiveSample(OffMain, ceil(0.20 * Problem.N)), ...
                    RIFT.objectiveSample(OffRegion, ceil(0.20 * Problem.N))], Problem.N, phase);
            end
        end
    end

    methods(Static)
        function Population = mainSelection(Population, N, phase)
            if isempty(Population)
                return;
            end
            CV = RIFT.fullCV(Population);
            feasible = CV <= 1e-12;
            if sum(feasible) >= N
                Population = RIFT.selectByObjective(Population(feasible), N, phase);
                return;
            end

            Next = false(1, numel(Population));
            if any(feasible)
                Next(find(feasible)) = true;
            end
            slots = N - sum(Next);
            if slots > 0
                idx = find(~feasible);
                score = RIFT.infeasibleScore(Population(idx), CV(idx), phase, 0.80);
                [~, order] = sort(score, 'ascend');
                Next(idx(order(1:min(slots, numel(order))))) = true;
            end
            Population = Population(Next);
        end

        function Population = regionSelection(Population, N, epsilon, phase)
            if isempty(Population)
                return;
            end
            CV = RIFT.fullCV(Population);
            near = CV <= max(epsilon, 1e-12);
            if sum(near) >= N
                Population = RIFT.selectByAugmentedObjective(Population(near), N, phase);
                return;
            end

            Next = false(1, numel(Population));
            if any(near)
                Next(find(near)) = true;
            end
            slots = N - sum(Next);
            if slots > 0
                idx = find(~near);
                score = RIFT.infeasibleScore(Population(idx), CV(idx), phase, 0.62);
                [~, order] = sort(score, 'ascend');
                Next(idx(order(1:min(slots, numel(order))))) = true;
            end
            Population = Population(Next);
        end

        function Population = objectiveSelection(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            Population = RIFT.selectByObjective(Population, N, max(0, phase - 0.15));
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
                decCrowd = RIFT.decisionCrowding(Population(Last).decs);
                mixCrowd = (0.78 + 0.12 * phase) * RIFT.scaleToUnit(crowd(:)) + ...
                           (0.22 - 0.12 * phase) * RIFT.scaleToUnit(decCrowd(:));
                [~, order] = sort(mixCrowd, 'descend');
                Next(Last(order(1:N - sum(Next)))) = true;
            end
            Population = Population(Next);
        end

        function Population = selectByAugmentedObjective(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            CV = RIFT.scaleToUnit(RIFT.fullCV(Population));
            AugObj = [RIFT.normalizeColumns(Population.objs), CV(:)];
            [FrontNo, MaxFNo] = NDSort(AugObj, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            if sum(Next) < N && ~isempty(Last)
                crowd = CrowdingDistance(AugObj(Last, :), FrontNo(Last));
                decCrowd = RIFT.decisionCrowding(Population(Last).decs);
                cvBonus = 1 - RIFT.scaleToUnit(CV(Last));
                mixCrowd = (0.58 + 0.20 * phase) * RIFT.scaleToUnit(crowd(:)) + ...
                           0.22 * RIFT.scaleToUnit(decCrowd(:)) + ...
                           (0.20 - 0.10 * phase) * cvBonus(:);
                [~, order] = sort(mixCrowd, 'descend');
                Next(Last(order(1:N - sum(Next)))) = true;
            end
            Population = Population(Next);
        end

        function Archive = updateArchive(Pool, K, epsilon, phase)
            if isempty(Pool)
                Archive = Pool;
                return;
            end
            CV = RIFT.fullCV(Pool);
            sortedCV = sort(CV, 'ascend');
            qIndex = max(1, min(numel(sortedCV), ceil((0.28 + 0.32 * (1 - phase)) * numel(sortedCV))));
            gate = max([epsilon, sortedCV(qIndex), 1e-12]);
            candidate = CV <= gate;
            if sum(candidate) < min(K, numel(Pool))
                [~, order] = sort(CV, 'ascend');
                candidate(order(1:min(numel(order), max(K, sum(candidate))))) = true;
            end

            Cand = Pool(candidate);
            CandCV = RIFT.fullCV(Cand);
            AugObj = [RIFT.normalizeColumns(Cand.objs), 0.55 * RIFT.scaleToUnit(CandCV(:))];
            baseScore = 0.62 * RIFT.scaleToUnit(CandCV(:)) + ...
                        0.38 * RIFT.scaleToUnit(sum(RIFT.normalizeColumns(Cand.objs), 2));
            pick = RIFT.greedyDiversePick(AugObj, baseScore, min(K, numel(Cand)));
            Archive = Cand(pick);
        end

        function epsilon = relaxThreshold(Population, cv0, phase, relaxPower)
            if isempty(Population)
                epsilon = cv0 * max(0, 1 - phase) ^ relaxPower;
                return;
            end
            CV = RIFT.fullCV(Population);
            CV = CV(isfinite(CV));
            if isempty(CV)
                epsilon = cv0 * max(0, 1 - phase) ^ relaxPower;
                return;
            end
            q = 0.72 - 0.42 * phase;
            q = min(0.75, max(0.20, q));
            sortedCV = sort(CV, 'ascend');
            adaptive = sortedCV(max(1, min(numel(sortedCV), ceil(q * numel(sortedCV)))));
            scheduled = cv0 * max(0, 1 - phase) ^ relaxPower;
            epsilon = max(0, min(adaptive, scheduled));
        end

        function [rank, crowd] = matingRank(Population, mode, epsilon, phase)
            if isempty(Population)
                rank = [];
                crowd = [];
                return;
            end
            Obj = RIFT.normalizeColumns(Population.objs);
            CV = RIFT.fullCV(Population);
            if isinf(epsilon)
                relaxedCV = zeros(size(CV));
            else
                relaxedCV = max(0, CV - max(0, epsilon));
            end
            cvPenalty = RIFT.scaleToUnit(relaxedCV(:));

            switch lower(mode)
                case 'objective'
                    RankObj = Obj;
                    penaltyWeight = 0;
                case 'region'
                    RankObj = [Obj, 0.70 * RIFT.scaleToUnit(CV(:))];
                    penaltyWeight = 0.20 + 0.25 * phase;
                case 'transfer'
                    RankObj = Obj + 0.35 * cvPenalty .* (1 + cvPenalty);
                    penaltyWeight = 0.35 + 0.25 * phase;
                otherwise
                    RankObj = Obj + 0.80 * cvPenalty .* (1 + cvPenalty);
                    penaltyWeight = 0.65 + 0.25 * phase;
            end

            [front, ~] = NDSort(RankObj, inf);
            crowd = CrowdingDistance(RankObj, front);
            crowd = RIFT.scaleToUnit(crowd(:)) + 0.18 * RIFT.decisionCrowding(Population.decs);
            rank = front(:) + penaltyWeight * cvPenalty(:);
        end

        function Offspring = generateOffspring(Problem, Source, rank, crowd, count, pBestRate, gaShare)
            Empty = Source(1:0);
            if isempty(Source) || count <= 0
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            if numel(Source) < 3
                parents = TournamentSelection(2, 2 * count, rank, -crowd);
                Offspring = OperatorGAhalf(Problem, Source(parents));
                return;
            end

            gaCount = min(count, max(1, round(gaShare * count)));
            deCount = max(0, count - gaCount);

            parents = TournamentSelection(2, 2 * gaCount, rank, -crowd);
            OffGA = OperatorGAhalf(Problem, Source(parents));

            if deCount > 0
                base = TournamentSelection(2, deCount, rank, -crowd);
                [~, order] = sortrows([rank(:), -crowd(:)], [1 2]);
                pCount = max(2, min(numel(Source), ceil(pBestRate * numel(Source))));
                guide = order(randi(pCount, 1, deCount));
                rand1 = randi(numel(Source), 1, deCount);
                F = 0.45 + 0.25 * rand(1);
                OffDE = OperatorDE(Problem, Source(base), Source(guide), Source(rand1), {0.95, F, 1, 20});
            else
                OffDE = Empty;
            end
            Offspring = [OffGA, OffDE];
        end

        function Offspring = generateRegionRefinement(Problem, Source, count, phase)
            Empty = Source(1:0);
            if isempty(Source) || count <= 0
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            if numel(Source) < 4
                Offspring = Empty;
                return;
            end

            CV = RIFT.fullCV(Source);
            Obj = sum(RIFT.normalizeColumns(Source.objs), 2);
            crowd = RIFT.decisionCrowding(Source.decs);
            score = 0.70 * RIFT.scaleToUnit(CV(:)) + ...
                    0.24 * RIFT.scaleToUnit(Obj(:)) - ...
                    0.06 * RIFT.scaleToUnit(crowd(:));
            [~, order] = sort(score, 'ascend');
            eliteCount = max(4, min(numel(Source), ceil(0.25 * numel(Source))));
            eliteIdx = order(1:eliteCount);

            baseIdx = order(randi(max(eliteCount, min(numel(order), ceil(0.65 * numel(order)))), 1, count));
            guideIdx = eliteIdx(randi(eliteCount, 1, count));
            donorA = eliteIdx(randi(eliteCount, 1, count));
            donorB = eliteIdx(randi(eliteCount, 1, count));

            Dec = Source.decs;
            weights = RIFT.decisionRegionWeights(Source, eliteIdx);
            weights = repmat(weights, count, 1);
            span = repmat(Problem.upper - Problem.lower, count, 1);
            lower = repmat(Problem.lower, count, 1);
            upper = repmat(Problem.upper, count, 1);

            Base = Dec(baseIdx, :);
            Guide = Dec(guideIdx, :);
            Diff = Dec(donorA, :) - Dec(donorB, :);
            alpha = 0.28 + 0.42 * phase + 0.18 * rand(count, 1);
            jitter = (0.035 * (1 - phase) + 0.006) .* randn(count, size(Dec, 2)) .* span;

            OffDec = Base + repmat(alpha, 1, size(Dec, 2)) .* weights .* (Guide - Base) + ...
                     0.16 * (1 - weights) .* Diff + jitter .* (1 - 0.55 * weights);
            OffDec = min(max(OffDec, lower), upper);
            Offspring = Problem.Evaluation(OffDec);
        end

        function weights = decisionRegionWeights(Population, eliteIdx)
            Dec = Population.decs;
            [N, D] = size(Dec);
            if N < 4 || isempty(eliteIdx)
                weights = ones(1, D);
                return;
            end
            Dec = RIFT.normalizeColumns(Dec);
            eliteIdx = eliteIdx(:)';
            allStd = std(Dec, 0, 1);
            eliteStd = std(Dec(eliteIdx, :), 0, 1);
            contraction = max(0, allStd - eliteStd) ./ max(allStd, eps);
            weights = 0.35 + 0.65 * RIFT.scaleToUnit(contraction);
            weights(~isfinite(weights)) = 0.50;
        end

        function Sample = lowCVSample(Population, K)
            if isempty(Population) || K <= 0
                Sample = Population(1:0);
                return;
            end
            CV = RIFT.fullCV(Population);
            obj = sum(RIFT.normalizeColumns(Population.objs), 2);
            decCrowd = RIFT.decisionCrowding(Population.decs);
            score = 0.74 * RIFT.scaleToUnit(CV(:)) + ...
                    0.20 * RIFT.scaleToUnit(obj(:)) - ...
                    0.06 * RIFT.scaleToUnit(decCrowd(:));
            [~, order] = sort(score, 'ascend');
            Sample = Population(order(1:min(K, numel(order))));
        end

        function Sample = objectiveSample(Population, K)
            if isempty(Population) || K <= 0
                Sample = Population(1:0);
                return;
            end
            Obj = Population.objs;
            [FrontNo, ~] = NDSort(Obj, min(K, numel(Population)));
            crowd = CrowdingDistance(Obj, FrontNo);
            [~, order] = sortrows([FrontNo(:), -crowd(:)], [1 2]);
            Sample = Population(order(1:min(K, numel(order))));
        end

        function score = infeasibleScore(Population, CV, phase, cvWeight)
            Obj = RIFT.normalizeColumns(Population.objs);
            objScore = RIFT.scaleToUnit(sum(Obj, 2));
            cvScore = RIFT.scaleToUnit(CV(:));
            decScore = 1 - RIFT.scaleToUnit(RIFT.decisionCrowding(Population.decs));
            wCV = min(0.94, cvWeight + 0.10 * phase);
            wObj = 0.20 - 0.08 * phase;
            wDec = max(0, 1 - wCV - wObj);
            score = wCV * cvScore(:) + wObj * objScore(:) + wDec * decScore(:);
        end

        function CV = fullCV(Population)
            if isempty(Population) || isempty(Population.cons)
                CV = zeros(numel(Population), 1);
                return;
            end
            CV = sum(max(Population.cons, 0), 2);
        end

        function rate = feasibleRate(Population)
            CV = RIFT.fullCV(Population);
            if isempty(CV)
                rate = 0;
            else
                rate = mean(CV <= 1e-12);
            end
        end

        function Pool = safePool(Pool, fallback)
            if isempty(Pool)
                Pool = fallback;
            end
        end

        function pick = greedyDiversePick(Obj, score, K)
            N = size(Obj, 1);
            if N <= K
                pick = 1:N;
                return;
            end
            Obj = RIFT.normalizeColumns(Obj);
            [~, first] = min(score);
            pick = first;
            while numel(pick) < K
                remain = setdiff(1:N, pick);
                d = RIFT.minDistance(Obj(remain, :), Obj(pick, :));
                value = 0.56 * RIFT.scaleToUnit(d(:)) + 0.44 * (1 - RIFT.scaleToUnit(score(remain)));
                [~, best] = max(value);
                pick(end+1) = remain(best); %#ok<AGROW>
            end
        end

        function crowd = decisionCrowding(Dec)
            N = size(Dec, 1);
            if N <= 2
                crowd = ones(N, 1);
                return;
            end
            Dec = RIFT.normalizeColumns(Dec);
            Distance = pdist2(Dec, Dec);
            Distance(logical(eye(N))) = inf;
            Distance = sort(Distance, 2, 'ascend');
            crowd = Distance(:, max(1, min(N - 1, floor(sqrt(N)))));
            crowd = RIFT.scaleToUnit(crowd);
        end

        function M = normalizeColumns(M)
            if isempty(M)
                return;
            end
            minV = min(M, [], 1);
            maxV = max(M, [], 1);
            span = max(maxV - minV, eps);
            M = (M - minV) ./ span;
        end

        function y = scaleToUnit(x)
            shape = size(x);
            x = x(:);
            finiteMask = isfinite(x);
            y = zeros(size(x));
            if any(finiteMask)
                vals = x(finiteMask);
                minV = min(vals);
                maxV = max(vals);
                if maxV - minV <= eps
                    y(finiteMask) = 0.5;
                else
                    y(finiteMask) = (vals - minV) ./ (maxV - minV);
                end
            end
            y(isinf(x) & x > 0) = 1;
            y(isinf(x) & x < 0) = 0;
            y = reshape(y, shape);
        end

        function d = minDistance(A, B)
            d = zeros(size(A, 1), 1);
            for i = 1:size(A, 1)
                diff = B - A(i, :);
                d(i) = min(sqrt(sum(diff.^2, 2)));
            end
        end
    end
end
