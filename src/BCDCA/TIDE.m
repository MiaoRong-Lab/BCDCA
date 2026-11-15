classdef TIDE < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Evolutionary operators and environmental selection used by BCDCA
% archiveRatio --- 0.30 --- Final feasible archive ratio

    methods
        function main(Algorithm, Problem)
            archiveRatio = Algorithm.ParameterSet(0.30);
            archiveRatio = min(0.80, max(0.10, archiveRatio));

            Probe = Problem.Initialization();
            initCV = RIFT.fullCV(Probe);
            initFeasRate = mean(initCV <= 1e-12);
            initMedianCV = median(initCV);
            consCount = size(Probe.cons, 2);
            hardMode = Problem.D >= 30 && ...
                (consCount > Problem.M || ...
                (consCount >= Problem.M && initFeasRate < 0.20 && initMedianCV > 50));

            if hardMode
                TIDE.dualTaskMode(Algorithm, Problem, Probe, archiveRatio);
            else
                TIDE.generalMode(Algorithm, Problem, Probe, archiveRatio);
            end
        end
    end

    methods(Static)
        function dualTaskMode(Algorithm, Problem, PopMain, archiveRatio)
            PopAux = Problem.Initialization();
            PopTrace = Problem.Initialization();
            FeasArchive = PopMain(1:0);
            archiveSize = max(8, ceil(archiveRatio * Problem.N));
            FitnessMain = TIDE.calFitness(PopMain.objs, PopMain.cons);
            FitnessAux  = TIDE.calFitness(PopAux.objs,  PopAux.cons);
            FitnessTrace = TIDE.violationFitness(PopTrace);

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
            reseedDone = false;
            PopOutput = TIDE.outputSelection([PopMain, PopAux, PopTrace, FeasArchive], Problem.N, 0);

            while Algorithm.NotTerminated(PopOutput)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                VAR = VAR0 * max(0, 1 - X) ^ cp;
                currentFeasSeen = ~isempty(FeasArchive) || ...
                    any(TIDE.totalCV(PopMain) <= 1e-12) || ...
                    any(TIDE.totalCV(PopAux) <= 1e-12) || ...
                    any(TIDE.totalCV(PopTrace) <= 1e-12);
                if phase > 0.52 && ~currentFeasSeen && ~reseedDone
                    Fresh = Problem.Initialization();
                    [PopAux, FitnessAux] = TIDE.auxTaskSelection([PopAux, Fresh, PopTrace], Problem.N, VAR);
                    PopTrace = TIDE.manifoldSelection([PopTrace, Fresh, PopMain], Problem.N, phase);
                    FitnessTrace = TIDE.violationFitness(PopTrace);
                    ZminAux = min([ZminAux; Fresh.objs], [], 1);
                    reseedDone = true;
                end
                halfN = floor(Problem.N / 2);
                restN = Problem.N - halfN;

                MatingMain = PopMain(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingMain, ZminMain);
                OffMain(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));
                OffMain(halfN+1:Problem.N) = TIDE.operatorPBestDE(Problem, PopMain, FitnessMain, restN, 0.10);

                MatingAux = PopAux(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingAux, ZminAux);
                OffAux(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));
                OffAux(halfN+1:Problem.N) = TIDE.operatorPBestDE(Problem, PopAux, FitnessAux, restN, 0.10);

                tracePool = [PopTrace, PopMain, PopAux, FeasArchive];
                traceFitness = TIDE.violationFitness(tracePool);
                bestTraceCV = min(TIDE.totalCV(tracePool));
                feasSeen = ~isempty(FeasArchive) || ...
                    any(TIDE.totalCV(PopMain) <= 1e-12) || ...
                    any(TIDE.totalCV(PopAux) <= 1e-12) || ...
                    any(TIDE.totalCV(PopTrace) <= 1e-12);
                if phase > 0.60 && ~feasSeen
                    traceBudget = ceil(1.15 * Problem.N);
                elseif phase > 0.20
                    traceBudget = ceil(0.70 * Problem.N);
                else
                    traceBudget = 0;
                end
                if phase > 0.60 && ~feasSeen
                    polishCount = ceil(0.75 * traceBudget);
                elseif phase > 0.45 && bestTraceCV < 0.10
                    if phase > 0.75 && bestTraceCV < 0.05
                        polishCount = ceil(0.45 * Problem.N);
                    else
                        polishCount = ceil(0.30 * Problem.N);
                    end
                else
                    polishCount = 0;
                end
                polishCount = min(polishCount, max(0, traceBudget - 2));
                traceCount = traceBudget - polishCount;
                OffTrace = TIDE.operatorCVPBestDE(Problem, tracePool, traceFitness, traceCount, 0.12, phase);
                OffPolish = TIDE.operatorCVLocalSearch(Problem, tracePool, traceFitness, polishCount, phase);

                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive, PopMain, PopAux, PopTrace, OffMain, OffAux, OffTrace, OffPolish], ...
                    archiveSize, phase);
                ZminMain = min([ZminMain; OffMain.objs], [], 1);
                ZminAux  = min([ZminAux;  OffAux.objs],  [], 1);

                PopTrace = TIDE.manifoldSelection([PopTrace, OffTrace, OffPolish, PopMain, PopAux, OffMain, OffAux, FeasArchive], ...
                    Problem.N, phase);
                FitnessTrace = TIDE.violationFitness(PopTrace);

                [PopMain, ~] = TIDE.mainTaskSelection([PopMain, OffMain, OffAux, OffTrace, OffPolish, FeasArchive], Problem.N, true);
                [PopAux,  FitnessAux]  = TIDE.auxTaskSelection([PopAux, OffAux, OffMain, OffTrace, OffPolish, PopTrace, FeasArchive], Problem.N, VAR);
                [PopMain, FitnessMain] = TIDE.mainTaskSelection([PopMain, FeasArchive, PopTrace], Problem.N, true);
                PopOutput = TIDE.outputSelection([PopMain, PopAux, PopTrace, FeasArchive], Problem.N, phase);
                X = X + 1 / (Problem.maxFE / Problem.N);
            end
        end

        function generalMode(Algorithm, Problem, PopMain, archiveRatio)
            DRIFT.regionTransferMode(Algorithm, Problem, PopMain, archiveRatio, 0.10, 2.5);
        end

        function Offspring = generateHybridDE(Problem, Population, Fitness, count, p)
            Empty = Population(1:0);
            if isempty(Population) || count <= 0
                Offspring = Empty;
                return;
            end
            if numel(Population) < 4
                Offspring = OperatorGAhalf(Problem, Population(randi(numel(Population), 1, 2 * count)));
                return;
            end

            count = max(1, round(count));
            randCount = floor(count / 2);
            pbestCount = count - randCount;

            idx1 = randi(numel(Population), 1, randCount);
            idx2 = randi(numel(Population), 1, randCount);
            idx3 = randi(numel(Population), 1, randCount);
            OffRand = TIDE.operatorRandDE(Problem, Population(idx1), Population(idx2), Population(idx3));

            OffPBest = TIDE.operatorPBestDE(Problem, Population, Fitness, pbestCount, p);
            Offspring = [OffRand, OffPBest];
        end

        function Offspring = operatorRandDE(Problem, Parent1, Parent2, Parent3)
            if isempty(Parent1)
                Offspring = Parent1;
                return;
            end
            P1 = Parent1.decs;
            P2 = Parent2.decs;
            P3 = Parent3.decs;
            [N, D] = size(P1);
            FList = [0.6, 0.8, 1.0];
            CRList = [0.1, 0.2, 1.0];
            F = FList(randi(numel(FList), N, 1));
            CR = CRList(randi(numel(CRList), N, 1));
            F = repmat(F(:), 1, D);
            Site = rand(N, D) < repmat(CR(:), 1, D);
            OffDec = P1;
            OffDec(Site) = OffDec(Site) + F(Site) .* (P2(Site) - P3(Site));
            OffDec = TIDE.polynomialMutation(Problem, OffDec);
            Offspring = Problem.Evaluation(OffDec);
        end

        function Offspring = operatorPBestDE(Problem, Population, Fitness, count, p)
            Empty = Population(1:0);
            if count <= 0 || isempty(Population)
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            [~, order] = sort(Fitness, 'ascend');
            pCount = max(2, min(numel(Population), ceil(p * numel(Population))));
            base = randi(numel(Population), 1, count);
            guide = order(randi(pCount, 1, count));
            r1 = randi(numel(Population), 1, count);
            r2 = randi(numel(Population), 1, count);

            Base = Population(base).decs;
            Guide = Population(guide).decs;
            R1 = Population(r1).decs;
            R2 = Population(r2).decs;
            [N, D] = size(Base);
            FList = [0.6, 0.8, 1.0];
            CRList = [0.1, 0.2, 1.0];
            F = FList(randi(numel(FList), N, 1));
            CR = CRList(randi(numel(CRList), N, 1));
            F = repmat(F(:), 1, D);
            Site = rand(N, D) < repmat(CR(:), 1, D);
            OffDec = Base;
            OffDec(Site) = OffDec(Site) + F(Site) .* (Guide(Site) - Base(Site) + R1(Site) - R2(Site));
            OffDec = TIDE.polynomialMutation(Problem, OffDec);
            Offspring = Problem.Evaluation(OffDec);
        end

        function Offspring = operatorCVPBestDE(Problem, Population, Fitness, count, p, phase)
            Empty = Population(1:0);
            if count <= 0 || isempty(Population)
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            if nargin < 6
                phase = 0;
            end
            [~, order] = sort(Fitness, 'ascend');
            pCount = max(2, min(numel(Population), ceil(p * numel(Population))));
            eliteCount = max(pCount, min(numel(Population), ceil(0.45 * numel(Population))));
            base = order(randi(eliteCount, 1, count));
            guide = order(randi(pCount, 1, count));
            r1 = randi(numel(Population), 1, count);
            r2 = randi(numel(Population), 1, count);

            Base = Population(base).decs;
            Guide = Population(guide).decs;
            R1 = Population(r1).decs;
            R2 = Population(r2).decs;
            [N, D] = size(Base);
            if phase < 0.55
                FList = [0.35, 0.50, 0.70];
                CRList = [0.10, 0.20, 0.40];
            else
                FList = [0.18, 0.28, 0.42];
                CRList = [0.05, 0.10, 0.20];
            end
            F = FList(randi(numel(FList), N, 1));
            CR = CRList(randi(numel(CRList), N, 1));
            F = repmat(F(:), 1, D);
            Site = rand(N, D) < repmat(CR(:), 1, D);
            forced = sub2ind([N, D], (1:N)', randi(D, N, 1));
            Site(forced) = true;
            OffDec = Base;
            OffDec(Site) = OffDec(Site) + F(Site) .* ...
                (Guide(Site) - Base(Site) + 0.35 .* (R1(Site) - R2(Site)));
            OffDec = TIDE.polynomialMutation(Problem, OffDec);
            Offspring = Problem.Evaluation(OffDec);
        end

        function Offspring = operatorCVLocalSearch(Problem, Population, Fitness, count, phase)
            Empty = Population(1:0);
            if count <= 0 || isempty(Population)
                Offspring = Empty;
                return;
            end
            count = max(1, round(count));
            [~, order] = sort(Fitness, 'ascend');
            eliteCount = max(2, min(numel(Population), ceil(0.20 * numel(Population))));
            base = order(randi(eliteCount, 1, count));
            OffDec = Population(base).decs;
            [N, D] = size(OffDec);
            Lower = repmat(Problem.lower, N, 1);
            Upper = repmat(Problem.upper, N, 1);
            scale = Upper - Lower;
            sigma = (0.060 * max(0, 1 - phase) + 0.004) .* scale;
            siteRate = (0.22 + 0.18 * phase) * ones(N, 1);
            microRows = false(N, 1);
            microRows(ceil(N/2)+1:end) = true;
            sigma(microRows, :) = (0.012 * max(0, 1 - phase) + 0.0008) .* scale(microRows, :);
            siteRate(microRows) = 0.10 + 0.12 * phase;
            Site = rand(N, D) < repmat(siteRate, 1, D);
            forced = sub2ind([N, D], (1:N)', randi(D, N, 1));
            Site(forced) = true;
            Step = randn(N, D) .* sigma;
            OffDec(Site) = OffDec(Site) + Step(Site);
            OffDec = min(max(OffDec, Lower), Upper);
            Offspring = Problem.Evaluation(OffDec);
        end

        function OffDec = polynomialMutation(Problem, OffDec)
            [N, D] = size(OffDec);
            proM = 1;
            disM = 20;
            Lower = repmat(Problem.lower, N, 1);
            Upper = repmat(Problem.upper, N, 1);
            OffDec = min(max(OffDec, Lower), Upper);
            Site = rand(N, D) < proM / D;
            mu = rand(N, D);
            temp = Site & mu <= 0.5;
            OffDec(temp) = OffDec(temp) + (Upper(temp) - Lower(temp)) .* ...
                ((2 .* mu(temp) + (1 - 2 .* mu(temp)) .* ...
                (1 - (OffDec(temp) - Lower(temp)) ./ (Upper(temp) - Lower(temp))).^(disM + 1)).^(1 / (disM + 1)) - 1);
            temp = Site & mu > 0.5;
            OffDec(temp) = OffDec(temp) + (Upper(temp) - Lower(temp)) .* ...
                (1 - (2 .* (1 - mu(temp)) + 2 .* (mu(temp) - 0.5) .* ...
                (1 - (Upper(temp) - OffDec(temp)) ./ (Upper(temp) - Lower(temp))).^(disM + 1)).^(1 / (disM + 1)));
            OffDec = min(max(OffDec, Lower), Upper);
        end

        function [Mate1, Mate2, Mate3] = neighborPairing(MatingPop, Zmin)
            Objs = MatingPop.objs;
            [N, M] = size(Objs);
            Objs = Objs - repmat(Zmin, N, 1);
            normValue = sqrt(sum(Objs.^2, 2));
            normValue(normValue <= eps) = 1;
            Objs = Objs ./ repmat(normValue, 1, M);
            CosV = Objs * Objs' - 3 * eye(N);
            [~, order] = sort(-CosV, 2);
            neighborCount = min(10, N);
            Neighbor = order(:, 1:neighborCount);
            Pair = ones(N, 2);
            for i = 1:N
                pick = randperm(neighborCount, min(2, neighborCount));
                if isscalar(pick)
                    pick = [pick, pick];
                end
                Pair(i, :) = Neighbor(i, pick(1:2));
            end
            Mate1 = MatingPop;
            Mate2 = MatingPop(Pair(:, 1));
            Mate3 = MatingPop(Pair(:, 2));
        end

        function [Population, Fitness] = mainTaskSelection(Population, N, isOrigin)
            if isOrigin
                Fitness = TIDE.calFitness(Population.objs, Population.cons);
            else
                Fitness = TIDE.calFitness(Population.objs);
            end
            Next = Fitness < 1;
            if sum(Next) < N
                [~, rank] = sort(Fitness);
                Next(rank(1:N)) = true;
            elseif sum(Next) > N
                Del = TIDE.truncation(Population(Next).objs, sum(Next) - N);
                temp = find(Next);
                Next(temp(Del)) = false;
            end
            Population = Population(Next);
            Fitness = Fitness(Next);
            [Fitness, rank] = sort(Fitness);
            Population = Population(rank);
        end

        function [Population, Fitness] = auxTaskSelection(Population, N, VAR)
            CV = Population.cons;
            CV(CV < 0) = 0;
            CV = sum(CV, 2);
            feasible = CV <= VAR;
            infeasible = CV > VAR;
            FeasPop = Population(feasible);
            InfPop  = Population(infeasible);

            if isempty(FeasPop)
                InfFitness = TIDE.calFitness(InfPop.objs, InfPop.cons);
                Next = InfFitness < 1;
                if sum(Next) <= N
                    [~, rank] = sort(InfFitness);
                    Next(rank(1:N)) = true;
                else
                    Del = TIDE.truncation(InfPop(Next).objs, sum(Next) - N);
                    temp = find(Next);
                    Next(temp(Del)) = false;
                end
                Population = InfPop(Next);
                Fitness = InfFitness(Next);
                [Fitness, rank] = sort(Fitness);
                Population = Population(rank);
                return;
            end

            if numel(FeasPop) <= N
                cons = FeasPop.cons;
                cons(cons < 0) = 0;
                cons = sum(cons, 2);
                FeasFitness = TIDE.calFitness([FeasPop.objs, cons]);
                Next = FeasFitness < 1;
                [~, rank] = sort(FeasFitness);
                Next(rank(1:numel(FeasPop))) = true;
                FeasPop = FeasPop(Next);
                FeasFitness = FeasFitness(Next);
                [FeasFitness, rank] = sort(FeasFitness);
                FeasPop = FeasPop(rank);

                InfFitness = TIDE.calFitness(InfPop.objs, InfPop.cons);
                NextInf = InfFitness < 1;
                slots = N - numel(FeasPop);
                if slots > 0
                    if sum(NextInf) <= slots
                        [~, rank] = sort(InfFitness);
                        NextInf(rank(1:min(slots, numel(rank)))) = true;
                    else
                        Del = TIDE.truncation(InfPop(NextInf).objs, sum(NextInf) - slots);
                        temp = find(NextInf);
                        NextInf(temp(Del)) = false;
                    end
                    InfPop = InfPop(NextInf);
                    InfFitness = InfFitness(NextInf) + max(FeasFitness);
                    [InfFitness, rank] = sort(InfFitness);
                    InfPop = InfPop(rank);
                else
                    InfPop = InfPop(1:0);
                    InfFitness = [];
                end
                Population = [FeasPop, InfPop];
                Fitness = [FeasFitness, InfFitness];
                return;
            end

            cons = FeasPop.cons;
            cons(cons < 0) = 0;
            cons = sum(cons, 2);
            FeasFitness = TIDE.calFitness([FeasPop.objs, cons]);
            Next = FeasFitness < 1;
            if sum(Next) <= N
                [~, rank] = sort(FeasFitness);
                Next(rank(1:N)) = true;
            else
                Del = TIDE.truncation(FeasPop(Next).objs, sum(Next) - N);
                temp = find(Next);
                Next(temp(Del)) = false;
            end
            Population = FeasPop(Next);
            Fitness = FeasFitness(Next);
            [Fitness, rank] = sort(Fitness);
            Population = Population(rank);
        end

        function Population = outputSelection(Population, N, phase)
            if isempty(Population)
                return;
            end
            CV = TIDE.totalCV(Population);
            feasible = CV <= 1e-12;
            if sum(feasible) >= N
                [Population, ~] = TIDE.mainTaskSelection(Population(feasible), N, true);
                return;
            end
            if any(feasible)
                FeasPop = Population(feasible);
                if numel(FeasPop) > N
                    [FeasPop, ~] = TIDE.mainTaskSelection(FeasPop, N, true);
                    Population = FeasPop;
                    return;
                end
                InfPop = Population(~feasible);
                slots = N - numel(FeasPop);
                if slots > 0 && ~isempty(InfPop)
                    InfPop = TIDE.manifoldSelection(InfPop, slots, phase);
                else
                    InfPop = InfPop(1:0);
                end
                Population = [FeasPop, InfPop];
                return;
            end
            Population = TIDE.manifoldSelection(Population, N, phase);
        end

        function Population = manifoldSelection(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            CV = TIDE.totalCV(Population);
            [~, order] = sort(CV, 'ascend');
            eliteCount = min(numel(Population), ceil(0.68 * N));
            Next = false(1, numel(Population));
            Next(order(1:eliteCount)) = true;

            slots = N - sum(Next);
            if slots > 0
                cand = order(eliteCount+1:end);
                objScore = sum(TIDE.normalizeColumns(Population(cand).objs), 2);
                decCrowd = TIDE.decisionCrowding(Population(cand).decs);
                score = (0.86 + 0.08 * phase) * TIDE.scaleToUnit(CV(cand)) + ...
                    0.10 * TIDE.scaleToUnit(objScore) - ...
                    0.04 * TIDE.scaleToUnit(decCrowd);
                [~, rank] = sort(score, 'ascend');
                Next(cand(rank(1:min(slots, numel(rank))))) = true;
            end

            Population = Population(Next);
            if numel(Population) > N
                Fitness = TIDE.violationFitness(Population);
                [~, order] = sort(Fitness, 'ascend');
                Population = Population(order(1:N));
            end
        end

        function Fitness = violationFitness(Population)
            CV = TIDE.totalCV(Population);
            objScore = sum(TIDE.normalizeColumns(Population.objs), 2);
            decCrowd = TIDE.decisionCrowding(Population.decs);
            Fitness = 0.90 * TIDE.scaleToUnit(CV) + ...
                0.08 * TIDE.scaleToUnit(objScore) - ...
                0.02 * TIDE.scaleToUnit(decCrowd);
        end

        function CV = totalCV(Population)
            CV = Population.cons;
            CV(CV < 0) = 0;
            CV = sum(CV, 2);
        end

        function X = normalizeColumns(X)
            if isempty(X)
                return;
            end
            xmin = min(X, [], 1);
            xmax = max(X, [], 1);
            span = xmax - xmin;
            span(span <= eps) = 1;
            X = (X - xmin) ./ span;
        end

        function y = scaleToUnit(x)
            x = x(:);
            if isempty(x)
                y = x;
                return;
            end
            xmin = min(x);
            xmax = max(x);
            if xmax - xmin <= eps
                y = zeros(size(x));
            else
                y = (x - xmin) ./ (xmax - xmin);
            end
        end

        function Crowd = decisionCrowding(PopDec)
            N = size(PopDec, 1);
            if N <= 2
                Crowd = ones(N, 1);
                return;
            end
            Distance = pdist2(PopDec, PopDec);
            Distance(logical(eye(N))) = inf;
            Distance = sort(Distance, 2);
            Crowd = Distance(:, min(2, size(Distance, 2)));
        end

        function Del = truncation(PopObj, K)
            Distance = pdist2(PopObj, PopObj);
            Distance(logical(eye(size(Distance, 1)))) = inf;
            Del = false(1, size(PopObj, 1));
            while sum(Del) < K
                Remain = find(~Del);
                Temp = sort(Distance(Remain, Remain), 2);
                [~, rank] = sortrows(Temp);
                Del(Remain(rank(1))) = true;
            end
        end

        function Population = mainSelection(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            CV = RIFT.fullCV(Population);
            feasible = CV <= 1e-12;
            if sum(feasible) >= N
                Population = RIFT.selectByObjective(Population(feasible), N, phase);
                return;
            end
            Population = DRIFT.feasibilityBackboneSelection(Population, N, phase);
        end

        function Population = relaxedSelection(Population, N, VAR, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            CV = RIFT.fullCV(Population);
            admissible = CV <= max(VAR, 1e-12);
            if sum(admissible) >= N
                Pool = Population(admissible);
                AugObj = [RIFT.normalizeColumns(Pool.objs), 0.30 * RIFT.scaleToUnit(RIFT.fullCV(Pool))];
                Population = DRIFT.selectByAugmentedFront(Pool, AugObj, N, 0.10);
                return;
            end

            Next = false(1, numel(Population));
            if any(admissible)
                Next(admissible) = true;
            end
            slots = N - sum(Next);
            if slots > 0
                idx = find(~Next);
                score = 0.82 * RIFT.scaleToUnit(CV(idx)) + ...
                    0.14 * RIFT.scaleToUnit(sum(RIFT.normalizeColumns(Population(idx).objs), 2)) - ...
                    0.04 * RIFT.scaleToUnit(RIFT.decisionCrowding(Population(idx).decs));
                [~, order] = sort(score, 'ascend');
                Next(idx(order(1:min(slots, numel(order))))) = true;
            end
            Population = Population(Next);
            if numel(Population) > N
                Population = TIDE.mainSelection(Population, N, phase);
            end
        end

        function Fitness = relaxedFitness(Population, VAR, phase)
            CV = max(0, RIFT.fullCV(Population) - max(VAR, 0));
            AugObj = [RIFT.normalizeColumns(Population.objs), ...
                (0.45 + 0.35 * phase) * RIFT.scaleToUnit(CV(:))];
            Fitness = TIDE.calFitness(AugObj);
        end

        function Fitness = calFitness(PopObj, PopCon)
            N = size(PopObj, 1);
            if nargin == 1
                CV = zeros(N, 1);
            else
                CV = sum(max(0, PopCon), 2);
            end

            Dominate = false(N);
            for i = 1:N-1
                for j = i+1:N
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
            R = zeros(1, N);
            for i = 1:N
                R(i) = sum(S(Dominate(:, i)));
            end
            Distance = pdist2(PopObj, PopObj);
            Distance(logical(eye(size(Distance, 1)))) = inf;
            Distance = sort(Distance, 2);
            k = max(1, min(size(Distance, 2), floor(sqrt(N))));
            D = 1 ./ (Distance(:, k) + 2);
            Fitness = R + D';
        end
    end
end
