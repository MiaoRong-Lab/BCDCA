classdef BCDCA < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Bounded Complementary Dual-Channel Algorithm
% archiveRatio --- 0.30 --- Fraction used to size the feasible archive
% rhoMin      --- 0.25 --- Minimum allocation fraction for each channel
    properties(SetAccess = private)
        auditTrace = struct([])
        modeCode = ''
    end

    methods
        function main(Algorithm,Problem)
            [archiveRatio,rhoMin] = Algorithm.ParameterSet(0.30,0.25);
            archiveRatio = min(0.70,max(0.10,archiveRatio));
            if ~isscalar(rhoMin) || ~isfinite(rhoMin) || rhoMin < 0 || rhoMin >= 0.5
                error('BCDCA:InvalidRhoMin','rhoMin must be a finite scalar in [0,0.5).');
            end

            PopMain = Problem.Initialization();
            PopAux = Problem.Initialization();
            PopTrace = Problem.Initialization();
            if BCDCA.isHardDecisionMode([PopMain,PopAux,PopTrace],Problem)
                Algorithm.modeCode = 'hard';
            elseif Problem.D >= 30
                Algorithm.modeCode = 'soft';
            else
                Algorithm.modeCode = 'low';
            end

            switch Algorithm.modeCode
                case 'hard'
                    Algorithm.runHard(Problem,PopMain,PopAux,PopTrace, ...
                        archiveRatio,rhoMin);
                case 'soft'
                    Algorithm.runSoft(Problem,PopMain,PopAux, ...
                        archiveRatio,rhoMin);
                otherwise
                    Algorithm.runLow(Problem,PopMain,PopAux, ...
                        archiveRatio,rhoMin);
            end
        end
    end

    methods(Access = private)
        function runHard(Algorithm,Problem,PopMain,PopAux,PopTrace,archiveRatio,rhoMin)
            FeasArchive = PopMain(1:0);
            archiveSize = max(8,ceil(archiveRatio*Problem.N));
            FitnessMain = BCDCA.calculateFitness(PopMain.objs,PopMain.cons);
            FitnessAux = BCDCA.calculateFitness(PopAux.objs,PopAux.cons);
            ZminMain = min(PopMain.objs,[],1);
            ZminAux = min(PopAux.objs,[],1);
            initCons = [PopMain.cons;PopAux.cons;PopTrace.cons];
            initCons(initCons < 0) = 0;
            VAR0 = max(sum(initCons,2));
            if isempty(VAR0) || VAR0 <= 0
                VAR0 = 1;
            end
            cp = (-log(VAR0)-6)/log(1-0.5);
            X = 0;
            PopOutput = BCDCA.outputSelection([PopMain,PopAux,PopTrace,FeasArchive],Problem.N,0);
            generation = 0;

            while Algorithm.NotTerminated(PopOutput)
                generation = generation+1;
                phase = min(1,Problem.FE/max(Problem.maxFE,1));
                VAR = VAR0*max(0,1-X)^cp;
                allocation = BCDCA.allocateOffspring(PopMain.cons,2*Problem.N,rhoMin);
                feBefore = Problem.FE;
                OffMain = Algorithm.makeHardMain(Problem,PopMain,FitnessMain,ZminMain, ...
                    allocation.constraintCount,phase,[FeasArchive,PopMain,PopAux,PopTrace]);
                OffAux = Algorithm.makeStandardChannel(Problem,PopAux,FitnessAux,ZminAux, ...
                    allocation.relaxCount,0.10);
                Algorithm.recordTrace(generation,allocation,feBefore,Problem.FE);

                tracePool = [PopTrace,PopMain,PopAux,FeasArchive];
                traceFitness = BCDCA.violationFitness(tracePool);
                traceFeasRate = BCDCA.feasibleRate(tracePool);
                if phase > 0.25 && traceFeasRate < 0.80
                    traceCount = ceil((0.35+0.35*max(0,0.80-traceFeasRate))*Problem.N);
                else
                    traceCount = ceil(0.12*Problem.N);
                end
                polishCount = 0;
                if phase > 0.45 && traceFeasRate < 0.35
                    bestTraceCV = min(BCDCA.traceConstraintViolation(tracePool));
                    if bestTraceCV < 0.10
                        polishCount = ceil(0.30*Problem.N);
                    else
                        polishCount = ceil(0.16*Problem.N);
                    end
                end
                OffTrace = BCDCA.operatorCVPBestDE(Problem,tracePool,traceFitness,traceCount,0.12,phase);
                OffPolish = BCDCA.operatorCVLocalSearch(Problem,tracePool,traceFitness,polishCount,phase);

                FeasArchive = BCDCA.updateFeasibleArchive([FeasArchive,PopMain,PopAux,PopTrace, ...
                    OffMain,OffAux,OffTrace,OffPolish],archiveSize,phase);
                ZminMain = min([ZminMain;OffMain.objs],[],1);
                ZminAux = min([ZminAux;OffAux.objs],[],1);
                PopTrace = BCDCA.manifoldSelection([PopTrace,OffTrace,OffPolish,PopMain,PopAux, ...
                    OffMain,OffAux,FeasArchive],Problem.N,phase);
                [PopMain,FitnessMain] = BCDCA.mainTaskSelection([PopMain,OffMain,OffAux, ...
                    OffTrace,OffPolish,PopTrace,FeasArchive],Problem.N,true);
                [PopAux,FitnessAux] = BCDCA.auxTaskSelection([PopAux,OffAux,OffMain, ...
                    OffTrace,OffPolish,PopTrace,FeasArchive],Problem.N,VAR);
                PopOutput = BCDCA.outputSelection([PopMain,PopAux,PopTrace,FeasArchive],Problem.N,phase);
                X = X+1/(Problem.maxFE/Problem.N);
            end
        end

        function runSoft(Algorithm,Problem,PopMain,PopAux,archiveRatio,rhoMin)
            PopFront = PopMain;
            PopScout = PopAux;
            PopFree = PopAux;
            FeasArchive = PopMain(1:0);
            archiveSize = max(8,ceil(archiveRatio*Problem.N));
            FitnessMain = BCDCA.calculateFitness(PopMain.objs,PopMain.cons);
            FitnessAux = BCDCA.calculateFitness(PopAux.objs,PopAux.cons);
            ZminMain = min(PopMain.objs,[],1);
            ZminAux = min(PopAux.objs,[],1);
            initCons = [PopMain.cons;PopAux.cons];
            initCons(initCons < 0) = 0;
            VAR0 = max(sum(initCons,2));
            if isempty(VAR0) || VAR0 <= 0
                VAR0 = 1;
            end
            cp = (-log(VAR0)-6)/log(1-0.5);
            X = 0;
            alpha = BCDCA.lowDimAlpha(Problem);
            [PopFront,RankFront] = BCDCA.selectConstrainedByRegion(PopFront,Problem.N,alpha);
            [PopScout,RankScout] = BCDCA.selectRelaxedByRegion(PopScout,Problem.N,alpha,1,0);
            PopOutput = BCDCA.outputSelection([PopMain,PopAux,PopFront,PopFree,FeasArchive],Problem.N,0);
            generation = 0;

            while Algorithm.NotTerminated(PopOutput)
                generation = generation+1;
                phase = min(1,Problem.FE/max(Problem.maxFE,1));
                VAR = VAR0*max(0,1-X)^cp;
                gen = max(1,ceil(Problem.FE/Problem.N));
                alpha = BCDCA.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                gamma = BCDCA.lowDimPopulationGap(PopFront,PopScout);
                allocation = BCDCA.allocateOffspring(PopMain.cons,2*Problem.N,rhoMin);
                feBefore = Problem.FE;
                OffMain = Algorithm.makeStandardChannel(Problem,PopMain,FitnessMain,ZminMain, ...
                    allocation.constraintCount,0.10);
                OffAux = Algorithm.makeStandardChannel(Problem,PopAux,FitnessAux,ZminAux, ...
                    allocation.relaxCount,0.10);
                Algorithm.recordTrace(generation,allocation,feBefore,Problem.FE);

                refineSource = BCDCA.feasibleRefinementSource([FeasArchive,PopMain,PopAux]);
                if phase > 0.38 && ~isempty(refineSource) && numel(refineSource) >= 6
                    refineCount = ceil(0.22*Problem.N);
                    refineFitness = BCDCA.objectiveRefinementFitness(refineSource,phase);
                    OffRefine = BCDCA.operatorPBestDE(Problem,refineSource,refineFitness,refineCount,0.12);
                else
                    OffRefine = PopMain(1:0);
                end
                if Problem.M == 2 && phase < 0.92
                    freePool = BCDCA.safePool([PopFree,PopAux,FeasArchive],PopFree);
                    [rankFree,crowdFree] = BCDCA.objectiveMatingRank(freePool);
                    freeCount = ceil((0.30-0.16*phase)*Problem.N);
                    OffFree = BCDCA.generateOffspring(Problem,freePool,rankFree,crowdFree,freeCount,0.18,0.65);
                else
                    OffFree = PopMain(1:0);
                end
                useFrontRoute = Problem.M >= 3 && (mod(gen,3) == 0 || (phase > 0.78 && mod(gen,2) == 0));
                if useFrontRoute
                    frontArchive = BCDCA.objectiveSample(FeasArchive,ceil(0.25*Problem.N));
                    [PopFront,RankFront] = BCDCA.selectConstrainedByRegion([PopFront,frontArchive],Problem.N,alpha);
                    [PopScout,RankScout] = BCDCA.selectRelaxedByRegion([PopScout,frontArchive, ...
                        BCDCA.lowCVSample(PopFront,ceil(0.12*Problem.N))],Problem.N,alpha,gamma,para);
                    frontRatio = 0.30-0.10*phase;
                    frontCount = ceil(frontRatio*Problem.N);
                    OffFront = BCDCA.rankedGAOffspring(Problem,[PopFront,PopScout], ...
                        [RankFront,RankScout],frontCount);
                else
                    OffFront = PopMain(1:0);
                end

                FeasArchive = BCDCA.updateFeasibleArchive([FeasArchive,PopMain,PopAux,PopFront,PopScout,PopFree, ...
                    OffMain,OffAux,OffRefine,OffFront,OffFree],archiveSize,phase);
                ZminMain = min([ZminMain;OffMain.objs],[],1);
                ZminAux = min([ZminAux;OffAux.objs],[],1);
                [PopMain,FitnessMain] = BCDCA.mainTaskSelection([PopMain,OffMain,OffAux,OffRefine,OffFree,FeasArchive], ...
                    Problem.N,true);
                [PopAux,FitnessAux] = BCDCA.auxTaskSelection([PopAux,OffAux,OffMain,OffRefine,OffFree,FeasArchive], ...
                    Problem.N,VAR);
                alpha = BCDCA.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                if useFrontRoute
                    frontMain = BCDCA.objectiveSample(OffMain,ceil(0.16*Problem.N));
                    frontAux = BCDCA.objectiveSample(OffAux,ceil(0.16*Problem.N));
                    frontArchive = BCDCA.objectiveSample(FeasArchive,ceil(0.25*Problem.N));
                    [PopFront,RankFront] = BCDCA.selectConstrainedByRegion([PopFront,OffFront,frontMain,frontAux,frontArchive], ...
                        Problem.N,alpha);
                    [PopScout,RankScout] = BCDCA.selectRelaxedByRegion([PopScout,OffFront,frontMain,frontAux,frontArchive], ...
                        Problem.N,alpha,gamma,para);
                end
                if Problem.M == 2
                    PopFree = BCDCA.selectByObjectiveAndNearestDistance([PopFree,OffFree,BCDCA.objectiveSample(OffMain,ceil(0.12*Problem.N)), ...
                        BCDCA.objectiveSample(OffAux,ceil(0.12*Problem.N)),FeasArchive],Problem.N,phase);
                end
                PopOutput = BCDCA.outputSelection([PopMain,PopAux,PopFront,PopScout,PopFree,FeasArchive],Problem.N,phase);
                X = X+1/(Problem.maxFE/Problem.N);
            end
        end

        function runLow(Algorithm,Problem,PopCon,PopObj,archiveRatio,rhoMin)
            PopObj = PopCon;
            FeasArchive = PopCon(1:0);
            archiveSize = max(8,ceil(archiveRatio*Problem.N));
            alpha = BCDCA.lowDimAlpha(Problem);
            [PopCon,RankCon] = BCDCA.selectConstrainedByRegion(PopCon,Problem.N,alpha);
            [PopObj,RankObj] = BCDCA.selectRelaxedByRegion(PopObj,Problem.N,alpha,1,0);
            PopOutput = PopCon;
            generation = 0;

            while Algorithm.NotTerminated(PopOutput)
                generation = generation+1;
                phase = min(1,Problem.FE/max(Problem.maxFE,1));
                PopCon = PopCon(randperm(numel(PopCon)));
                PopObj = PopObj(randperm(numel(PopObj)));
                alpha = BCDCA.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                gamma = BCDCA.lowDimPopulationGap(PopCon,PopObj);
                [PopCon,RankCon] = BCDCA.selectConstrainedByRegion(PopCon,Problem.N,alpha);
                [PopObj,RankObj] = BCDCA.selectRelaxedByRegion(PopObj,Problem.N,alpha,gamma,para);
                allocation = BCDCA.allocateOffspring(PopCon.cons,Problem.N,rhoMin);
                feBefore = Problem.FE;
                MatingCon = TournamentSelection(2,2*allocation.constraintCount,RankCon);
                OffCon = OperatorGAhalf(Problem,PopCon(MatingCon));
                MatingObj = TournamentSelection(2,2*allocation.relaxCount,RankObj);
                OffObj = OperatorGAhalf(Problem,PopObj(MatingObj));
                Offspring = [OffCon,OffObj];
                Algorithm.recordTrace(generation,allocation,feBefore,Problem.FE);

                FeasArchive = BCDCA.updateLowModeArchive([FeasArchive,PopCon,PopObj,Offspring],archiveSize,phase);
                alpha = BCDCA.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                [PopCon,RankCon] = BCDCA.selectConstrainedByRegion([PopCon,Offspring],Problem.N,alpha);
                [PopObj,RankObj] = BCDCA.selectRelaxedByRegion([PopObj,Offspring],Problem.N,alpha,gamma,para);
                PopOutput = PopCon;
            end
        end

        function Offspring = makeHardMain(~,Problem,Population,Fitness,Zmin,count,phase,refinePool)
            Empty = Population(1:0);
            randCount = floor(count/2);
            restCount = count-randCount;
            if randCount > 0
                Mating = Population(randperm(Problem.N));
                [Mate1,Mate2,Mate3] = BCDCA.neighborPairing(Mating,Zmin);
                OffRand = BCDCA.operatorRandDE(Problem,Mate1(1:randCount), ...
                    Mate2(1:randCount),Mate3(1:randCount));
            else
                OffRand = Empty;
            end
            refineSource = BCDCA.feasibleRefinementSource(refinePool);
            refineCount = BCDCA.refinementBudget(restCount,phase,refineSource);
            pbestCount = restCount-refineCount;
            if pbestCount > 0
                OffPBest = BCDCA.operatorPBestDE(Problem,Population,Fitness,pbestCount,0.10);
            else
                OffPBest = Empty;
            end
            if refineCount > 0
                refineFitness = BCDCA.objectiveRefinementFitness(refineSource,phase);
                OffRefine = BCDCA.operatorPBestDE(Problem,refineSource,refineFitness,refineCount,0.12);
            else
                OffRefine = Empty;
            end
            Offspring = [OffRand,OffPBest,OffRefine];
        end

        function Offspring = makeStandardChannel(~,Problem,Population,Fitness,Zmin,count,p)
            Empty = Population(1:0);
            randCount = floor(count/2);
            restCount = count-randCount;
            if randCount > 0
                Mating = Population(randperm(Problem.N));
                [Mate1,Mate2,Mate3] = BCDCA.neighborPairing(Mating,Zmin);
                OffRand = BCDCA.operatorRandDE(Problem,Mate1(1:randCount), ...
                    Mate2(1:randCount),Mate3(1:randCount));
            else
                OffRand = Empty;
            end
            if restCount > 0
                OffPBest = BCDCA.operatorPBestDE(Problem,Population,Fitness,restCount,p);
            else
                OffPBest = Empty;
            end
            Offspring = [OffRand,OffPBest];
        end

        function recordTrace(Algorithm,generation,allocation,feBefore,feAfter)
            entry = struct( ...
                'generation',generation, ...
                'modeCode',Algorithm.modeCode, ...
                'feasibleRate',allocation.feasibleRate, ...
                'scarcity',allocation.scarcity, ...
                'rhoConstraint',allocation.rhoConstraint, ...
                'budget',allocation.budget, ...
                'constraintCount',allocation.constraintCount, ...
                'relaxCount',allocation.relaxCount, ...
                'feBeforeControlled',feBefore, ...
                'feAfterControlled',feAfter);
            if isempty(Algorithm.auditTrace)
                Algorithm.auditTrace = entry;
            else
                Algorithm.auditTrace(end+1) = entry;
            end
        end
    end

    methods(Static,Access = private)
        function allocation = allocateOffspring(cons,budget,rhoMin)
            if ~isscalar(budget) || ~isfinite(budget) || budget < 2 || budget ~= floor(budget)
                error('BCDCA:InvalidBudget','Budget must be an integer no smaller than two.');
            end

            feasibleRate = mean(sum(max(0,cons),2) <= 0);
            scarcity = 1 - feasibleRate;
            rhoConstraint = rhoMin+(1-2*rhoMin)*scarcity;
            floorCount = ceil(rhoMin*budget);
            roundedBudget = budget*rhoConstraint;
            % Avoid an off-by-one at half-integer budgets due to rounding.
            constraintCount = min(budget-floorCount, ...
                max(floorCount,floor(roundedBudget+0.5+ ...
                eps(max(1,abs(roundedBudget))))));
            allocation = struct( ...
                'feasibleRate',feasibleRate, ...
                'scarcity',scarcity, ...
                'rhoConstraint',rhoConstraint, ...
                'budget',budget, ...
                'constraintCount',constraintCount, ...
                'relaxCount',budget-constraintCount);
        end

        % Offspring variation and feasible refinement.
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

        function Source = feasibleRefinementSource(Population)
            if isempty(Population)
                Source = Population;
                return;
            end
            CV = BCDCA.traceConstraintViolation(Population);
            feasible = CV <= 1e-12;
            if any(feasible)
                Source = Population(feasible);
            else
                Source = Population(1:0);
            end
        end

        function Fitness = objectiveRefinementFitness(Population, phase)
            Obj = BCDCA.normalizeForRefinement(Population.objs);
            FrontNo = NDSort(Obj, inf);
            crowd = BCDCA.secondNearestDecisionDistance(Population.decs);
            radius = sqrt(sum(Obj .^ 2, 2));
            Fitness = FrontNo(:) + ...
                (0.18 + 0.10 * phase) * BCDCA.scaleFiniteScores(radius) - ...
                0.05 * BCDCA.scaleFiniteScores(crowd);
        end

        function count = refinementBudget(maxCount, phase, Source)
            if phase < 0.30 || isempty(Source) || numel(Source) < 6
                count = 0;
                return;
            end
            if phase < 0.65
                count = ceil(0.20 * maxCount);
            else
                count = ceil(0.30 * maxCount);
            end
            count = min(maxCount, max(0, count));
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
            OffDec = BCDCA.polynomialMutation(Problem, OffDec);
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
            OffDec = BCDCA.polynomialMutation(Problem, OffDec);
            Offspring = Problem.Evaluation(OffDec);
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
            OffDec = BCDCA.polynomialMutation(Problem, OffDec);
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

        % Parent sampling and feasible archives.
        function Sample = lowCVSample(Population, K)
            if isempty(Population) || K <= 0
                Sample = Population(1:0);
                return;
            end
            [~, order] = sort(BCDCA.totalConstraintViolation(Population), 'ascend');
            Sample = Population(order(1:min(K, numel(order))));
        end

        function Sample = objectiveSample(Population, K)
            if isempty(Population) || K <= 0
                Sample = Population(1:0);
                return;
            end
            score = sum(BCDCA.normalizeByColumnRange(Population.objs), 2) + ...
                    0.15 * BCDCA.scaleMatingScores(BCDCA.totalConstraintViolation(Population));
            [~, order] = sort(score, 'ascend');
            Sample = Population(order(1:min(K, numel(order))));
        end

        function Pool = safePool(Pool, Fallback)
            if isempty(Pool)
                Pool = Fallback;
            end
        end

        function Population = selectByObjectiveAndNearestDistance(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            Obj = Population.objs;
            [FrontNo, MaxFNo] = NDSort(Obj, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            if sum(Next) < N && ~isempty(Last)
                crowd = CrowdingDistance(Obj(Last, :), FrontNo(Last));
                decCrowd = BCDCA.nearestDecisionDistance(Population(Last).decs);
                mixCrowd = (0.80 + 0.12 * phase) * BCDCA.scaleMatingScores(crowd(:)) + ...
                           (0.20 - 0.12 * phase) * BCDCA.scaleMatingScores(decCrowd(:));
                [~, order] = sort(mixCrowd, 'descend');
                Next(Last(order(1:N - sum(Next)))) = true;
            end
            Population = Population(Next);
        end

        function Archive = updateLowModeArchive(Pool, K, phase)
            if isempty(Pool)
                Archive = Pool;
                return;
            end
            feasible = BCDCA.totalConstraintViolation(Pool) <= 1e-12;
            if ~any(feasible)
                Archive = Pool(1:0);
                return;
            end
            Archive = BCDCA.selectByObjectiveAndNearestDistance(Pool(feasible), min(K, sum(feasible)), phase);
        end

        function Archive = updateFeasibleArchive(Pool, K, phase)
            if isempty(Pool) || K <= 0
                Archive = Pool(1:0);
                return;
            end
            feasible = BCDCA.archiveConstraintViolation(Pool) <= 1e-12;
            if ~any(feasible)
                Archive = Pool(1:0);
                return;
            end
            Feas = Pool(feasible);
            if numel(Feas) <= K
                Archive = Feas;
            else
                Archive = BCDCA.selectByObjectiveAndDecisionCrowding(Feas, K, phase);
            end
        end

        function crowd = normalizedDecisionCrowding(Dec)
            N = size(Dec, 1);
            if N <= 2
                crowd = ones(N, 1);
                return;
            end
            Dec = BCDCA.normalizeWithEpsilonFloor(Dec);
            Distance = pdist2(Dec, Dec);
            Distance(logical(eye(N))) = inf;
            Distance = sort(Distance, 2, 'ascend');
            crowd = Distance(:, max(1, min(N - 1, floor(sqrt(N)))));
            crowd = BCDCA.scaleArchiveScores(crowd);
        end

        function CV = archiveConstraintViolation(Population)
            if isempty(Population) || isempty(Population.cons)
                CV = zeros(numel(Population), 1);
                return;
            end
            CV = sum(max(Population.cons, 0), 2);
        end

        function M = normalizeWithEpsilonFloor(M)
            if isempty(M)
                return;
            end
            minV = min(M, [], 1);
            maxV = max(M, [], 1);
            span = max(maxV - minV, eps);
            M = (M - minV) ./ span;
        end

        function y = scaleArchiveScores(x)
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

        function Population = selectByObjectiveAndDecisionCrowding(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            Obj = Population.objs;
            [FrontNo, MaxFNo] = NDSort(Obj, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            if sum(Next) < N && ~isempty(Last)
                crowd = CrowdingDistance(Obj(Last, :), FrontNo(Last));
                decCrowd = BCDCA.normalizedDecisionCrowding(Population(Last).decs);
                mixCrowd = (0.78 + 0.12 * phase) * BCDCA.scaleArchiveScores(crowd(:)) + ...
                           (0.22 - 0.12 * phase) * BCDCA.scaleArchiveScores(decCrowd(:));
                [~, order] = sort(mixCrowd, 'descend');
                Next(Last(order(1:N - sum(Next)))) = true;
            end
            Population = Population(Next);
        end

        % Ranking and environmental selection.
        function alpha = lowDimAlpha(Problem)
            alpha = 2 ./ (1 + exp(-Problem.FE * 10 / max(Problem.maxFE, 1))) - 1;
            alpha = min(1, max(0, alpha));
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
                Dist(bad, :) = pdist2(BCDCA.normalizeByColumnRange(PopObj(bad, :)), W, 'euclidean');
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

        function rank = lowDimConvergenceRank(PopObj, frontNo, crowd, N)
            middle = zeros(numel(frontNo), 1);
            firstFront = frontNo(:) == 1;
            if sum(firstFront) > N
                del = BCDCA.lowDimTruncation(PopObj(firstFront, :), sum(firstFront) - N);
                temp = find(firstFront);
                middle(temp(del)) = 1;
                [~, order] = sortrows([frontNo(:), middle(:), -crowd(:)]);
            else
                [~, order] = sortrows([frontNo(:), -crowd(:)]);
            end
            rank = zeros(1, numel(frontNo));
            rank(order) = 1:numel(frontNo);
        end

        function [Population, RankSolution] = selectConstrainedByRegion(Population, N, alpha)
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
            PopCon = BCDCA.consMatrix(Population);
            CV = sum(max(0, PopCon), 2);
            z = min(PopObj, [], 1);
            n = max(PopObj, [], 1);
            M = size(PopObj, 2);
            [W, ~] = UniformPoint(N, M);
            Region = BCDCA.lowDimAssignRegion(PopObj, z, W);

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

            FrontNo = BCDCA.lowDimConstrainedFront(PopObj, CV);
            crowd = BCDCA.lowDimKthCrowding(PopObj);
            rankConvg = BCDCA.lowDimConvergenceRank(PopObj, FrontNo, crowd, N);

            frontD = BCDCA.lowDimRegionRanks(PopObj2, z, W, Region);
            frontD = frontD + infeasible * popSize;
            rankDivs = BCDCA.rankByFrontCrowd(frontD, crowd);

            rankAll = alpha * rankConvg + (1 - alpha) * rankDivs;
            [~, order] = sort(rankAll, 'ascend');
            Population = Population(order(1:N));
            RankSolution = 1:N;
        end

        function [Population, RankSolution] = selectRelaxedByRegion(Population, N, alpha, gamma, para)
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
            PopCon = BCDCA.consMatrix(Population);
            CV = sum(max(0, PopCon), 2);
            z = min(PopObj, [], 1);
            M = size(PopObj, 2);
            [W, ~] = UniformPoint(N, M);
            Region = BCDCA.lowDimAssignRegion(PopObj, z, W);
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

            FrontNo = BCDCA.lowDimParetoFront(PopObj2);
            crowd = BCDCA.lowDimKthCrowding(PopObj);
            rankConvg = BCDCA.lowDimConvergenceRank(PopObj, FrontNo, crowd, N);

            frontD = BCDCA.lowDimRegionRanks(PopObj2, z, W, Region);
            rankDivs = BCDCA.rankByFrontCrowd(frontD, crowd);

            rankAll = alpha * rankConvg + (1 - alpha) * rankDivs;
            [~, order] = sort(rankAll, 'ascend');
            Population = Population(order(1:N));
            RankSolution = 1:N;
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
            crowd = BCDCA.scaleMatingScores(objCrowd(:)) + ...
                0.10 * BCDCA.scaleMatingScores(BCDCA.nearestDecisionDistance(Population.decs));
        end

        function rank = rankByFrontCrowd(frontNo, crowd)
            [~, order] = sortrows([frontNo(:), -crowd(:)]);
            rank = zeros(1, numel(frontNo));
            rank(order) = 1:numel(frontNo);
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
                InfFitness = BCDCA.calculateFitness(InfPop.objs, InfPop.cons);
                Next = InfFitness < 1;
                if sum(Next) <= N
                    [~, rank] = sort(InfFitness);
                    Next(rank(1:N)) = true;
                else
                    Del = BCDCA.truncation(InfPop(Next).objs, sum(Next) - N);
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
                FeasFitness = BCDCA.calculateFitness([FeasPop.objs, cons]);
                Next = FeasFitness < 1;
                [~, rank] = sort(FeasFitness);
                Next(rank(1:numel(FeasPop))) = true;
                FeasPop = FeasPop(Next);
                FeasFitness = FeasFitness(Next);
                [FeasFitness, rank] = sort(FeasFitness);
                FeasPop = FeasPop(rank);

                InfFitness = BCDCA.calculateFitness(InfPop.objs, InfPop.cons);
                NextInf = InfFitness < 1;
                slots = N - numel(FeasPop);
                if slots > 0
                    if sum(NextInf) <= slots
                        [~, rank] = sort(InfFitness);
                        NextInf(rank(1:min(slots, numel(rank)))) = true;
                    else
                        Del = BCDCA.truncation(InfPop(NextInf).objs, sum(NextInf) - slots);
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
            FeasFitness = BCDCA.calculateFitness([FeasPop.objs, cons]);
            Next = FeasFitness < 1;
            if sum(Next) <= N
                [~, rank] = sort(FeasFitness);
                Next(rank(1:N)) = true;
            else
                Del = BCDCA.truncation(FeasPop(Next).objs, sum(Next) - N);
                temp = find(Next);
                Next(temp(Del)) = false;
            end
            Population = FeasPop(Next);
            Fitness = FeasFitness(Next);
            [Fitness, rank] = sort(Fitness);
            Population = Population(rank);
        end

        function Fitness = calculateFitness(PopObj, PopCon)
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

        function [Population, Fitness] = mainTaskSelection(Population, N, isOrigin)
            if isOrigin
                Fitness = BCDCA.calculateFitness(Population.objs, Population.cons);
            else
                Fitness = BCDCA.calculateFitness(Population.objs);
            end
            Next = Fitness < 1;
            if sum(Next) < N
                [~, rank] = sort(Fitness);
                Next(rank(1:N)) = true;
            elseif sum(Next) > N
                Del = BCDCA.truncation(Population(Next).objs, sum(Next) - N);
                temp = find(Next);
                Next(temp(Del)) = false;
            end
            Population = Population(Next);
            Fitness = Fitness(Next);
            [Fitness, rank] = sort(Fitness);
            Population = Population(rank);
        end

        function Population = manifoldSelection(Population, N, phase)
            if isempty(Population) || numel(Population) <= N
                return;
            end
            CV = BCDCA.traceConstraintViolation(Population);
            [~, order] = sort(CV, 'ascend');
            eliteCount = min(numel(Population), ceil(0.68 * N));
            Next = false(1, numel(Population));
            Next(order(1:eliteCount)) = true;

            slots = N - sum(Next);
            if slots > 0
                cand = order(eliteCount+1:end);
                objScore = sum(BCDCA.normalizeForRefinement(Population(cand).objs), 2);
                decCrowd = BCDCA.secondNearestDecisionDistance(Population(cand).decs);
                score = (0.86 + 0.08 * phase) * BCDCA.scaleFiniteScores(CV(cand)) + ...
                    0.10 * BCDCA.scaleFiniteScores(objScore) - ...
                    0.04 * BCDCA.scaleFiniteScores(decCrowd);
                [~, rank] = sort(score, 'ascend');
                Next(cand(rank(1:min(slots, numel(rank))))) = true;
            end

            Population = Population(Next);
            if numel(Population) > N
                Fitness = BCDCA.violationFitness(Population);
                [~, order] = sort(Fitness, 'ascend');
                Population = Population(order(1:N));
            end
        end

        function Population = outputSelection(Population, N, phase)
            if isempty(Population)
                return;
            end
            CV = BCDCA.traceConstraintViolation(Population);
            feasible = CV <= 1e-12;
            if sum(feasible) >= N
                [Population, ~] = BCDCA.mainTaskSelection(Population(feasible), N, true);
                return;
            end
            if any(feasible)
                FeasPop = Population(feasible);
                if numel(FeasPop) > N
                    [FeasPop, ~] = BCDCA.mainTaskSelection(FeasPop, N, true);
                    Population = FeasPop;
                    return;
                end
                InfPop = Population(~feasible);
                slots = N - numel(FeasPop);
                if slots > 0 && ~isempty(InfPop)
                    InfPop = BCDCA.manifoldSelection(InfPop, slots, phase);
                else
                    InfPop = InfPop(1:0);
                end
                Population = [FeasPop, InfPop];
                return;
            end
            Population = BCDCA.manifoldSelection(Population, N, phase);
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

        function Fitness = violationFitness(Population)
            CV = BCDCA.traceConstraintViolation(Population);
            objScore = sum(BCDCA.normalizeForRefinement(Population.objs), 2);
            decCrowd = BCDCA.secondNearestDecisionDistance(Population.decs);
            Fitness = 0.90 * BCDCA.scaleFiniteScores(CV) + ...
                0.08 * BCDCA.scaleFiniteScores(objScore) - ...
                0.02 * BCDCA.scaleFiniteScores(decCrowd);
        end

        % Mode dispatch and numeric helpers.
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

        function DecCrowd = nearestDecisionDistance(Dec)
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

        function rate = feasibleRate(Population)
            if isempty(Population)
                rate = 0;
            else
                rate = mean(BCDCA.totalConstraintViolation(Population) <= 1e-12);
            end
        end

        function tf = isHardDecisionMode(Population, Problem)
            CV = BCDCA.totalConstraintViolation(Population);
            consCount = size(BCDCA.consMatrix(Population), 2);
            initFeasRate = mean(CV <= 1e-12);
            initMedianCV = median(CV);
            tf = Problem.D >= 30 && ...
                (consCount > Problem.M || ...
                (consCount >= Problem.M && initFeasRate < 0.20 && initMedianCV > 50));
        end

        function Obj = normalizeByColumnRange(Obj)
            if isempty(Obj)
                return;
            end
            lo = min(Obj, [], 1);
            hi = max(Obj, [], 1);
            span = hi - lo;
            span(span < 1e-12) = 1;
            Obj = (Obj - lo) ./ span;
        end

        function y = scaleMatingScores(x)
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

        function CV = totalConstraintViolation(Population)
            C = BCDCA.consMatrix(Population);
            if isempty(C)
                CV = zeros(numel(Population), 1);
            else
                CV = sum(max(C, 0), 2);
            end
        end

        function Crowd = secondNearestDecisionDistance(PopDec)
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

        function X = normalizeForRefinement(X)
            if isempty(X)
                return;
            end
            xmin = min(X, [], 1);
            xmax = max(X, [], 1);
            span = xmax - xmin;
            span(span <= eps) = 1;
            X = (X - xmin) ./ span;
        end

        function y = scaleFiniteScores(x)
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

        function CV = traceConstraintViolation(Population)
            CV = Population.cons;
            CV(CV < 0) = 0;
            CV = sum(CV, 2);
        end
    end
end
