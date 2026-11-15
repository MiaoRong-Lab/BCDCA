classdef SIFT < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Feasible-region refinement routines used by BCDCA
% archiveRatio --- 0.30 --- Feasible output archive ratio

    methods
        function main(Algorithm, Problem)
            archiveRatio = Algorithm.ParameterSet(0.30);
            archiveRatio = min(0.80, max(0.10, archiveRatio));

            PopMain = Problem.Initialization();
            PopAux  = Problem.Initialization();
            FeasArchive = PopMain(1:0);
            archiveSize = max(8, ceil(archiveRatio * Problem.N));

            FitnessMain = TIDE.calFitness(PopMain.objs, PopMain.cons);
            FitnessAux  = TIDE.calFitness(PopAux.objs,  PopAux.cons);
            ZminMain = min(PopMain.objs, [], 1);
            ZminAux  = min(PopAux.objs,  [], 1);

            initCons = [PopMain.cons; PopAux.cons];
            initCons(initCons < 0) = 0;
            initCV = sum(initCons, 2);
            VAR0 = max(initCV);
            if isempty(VAR0) || VAR0 <= 0
                VAR0 = 1;
            end
            cp = (-log(VAR0) - 6) / log(1 - 0.5);
            X = 0;
            PopOutput = TIDE.outputSelection([PopMain, PopAux, FeasArchive], Problem.N, 0);

            while Algorithm.NotTerminated(PopOutput)
                phase = min(1, Problem.FE / max(Problem.maxFE, 1));
                VAR = VAR0 * max(0, 1 - X) ^ cp;
                halfN = floor(Problem.N / 2);
                restN = Problem.N - halfN;

                MatingMain = PopMain(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingMain, ZminMain);
                OffMain(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));
                refineSource = SIFT.feasibleRefinementSource([FeasArchive, PopMain, PopAux]);
                refineN = SIFT.refinementBudget(restN, phase, refineSource);
                pbestN = restN - refineN;
                if pbestN > 0
                    OffMain(halfN+1:halfN+pbestN) = TIDE.operatorPBestDE(Problem, PopMain, FitnessMain, pbestN, 0.10);
                end
                if refineN > 0
                    refineFitness = SIFT.objectiveRefinementFitness(refineSource, phase);
                    OffMain(halfN+pbestN+1:Problem.N) = TIDE.operatorPBestDE(Problem, ...
                        refineSource, refineFitness, refineN, 0.12);
                end

                MatingAux = PopAux(randperm(Problem.N));
                [Mate1, Mate2, Mate3] = TIDE.neighborPairing(MatingAux, ZminAux);
                OffAux(1:halfN) = TIDE.operatorRandDE(Problem, Mate1(1:halfN), Mate2(1:halfN), Mate3(1:halfN));
                OffAux(halfN+1:Problem.N) = TIDE.operatorPBestDE(Problem, PopAux, FitnessAux, restN, 0.10);

                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive, PopMain, PopAux, OffMain, OffAux], ...
                    archiveSize, phase);
                ZminMain = min([ZminMain; OffMain.objs], [], 1);
                ZminAux  = min([ZminAux;  OffAux.objs],  [], 1);

                [PopMain, FitnessMain] = TIDE.mainTaskSelection([PopMain, OffMain, OffAux, FeasArchive], Problem.N, true);
                [PopAux,  FitnessAux]  = TIDE.auxTaskSelection([PopAux, OffAux, OffMain, FeasArchive], Problem.N, VAR);
                PopOutput = TIDE.outputSelection([PopMain, PopAux, FeasArchive], Problem.N, phase);
                X = X + 1 / (Problem.maxFE / Problem.N);
            end
        end
    end

    methods(Static)
        function Source = feasibleRefinementSource(Population)
            if isempty(Population)
                Source = Population;
                return;
            end
            CV = TIDE.totalCV(Population);
            feasible = CV <= 1e-12;
            if any(feasible)
                Source = Population(feasible);
            else
                Source = Population(1:0);
            end
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

        function Fitness = objectiveRefinementFitness(Population, phase)
            Obj = TIDE.normalizeColumns(Population.objs);
            FrontNo = NDSort(Obj, inf);
            crowd = TIDE.decisionCrowding(Population.decs);
            radius = sqrt(sum(Obj .^ 2, 2));
            Fitness = FrontNo(:) + ...
                (0.18 + 0.10 * phase) * TIDE.scaleToUnit(radius) - ...
                0.05 * TIDE.scaleToUnit(crowd);
        end

    end
end
