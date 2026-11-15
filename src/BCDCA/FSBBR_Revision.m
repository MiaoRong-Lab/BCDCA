classdef FSBBR_Revision < ALGORITHM
% Feasibility-scarcity and sample-bottleneck guided dual-channel search.
    properties(SetAccess = private)
        auditTrace = struct([])
        modeCode = ''
        variantCode = NaN
    end

    methods
        function main(Algorithm,Problem)
            [archiveRatio,variantCode,rhoMin] = Algorithm.ParameterSet(0.30,4,0.25);
            archiveRatio = min(0.70,max(0.10,archiveRatio));
            FSBBR_Revision.validateParameters(variantCode,rhoMin);
            Algorithm.variantCode = variantCode;

            PopHard = Problem.Initialization();
            PopActive = Problem.Initialization();
            PopBridge = Problem.Initialization();
            if BRACE.isHardDecisionMode([PopHard,PopActive,PopBridge],Problem)
                Algorithm.modeCode = 'hard';
            elseif Problem.D >= 30
                Algorithm.modeCode = 'soft';
            else
                Algorithm.modeCode = 'low';
            end

            if variantCode == 0
                switch Algorithm.modeCode
                    case 'hard'
                        BRACE.hardDecisionMode(Algorithm,Problem,PopHard,PopActive,PopBridge,archiveRatio);
                    case 'soft'
                        BRACE.softHighDimMode(Algorithm,Problem,PopHard,PopActive,archiveRatio);
                    otherwise
                        BRACE.lowDimCompatibilityMode(Algorithm,Problem,PopHard,PopActive,archiveRatio);
                end
                return;
            end

            switch Algorithm.modeCode
                case 'hard'
                    Algorithm.runHard(Problem,PopHard,PopActive,PopBridge, ...
                        archiveRatio,rhoMin,variantCode);
                case 'soft'
                    Algorithm.runSoft(Problem,PopHard,PopActive, ...
                        archiveRatio,rhoMin,variantCode);
                otherwise
                    Algorithm.runLow(Problem,PopHard,PopActive, ...
                        archiveRatio,rhoMin,variantCode);
            end
        end
    end

    methods(Access = private)
        function runHard(Algorithm,Problem,PopMain,PopAux,PopTrace,archiveRatio,rhoMin,variantCode)
            FeasArchive = PopMain(1:0);
            archiveSize = max(8,ceil(archiveRatio*Problem.N));
            FitnessMain = TIDE.calFitness(PopMain.objs,PopMain.cons);
            FitnessAux = TIDE.calFitness(PopAux.objs,PopAux.cons);
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
            PopOutput = TIDE.outputSelection([PopMain,PopAux,PopTrace,FeasArchive],Problem.N,0);
            generation = 0;

            while Algorithm.NotTerminated(PopOutput)
                generation = generation+1;
                phase = min(1,Problem.FE/max(Problem.maxFE,1));
                VAR = VAR0*max(0,1-X)^cp;
                state = FSBBR_Controller.measure(PopMain.decs,PopMain.cons,Problem.lower,Problem.upper);
                allocation = FSBBR_Controller.allocate( ...
                    state.scarcity,state.bottleneck,2*Problem.N,rhoMin,variantCode);
                feBefore = Problem.FE;
                OffMain = Algorithm.makeHardMain(Problem,PopMain,FitnessMain,ZminMain, ...
                    allocation.constraintCount,phase,[FeasArchive,PopMain,PopAux,PopTrace]);
                OffAux = Algorithm.makeStandardChannel(Problem,PopAux,FitnessAux,ZminAux, ...
                    allocation.relaxCount,0.10);
                Algorithm.recordTrace(generation,state,allocation,feBefore,Problem.FE);

                tracePool = [PopTrace,PopMain,PopAux,FeasArchive];
                traceFitness = TIDE.violationFitness(tracePool);
                traceFeasRate = BRACE.feasibleRate(tracePool);
                if phase > 0.25 && traceFeasRate < 0.80
                    traceCount = ceil((0.35+0.35*max(0,0.80-traceFeasRate))*Problem.N);
                else
                    traceCount = ceil(0.12*Problem.N);
                end
                polishCount = 0;
                if phase > 0.45 && traceFeasRate < 0.35
                    bestTraceCV = min(TIDE.totalCV(tracePool));
                    if bestTraceCV < 0.10
                        polishCount = ceil(0.30*Problem.N);
                    else
                        polishCount = ceil(0.16*Problem.N);
                    end
                end
                OffTrace = TIDE.operatorCVPBestDE(Problem,tracePool,traceFitness,traceCount,0.12,phase);
                OffPolish = TIDE.operatorCVLocalSearch(Problem,tracePool,traceFitness,polishCount,phase);

                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive,PopMain,PopAux,PopTrace, ...
                    OffMain,OffAux,OffTrace,OffPolish],archiveSize,phase);
                ZminMain = min([ZminMain;OffMain.objs],[],1);
                ZminAux = min([ZminAux;OffAux.objs],[],1);
                PopTrace = TIDE.manifoldSelection([PopTrace,OffTrace,OffPolish,PopMain,PopAux, ...
                    OffMain,OffAux,FeasArchive],Problem.N,phase);
                [PopMain,FitnessMain] = TIDE.mainTaskSelection([PopMain,OffMain,OffAux, ...
                    OffTrace,OffPolish,PopTrace,FeasArchive],Problem.N,true);
                [PopAux,FitnessAux] = TIDE.auxTaskSelection([PopAux,OffAux,OffMain, ...
                    OffTrace,OffPolish,PopTrace,FeasArchive],Problem.N,VAR);
                PopOutput = TIDE.outputSelection([PopMain,PopAux,PopTrace,FeasArchive],Problem.N,phase);
                X = X+1/(Problem.maxFE/Problem.N);
            end
        end

        function runSoft(Algorithm,Problem,PopMain,PopAux,archiveRatio,rhoMin,variantCode)
            PopFront = PopMain;
            PopScout = PopAux;
            PopFree = PopAux;
            FeasArchive = PopMain(1:0);
            archiveSize = max(8,ceil(archiveRatio*Problem.N));
            FitnessMain = TIDE.calFitness(PopMain.objs,PopMain.cons);
            FitnessAux = TIDE.calFitness(PopAux.objs,PopAux.cons);
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
            alpha = BRACE.lowDimAlpha(Problem);
            [PopFront,RankFront] = BRACE.lowDimDPEAConSelection(PopFront,Problem.N,alpha);
            [PopScout,RankScout] = BRACE.lowDimDPEAObjSelection(PopScout,Problem.N,alpha,1,0);
            PopOutput = TIDE.outputSelection([PopMain,PopAux,PopFront,PopFree,FeasArchive],Problem.N,0);
            generation = 0;

            while Algorithm.NotTerminated(PopOutput)
                generation = generation+1;
                phase = min(1,Problem.FE/max(Problem.maxFE,1));
                VAR = VAR0*max(0,1-X)^cp;
                gen = max(1,ceil(Problem.FE/Problem.N));
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                gamma = BRACE.lowDimPopulationGap(PopFront,PopScout);
                state = FSBBR_Controller.measure(PopMain.decs,PopMain.cons,Problem.lower,Problem.upper);
                allocation = FSBBR_Controller.allocate( ...
                    state.scarcity,state.bottleneck,2*Problem.N,rhoMin,variantCode);
                feBefore = Problem.FE;
                OffMain = Algorithm.makeStandardChannel(Problem,PopMain,FitnessMain,ZminMain, ...
                    allocation.constraintCount,0.10);
                OffAux = Algorithm.makeStandardChannel(Problem,PopAux,FitnessAux,ZminAux, ...
                    allocation.relaxCount,0.10);
                Algorithm.recordTrace(generation,state,allocation,feBefore,Problem.FE);

                refineSource = SIFT.feasibleRefinementSource([FeasArchive,PopMain,PopAux]);
                if phase > 0.38 && ~isempty(refineSource) && numel(refineSource) >= 6
                    refineCount = ceil(0.22*Problem.N);
                    refineFitness = SIFT.objectiveRefinementFitness(refineSource,phase);
                    OffRefine = TIDE.operatorPBestDE(Problem,refineSource,refineFitness,refineCount,0.12);
                else
                    OffRefine = PopMain(1:0);
                end
                if Problem.M == 2 && phase < 0.92
                    freePool = BRACE.safePool([PopFree,PopAux,FeasArchive],PopFree);
                    [rankFree,crowdFree] = BRACE.objectiveMatingRank(freePool);
                    freeCount = ceil((0.30-0.16*phase)*Problem.N);
                    OffFree = BRACE.generateOffspring(Problem,freePool,rankFree,crowdFree,freeCount,0.18,0.65);
                else
                    OffFree = PopMain(1:0);
                end
                useFrontRoute = Problem.M >= 3 && (mod(gen,3) == 0 || (phase > 0.78 && mod(gen,2) == 0));
                if useFrontRoute
                    frontArchive = BRACE.objectiveSample(FeasArchive,ceil(0.25*Problem.N));
                    [PopFront,RankFront] = BRACE.lowDimDPEAConSelection([PopFront,frontArchive],Problem.N,alpha);
                    [PopScout,RankScout] = BRACE.lowDimDPEAObjSelection([PopScout,frontArchive, ...
                        BRACE.lowCVSample(PopFront,ceil(0.12*Problem.N))],Problem.N,alpha,gamma,para);
                    frontRatio = 0.30-0.10*phase;
                    frontCount = ceil(frontRatio*Problem.N);
                    OffFront = BRACE.rankedGAOffspring(Problem,[PopFront,PopScout], ...
                        [RankFront,RankScout],frontCount);
                else
                    OffFront = PopMain(1:0);
                end

                FeasArchive = DRIFT.updateFeasibleArchive([FeasArchive,PopMain,PopAux,PopFront,PopScout,PopFree, ...
                    OffMain,OffAux,OffRefine,OffFront,OffFree],archiveSize,phase);
                ZminMain = min([ZminMain;OffMain.objs],[],1);
                ZminAux = min([ZminAux;OffAux.objs],[],1);
                [PopMain,FitnessMain] = TIDE.mainTaskSelection([PopMain,OffMain,OffAux,OffRefine,OffFree,FeasArchive], ...
                    Problem.N,true);
                [PopAux,FitnessAux] = TIDE.auxTaskSelection([PopAux,OffAux,OffMain,OffRefine,OffFree,FeasArchive], ...
                    Problem.N,VAR);
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                if useFrontRoute
                    frontMain = BRACE.objectiveSample(OffMain,ceil(0.16*Problem.N));
                    frontAux = BRACE.objectiveSample(OffAux,ceil(0.16*Problem.N));
                    frontArchive = BRACE.objectiveSample(FeasArchive,ceil(0.25*Problem.N));
                    [PopFront,RankFront] = BRACE.lowDimDPEAConSelection([PopFront,OffFront,frontMain,frontAux,frontArchive], ...
                        Problem.N,alpha);
                    [PopScout,RankScout] = BRACE.lowDimDPEAObjSelection([PopScout,OffFront,frontMain,frontAux,frontArchive], ...
                        Problem.N,alpha,gamma,para);
                end
                if Problem.M == 2
                    PopFree = BRACE.selectByObjective([PopFree,OffFree,BRACE.objectiveSample(OffMain,ceil(0.12*Problem.N)), ...
                        BRACE.objectiveSample(OffAux,ceil(0.12*Problem.N)),FeasArchive],Problem.N,phase);
                end
                PopOutput = TIDE.outputSelection([PopMain,PopAux,PopFront,PopScout,PopFree,FeasArchive],Problem.N,phase);
                X = X+1/(Problem.maxFE/Problem.N);
            end
        end

        function runLow(Algorithm,Problem,PopCon,PopObj,archiveRatio,rhoMin,variantCode)
            PopObj = PopCon;
            FeasArchive = PopCon(1:0);
            archiveSize = max(8,ceil(archiveRatio*Problem.N));
            alpha = BRACE.lowDimAlpha(Problem);
            [PopCon,RankCon] = BRACE.lowDimDPEAConSelection(PopCon,Problem.N,alpha);
            [PopObj,RankObj] = BRACE.lowDimDPEAObjSelection(PopObj,Problem.N,alpha,1,0);
            PopOutput = PopCon;
            generation = 0;

            while Algorithm.NotTerminated(PopOutput)
                generation = generation+1;
                phase = min(1,Problem.FE/max(Problem.maxFE,1));
                PopCon = PopCon(randperm(numel(PopCon)));
                PopObj = PopObj(randperm(numel(PopObj)));
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                gamma = BRACE.lowDimPopulationGap(PopCon,PopObj);
                [PopCon,RankCon] = BRACE.lowDimDPEAConSelection(PopCon,Problem.N,alpha);
                [PopObj,RankObj] = BRACE.lowDimDPEAObjSelection(PopObj,Problem.N,alpha,gamma,para);
                state = FSBBR_Controller.measure(PopCon.decs,PopCon.cons,Problem.lower,Problem.upper);
                allocation = FSBBR_Controller.allocate( ...
                    state.scarcity,state.bottleneck,Problem.N,rhoMin,variantCode);
                feBefore = Problem.FE;
                MatingCon = TournamentSelection(2,2*allocation.constraintCount,RankCon);
                OffCon = OperatorGAhalf(Problem,PopCon(MatingCon));
                MatingObj = TournamentSelection(2,2*allocation.relaxCount,RankObj);
                OffObj = OperatorGAhalf(Problem,PopObj(MatingObj));
                Offspring = [OffCon,OffObj];
                Algorithm.recordTrace(generation,state,allocation,feBefore,Problem.FE);

                FeasArchive = BRACE.updateFeasibleArchive([FeasArchive,PopCon,PopObj,Offspring],archiveSize,phase);
                alpha = BRACE.lowDimAlpha(Problem);
                para = ceil(Problem.maxFE/Problem.N)/2-ceil(Problem.FE/Problem.N);
                [PopCon,RankCon] = BRACE.lowDimDPEAConSelection([PopCon,Offspring],Problem.N,alpha);
                [PopObj,RankObj] = BRACE.lowDimDPEAObjSelection([PopObj,Offspring],Problem.N,alpha,gamma,para);
                PopOutput = PopCon;
            end
        end

        function Offspring = makeHardMain(~,Problem,Population,Fitness,Zmin,count,phase,refinePool)
            Empty = Population(1:0);
            randCount = floor(count/2);
            restCount = count-randCount;
            if randCount > 0
                Mating = Population(randperm(Problem.N));
                [Mate1,Mate2,Mate3] = TIDE.neighborPairing(Mating,Zmin);
                OffRand = TIDE.operatorRandDE(Problem,Mate1(1:randCount), ...
                    Mate2(1:randCount),Mate3(1:randCount));
            else
                OffRand = Empty;
            end
            refineSource = SIFT.feasibleRefinementSource(refinePool);
            refineCount = SIFT.refinementBudget(restCount,phase,refineSource);
            pbestCount = restCount-refineCount;
            if pbestCount > 0
                OffPBest = TIDE.operatorPBestDE(Problem,Population,Fitness,pbestCount,0.10);
            else
                OffPBest = Empty;
            end
            if refineCount > 0
                refineFitness = SIFT.objectiveRefinementFitness(refineSource,phase);
                OffRefine = TIDE.operatorPBestDE(Problem,refineSource,refineFitness,refineCount,0.12);
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
                [Mate1,Mate2,Mate3] = TIDE.neighborPairing(Mating,Zmin);
                OffRand = TIDE.operatorRandDE(Problem,Mate1(1:randCount), ...
                    Mate2(1:randCount),Mate3(1:randCount));
            else
                OffRand = Empty;
            end
            if restCount > 0
                OffPBest = TIDE.operatorPBestDE(Problem,Population,Fitness,restCount,p);
            else
                OffPBest = Empty;
            end
            Offspring = [OffRand,OffPBest];
        end

        function recordTrace(Algorithm,generation,state,allocation,feBefore,feAfter)
            entry = struct( ...
                'generation',generation, ...
                'modeCode',Algorithm.modeCode, ...
                'variantCode',Algorithm.variantCode, ...
                'feasibleRate',state.feasibleRate, ...
                'scarcity',state.scarcity, ...
                'bottleneck',state.bottleneck, ...
                'effectivePointCount',state.effectivePointCount, ...
                'validDimensionCount',state.validDimensionCount, ...
                'degenerate',state.degenerate, ...
                'degenerateReason',state.degenerateReason, ...
                'q',allocation.q, ...
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
        function validateParameters(variantCode,rhoMin)
            if ~isscalar(variantCode) || ~isfinite(variantCode) || ...
                    variantCode ~= floor(variantCode) || ~ismember(variantCode,0:4)
                error('FSBBR:InvalidVariant', ...
                    'variantCode must be an integer from zero through four.');
            end
            if ~isscalar(rhoMin) || ~isfinite(rhoMin) || rhoMin < 0 || rhoMin >= 0.5
                error('FSBBR:InvalidRhoMin', ...
                    'rhoMin must be a finite scalar in [0,0.5).');
            end
        end
    end
end
