classdef FSBBR_Controller
% Stateless feasibility-scarcity and sample-bottleneck budget controller.
    methods(Static)
        function state = measure(decs,cons,lower,upper)
            if isempty(decs) || size(decs,1) == 0
                error('FSBBR:EmptyPopulation','Decision matrix must contain at least one row.');
            end

            lower = reshape(lower,1,[]);
            upper = reshape(upper,1,[]);
            variableCount = numel(lower);
            if numel(upper) ~= variableCount || size(decs,2) ~= variableCount || ...
                    size(cons,1) ~= size(decs,1)
                error('FSBBR:DimensionMismatch', ...
                    'Decisions, constraints, and bounds must have compatible dimensions.');
            end

            cv = sum(max(0,cons),2);
            feasibleRate = mean(cv <= 0);
            validDimensions = isfinite(lower) & isfinite(upper) & upper > lower;
            state = FSBBR_Controller.baseState( ...
                feasibleRate,size(decs,1),sum(validDimensions));

            if ~any(validDimensions)
                state = FSBBR_Controller.degenerate(state,'no-valid-dimension');
                return;
            end

            decs = decs(:,validDimensions);
            if any(~isfinite(decs(:)))
                state = FSBBR_Controller.degenerate(state,'nonfinite-decision');
                return;
            end

            scale = upper(validDimensions)-lower(validDimensions);
            normalized = (decs-lower(validDimensions))./scale;
            points = unique(normalized,'rows','stable');
            state.effectivePointCount = size(points,1);
            if state.effectivePointCount < 3
                state = FSBBR_Controller.degenerate(state,'fewer-than-three-points');
                return;
            end

            edgeLengths = FSBBR_Controller.primEdgeLengths(points);
            totalLength = sum(edgeLengths);
            if totalLength <= 0 || ~isfinite(totalLength)
                state = FSBBR_Controller.degenerate(state,'zero-tree-length');
                return;
            end

            pointCount = numel(edgeLengths)+1;
            state.bottleneck = ((pointCount-1)*max(edgeLengths)/totalLength-1)/(pointCount-2);
            state.bottleneck = min(1,max(0,state.bottleneck));
        end

        function allocation = allocate(scarcity,bottleneck,budget,rhoMin,variantCode)
            if ~FSBBR_Controller.isUnitScalar(scarcity) || ...
                    ~FSBBR_Controller.isUnitScalar(bottleneck)
                error('FSBBR:InvalidState','State values must be finite scalars in [0,1].');
            end
            if ~isscalar(budget) || ~isfinite(budget) || budget < 2 || budget ~= floor(budget)
                error('FSBBR:InvalidBudget','Budget must be an integer no smaller than two.');
            end
            if ~isscalar(rhoMin) || ~isfinite(rhoMin) || rhoMin < 0 || rhoMin >= 0.5
                error('FSBBR:InvalidRhoMin','rhoMin must be a finite scalar in [0,0.5).');
            end
            if ~isscalar(variantCode) || ~ismember(variantCode,1:4)
                error('FSBBR:InvalidVariant','variantCode must be an integer from one through four.');
            end

            switch variantCode
                case 1
                    q = 0.5;
                case 2
                    q = scarcity;
                case 3
                    q = 1-bottleneck;
                case 4
                    q = (scarcity+1-bottleneck)/2;
            end
            rhoConstraint = rhoMin+(1-2*rhoMin)*q;
            floorCount = ceil(rhoMin*budget);
            roundedBudget = budget*rhoConstraint;
            constraintCount = min(budget-floorCount, ...
                max(floorCount,floor(roundedBudget+0.5+ ...
                eps(max(1,abs(roundedBudget))))));
            allocation = struct( ...
                'variantCode',variantCode, ...
                'q',q, ...
                'rhoConstraint',rhoConstraint, ...
                'budget',budget, ...
                'floorCount',floorCount, ...
                'constraintCount',constraintCount, ...
                'relaxCount',budget-constraintCount);
        end
    end

    methods(Static,Access = private)
        function state = baseState(feasibleRate,populationSize,validDimensionCount)
            state = struct( ...
                'feasibleRate',feasibleRate, ...
                'scarcity',1-feasibleRate, ...
                'bottleneck',1, ...
                'populationSize',populationSize, ...
                'effectivePointCount',0, ...
                'validDimensionCount',validDimensionCount, ...
                'degenerate',false, ...
                'degenerateReason','');
        end

        function state = degenerate(state,reason)
            state.bottleneck = 1;
            state.degenerate = true;
            state.degenerateReason = reason;
        end

        function edgeLengths = primEdgeLengths(points)
            pointCount = size(points,1);
            inTree = false(pointCount,1);
            nearest = inf(pointCount,1);
            nearest(1) = 0;
            edgeLengths = zeros(pointCount-1,1);

            for step = 1:pointCount
                candidate = find(~inTree & nearest == min(nearest(~inTree)),1,'first');
                inTree(candidate) = true;
                if step > 1
                    edgeLengths(step-1) = nearest(candidate);
                end
                delta = points-points(candidate,:);
                distance = sqrt(sum(delta.^2,2));
                update = ~inTree & distance < nearest;
                nearest(update) = distance(update);
            end
        end

        function tf = isUnitScalar(value)
            tf = isscalar(value) && isfinite(value) && value >= 0 && value <= 1;
        end
    end
end
