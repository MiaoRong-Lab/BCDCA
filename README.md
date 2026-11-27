# BCDCA for PlatEMO

MATLAB implementation of the Bounded Complementary Dual-Channel Algorithm (BCDCA) for constrained multiobjective optimization.

## Requirements

- MATLAB with the Statistics and Machine Learning Toolbox.
- [PlatEMO v4.17](https://github.com/BIMK/PlatEMO/releases/tag/v4.17), installed separately.

The examples were checked with MATLAB R2025b.

## Run BCDCA

In MATLAB, add the PlatEMO directory containing `platemo.m` and this repository's source folder to the path. Then run the paper's configuration on MW1.

```matlab
repoRoot = 'path/to/BCDCA';
platemoRoot = 'path/to/PlatEMO/PlatEMO';
addpath(genpath(platemoRoot));
addpath(fullfile(repoRoot, 'src', 'BCDCA'));

problem = MW1('N', 100, 'maxFE', 100000);
algorithm = BCDCA('save', 0);
algorithm.Solve(problem);
```

The defaults are `archiveRatio = 0.30` and `rhoMin = 0.25`. To supply these two parameters explicitly, use `BCDCA('parameter', {0.30, 0.25}, 'save', 0)`. For the PlatEMO GUI, launch `platemo` and select `src/BCDCA/BCDCA.m` through the algorithm selector's **Open file** option.

To check the installation, add the examples directory and run the included test.

```matlab
addpath(fullfile(repoRoot, 'examples'));
smoke_test_bcdca
```

The test runs MW1, LIRCMOP1, and SDC1 and checks that each search mode runs and preserves the controlled offspring quota.

## Citation

> M. Rong, T. Xiong, H. Sang, and L. Wang, “Bounded Complementary Dual-Channel Algorithm for Constrained Multiobjective Optimization,” *IEEE Transactions on Evolutionary Computation*, 2026, doi: [10.1109/TEVC.2026.3755556](https://doi.org/10.1109/TEVC.2026.3755556).

This implementation uses PlatEMO. Please also cite:

> Y. Tian, R. Cheng, X. Zhang, and Y. Jin, “PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective Optimization [Educational Forum],” *IEEE Computational Intelligence Magazine*, vol. 12, no. 4, pp. 73–87, 2017.

## License

See [LICENSE](LICENSE) for the MIT License. PlatEMO is a separate dependency subject to its own terms.
