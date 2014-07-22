#!/bin/bash
~/.local/bin/plotypus -i ../stellar/data/lmc/v/cep/f -o results/LMC-V-lasso --periods classical-cepheid-periods.dat -p 16 > results/OGLE-LMC-FU-CEP-V-lasso.dat &
~/.local/bin/plotypus -i ../stellar/data/lmc/v/cep/f -o results/LMC-V-ols --periods classical-cepheid-periods.dat -p 16 --regressor OLS --predictor Baart > results/OGLE-LMC-FU-CEP-V-ols.dat &
