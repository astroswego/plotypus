#!/bin/bash
~/.local/bin/plotypus -i data/I -o results/lasso-I --periods OGLE-periods.dat -p 8 > results/lasso-I.dat &
~/.local/bin/plotypus -i data/V -o results/lasso-V --periods OGLE-periods.dat -p 8 > results/lasso-V.dat &
~/.local/bin/plotypus -i data/I -o results/baart-I --periods OGLE-periods.dat -p 8 --regressor OLS --selector Baart > results/baart-I.dat &
~/.local/bin/plotypus -i data/V -o results/baart-V --periods OGLE-periods.dat -p 8 --regressor OLS --selector Baart > results/baart-V.dat &
