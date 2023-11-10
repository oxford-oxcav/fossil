#! /bin/bash

function print_experiments_completed() {
    local N=$1
    local TOTAL_EXPERIMENTS=170
    echo "$N out of $TOTAL_EXPERIMENTS experiments completed"
}

N_EXP=10

# Run experiments for the paper
# Comment out experiments that are not needed
python3 -m experiments.benchmarks.lyap.non_poly_2 --record --repeat $N_EXP 
print_experiments_completed 10
python3 -m experiments.benchmarks.lyap.ctrllyap_invpend --record --repeat $N_EXP 
print_experiments_completed 20
python3 -m experiments.benchmarks.roa.ctrl_roa_1 --record --repeat $N_EXP 
print_experiments_completed 30

python3 -m experiments.benchmarks.barrier.hi_ord_8 --record --repeat $N_EXP
print_experiments_completed 40
python3 -m experiments.benchmarks.barrier.ctrlbarr_obstacle --record --repeat $N_EXP 
print_experiments_completed 50


python3 -m experiments.benchmarks.stabsafe.stablesafe1 --record --repeat $N_EXP 
print_experiments_completed 60
python3 -m experiments.benchmarks.stabsafe.ctrl_stablesafe1 --record --repeat $N_EXP 
print_experiments_completed 70

python3 -m experiments.benchmarks.rwa.rwa_3 --record --repeat $N_EXP
print_experiments_completed 80
python3 -m experiments.benchmarks.rwa.ctrl_rwa_4 --record --repeat $N_EXP  
print_experiments_completed 90

python3 -m experiments.benchmarks.rswa.rswa_3 --record --repeat $N_EXP 
print_experiments_completed 100
python3 -m experiments.benchmarks.rswa.ctrl_rswa_invpend --record --repeat $N_EXP 
print_experiments_completed 110
 
python3 -m experiments.benchmarks.rar.rar_softplus --record --repeat $N_EXP #31
print_experiments_completed 120
python3 -m experiments.benchmarks.rar.ctrl_rar_invpend --record --repeat $N_EXP 
print_experiments_completed 130
# Discrete Time
python3 -m experiments.benchmarks.lyap.lyap_dt --record --repeat $N_EXP
print_experiments_completed 140
python3 -m experiments.benchmarks.lyap.ctrllyap_dt --record --repeat $N_EXP
print_experiments_completed 150
python3 -m experiments.benchmarks.barrier.barr_2room_DT --record --repeat $N_EXP
print_experiments_completed 160
python3 -m experiments.benchmarks.barrier.ctrlbarr_2roomtemp_DT --record --repeat $N_EXP
print_experiments_completed 170
# Plot results
python3 fossil/analysis.py
