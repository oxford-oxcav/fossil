#! /bin/bash

# Run experiments for the paper
python3 -m experiments.benchmarks.lyap.non_poly_0 --record --repeat 10
python3 -m experiments.benchmarks.lyap.poly_1 --record --repeat 10
python3 -m experiments.benchmarks.lyap.ctrllyap0_ct --record --repeat 10
python3 -m experiments.benchmarks.lyap.ctrllyap_invpend --record --repeat 10

python3 -m experiments.benchmarks.barrier.barr_2 --record --repeat 10
python3 -m experiments.benchmarks.barrier.barr_4 --record --repeat 10
python3 -m experiments.benchmarks.barrier.ctrlbarr_obstacle --record --repeat 10
python3 -m experiments.benchmarks.barrier.ctrlbarr_quadrotor2d --record --repeat 10

python3 -m experiments.benchmarks.rwa.rwa_1 --record --repeat 10
python3 -m experiments.benchmarks.rwa.rwa_2 --record --repeat 10
python3 -m experiments.benchmarks.rwa.rwa_3 --record --repeat 10
python3 -m experiments.benchmarks.rwa.ctrl_rws_1 --record --repeat 10

python3 -m experiments.benchmarks.rswa.rswa_1 --record --repeat 10
python3 -m experiments.benchmarks.rswa.rswa_2 --record --repeat 10
python3 -m experiments.benchmarks.rswa.rswa_3 --record --repeat 10
 
python3 -m experiments.benchmarks.stabsafe.stablesafe1 --record --repeat 10

python3 -m experiments.benchmarks.roa.roa_1 --record --repeat 10

# Plot results
python3 -m experiments.analysis
