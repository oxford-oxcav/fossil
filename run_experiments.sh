#! /bin/bash

N_EXP=10

# Run experiments for the paper
# Comment out experiments that are not needed
python3 -m experiments.benchmarks.lyap.non_poly_0 --record --repeat $N_EXP
python3 -m experiments.benchmarks.lyap.poly_1 --record --repeat $N_EXP
python3 -m experiments.benchmarks.lyap.ctrllyap0_ct --record --repeat $N_EXP
python3 -m experiments.benchmarks.lyap.ctrllyap_invpend --record --repeat $N_EXP

python3 -m experiments.benchmarks.barrier.barr_2 --record --repeat $N_EXP
python3 -m experiments.benchmarks.barrier.barr_4 --record --repeat $N_EXP
python3 -m experiments.benchmarks.barrier.ctrlbarr_obstacle --record --repeat $N_EXP
python3 -m experiments.benchmarks.barrier.ctrlbarr_quadrotor2d --record --repeat $N_EXP

python3 -m experiments.benchmarks.rwa.rwa_1 --record --repeat $N_EXP
# python3 -m experiments.benchmarks.rwa.rwa_2 --record --repeat $N_EXP
python3 -m experiments.benchmarks.rwa.rwa_3 --record --repeat $N_EXP
# python3 -m experiments.benchmarks.rwa.rwa_inv_pend --record --repeat $N_EXP
python3 -m experiments.benchmarks.rwa.ctrl_rwa_1 --record --repeat $N_EXP
# python3 -m experiments.benchmarks.rwa.ctrl_rwa_2 --record --repeat $N_EXP
python3 -m experiments.benchmarks.rwa.ctrl_rwa_3 --record --repeat $N_EXP
# python3 -m experiments.benchmarks.rwa.ctrl_rwa_inv_pend --record --repeat $N_EXP

# python3 -m experiments.benchmarks.rswa.rswa_1 --record --repeat $N_EXP
python3 -m experiments.benchmarks.rswa.rswa_2 --record --repeat $N_EXP
# python3 -m experiments.benchmarks.rswa.rswa_3 --record --repeat $N_EXP
python3 -m experiments.benchmarks.rswa.rswa_invpend --record --repeat $N_EXP
# python3 -m experiments.benchmarks.rswa.ctrl_rswa_1 --record --repeat $N_EXP
python3 -m experiments.benchmarks.rswa.ctrl_rswa_2 --record --repeat $N_EXP
# python3 -m experiments.benchmarks.rswa.ctrl_rswa_3 --record --repeat $N_EXP
python3 -m experiments.benchmarks.rswa.ctrl_rswa_invpend --record --repeat $N_EXP
 
python3 -m experiments.benchmarks.stabsafe.stablesafe1 --record --repeat $N_EXP

python3 -m experiments.benchmarks.roa.roa_1 --record --repeat $N_EXP

# Plot results
python3 -m experiments.analysis
