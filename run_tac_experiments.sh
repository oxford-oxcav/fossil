#! /bin/bash

N_EXP=10

# Run experiments for the paper
# Comment out experiments that are not needed
python3 -m experiments.benchmarks.lyap.non_poly_0 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.lyap.poly_1 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.lyap.ctrllyap0_ct --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.lyap.ctrllyap_invpend --record --plot --repeat $N_EXP 

python3 -m experiments.benchmarks.roa.roa_1 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.roa.ctrl_roa_1 --record --plot --repeat $N_EXP 

python3 -m experiments.benchmarks.barrier.barr_2 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.barrier.barr_4 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.barrier.hi_ord_8 --record --plot --repeat $N_EXP
python3 -m experiments.benchmarks.barrier.ctrlbarr_obstacle --record --plot --repeat $N_EXP 


python3 -m experiments.benchmarks.stabsafe.stablesafe1 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.stabsafe.stablesafe2 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.stabsafe.ctrl_stablesafe1 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.stabsafe.ctrl_stablesafe3 --record --plot --repeat $N_EXP 

# python3 -m experiments.benchmarks.rwa.rwa_1 --record --plot --repeat $N_EXP #13
python3 -m experiments.benchmarks.rwa.rwa_2 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.rwa.rwa_3 --record --plot --repeat $N_EXP 
# python3 -m experiments.benchmarks.rwa.rwa_inv_pend --record --plot --repeat $N_EXP 
# python3 -m experiments.benchmarks.rwa.ctrl_rwa_1 --record --plot --repeat $N_EXP 
# python3 -m experiments.benchmarks.rwa.ctrl_rwa_2 --record --plot --repeat $N_EXP 
# python3 -m experiments.benchmarks.rwa.ctrl_rwa_2_complex --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.rwa.ctrl_rwa_4 --record --plot --repeat $N_EXP  # I thought I'd lost this benchmark, so rwa_2_complex was meant to replace it. I did it differently, and I'm not sure if both are right or if one is better, so will work out later
python3 -m experiments.benchmarks.rwa.ctrl_rwa_3 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.rwa.ctrl_rwa_inv_pend --record --plot --repeat $N_EXP 

# python3 -m experiments.benchmarks.rswa.rswa_1 --record --plot --repeat $N_EXP #23
python3 -m experiments.benchmarks.rswa.rswa_2 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.rswa.rswa_3 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.rswa.rswa_invpend --record --plot --repeat $N_EXP 
# python3 -m experiments.benchmarks.rswa.ctrl_rswa_1 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.rswa.ctrl_rswa_2 --record --plot --repeat $N_EXP 
# python3 -m experiments.benchmarks.rswa.ctrl_rswa_3 --record --plot --repeat $N_EXP 
python3 -m experiments.benchmarks.rswa.ctrl_rswa_invpend --record --plot --repeat $N_EXP 
 
python3 -m experiments.benchmarks.rar.rar_softplus --record --plot --repeat $N_EXP #31
python3 -m experiments.benchmarks.rar.ctrl_rar_invpend --record --plot --repeat $N_EXP 

# Plot results
python3 -m experiments.analysis
