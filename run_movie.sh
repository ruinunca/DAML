#!/bin/bash

set -xeu

# maml for movie domain
s=15
ss=1
cuda=0
# e_num=10
for idx in 0 1 2 3 4 5 6 7 8 9; do
	# barely test

	# python model.py -mode test \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m.pkl" \
	# 					 model_path="./models/fixed${idx}_rwb_${ss}mstep.pkl" \
	# 					 result_path="./results/fixed${idx}_r_w_b${ss}mstep-m.csv" \
	# 					 cuda_device=${cuda}

	# adaptation

	cp ./models/fixed${idx}_rwb_${ss}mstep.pkl ./models/fixed${idx}_rwb_${ss}mstep_${s}adpm.pkl


	python model.py -mode adjust \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m.pkl" \
						 model_path="./models/fixed${idx}_rwb_${ss}mstep_${s}adpm.pkl" \
						 result_path="./results/fixed${idx}_rwb_${ss}mstep_${s}adpm-m.csv" \
						 cuda_device=${cuda}

	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m.pkl" \
						 model_path="./models/fixed${idx}_rwb_${ss}mstep_${s}adpm.pkl" \
						 result_path="./results/fixed${idx}_rwb_${ss}mstep_${s}adpm-m.csv" \
						 cuda_device=${cuda}
done