#!/bin/bash

set -xeu

# maml for movie domain
s=150
ss=3
cuda=7
# e_num=10
for idx in 0 1 2 3 4 5 6 7 8 9; do
	# barely test

	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/fixed${idx}_r_w_b${ss}.pkl" \
						 result_path="./results/fixed${idx}_r_w_b${ss}-m.csv" \
						 cuda_device=${cuda}

	# adaptation

	cp ./models/fixed${idx}_r_w_b${ss}.pkl ./models/fixed${idx}_r_w_b${ss}_adpm.pkl


	python model.py -mode adjust \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/fixed${idx}_r_w_b${ss}_adpm.pkl" \
						 result_path="./results/fixed${idx}_r_w_b${ss}_adpm-m.csv" \
						 cuda_device=${cuda}

	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/fixed${idx}_r_w_b${ss}_adpm.pkl" \
						 result_path="./results/fixed${idx}_r_w_b${ss}_adpm-m.csv" \
						 cuda_device=${cuda}
done



# # # movie test for sequicity transfer learning
# s=2
# ss=${s}
# cuda=6
# # e_num=17
# for idx in 1 2 3 4 5 6 7 8 9; do
# 	# barely test

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="../sequicity_orig/models/fixed${idx}_r_w_b${ss}.pkl" \
# 						 result_path="./results/orig_fixed${idx}_r_w_b${ss}-m.csv" \
# 						 cuda_device=${cuda}

	# adaptation

	# cp ../sequicity_orig/models/fixed${idx}_r_w_b${ss}.pkl ./models/orig_fixed${idx}_r_w_b${ss}_adpm_${e_num}step.pkl


	# python model.py -mode adjust \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
	# 					 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpm_${e_num}step.pkl" \
	# 					 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpm_${e_num}step-m.csv" \
	# 					 cuda_device=${cuda} \
	# 					 split=110 \
	# 					 epoch_num=${e_num}

	# # test again
	# python model.py -mode test \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
	# 					 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpm_${e_num}step.pkl" \
	# 					 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpm_${e_num}step-m.csv" \
	# 					 cuda_device=${cuda}

# done


# # single domain meta-learning
# s=15
# ss=${s}
# cuda=4
# for idx in 1 2 3 4 5 6 7 8 9; do
# 	# barely test

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r${ss}.pkl" \
# 						 result_path="./results/fixed${idx}_r${ss}-m.csv" \
# 						 cuda_device=${cuda}

# 	# adaptation

# 	cp ./models/fixed${idx}_r${ss}.pkl ./models/fixed${idx}_r${ss}_adpm.pkl


# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r${ss}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_r${ss}_adpm-m.csv" \
# 						 cuda_device=${cuda} #\
# 						 # split=110




# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r${ss}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_r${ss}_adpm-m.csv" \
# 						 cuda_device=${cuda}

# done

# movie test for sequicity transfer learning(sequicity trained)
# s=15
# ss=${s}
# cuda=4
# for idx in 1 2 3 4 5 6 7 8 9; do
# 	# barely test

# 	# python model.py -mode test \
# 	# 				-model tsdf-camrest \
# 	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 	# 					 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 	# 					 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 	# 					 model_path="../sequicity_orig/models/fixed${idx}_r_w_b${ss}.pkl" \
# 	# 					 result_path="./results/orig_fixed${idx}_r_w_b${ss}-m.csv" \
# 						 # cuda_device=${cuda}

# 	# adaptation

# 	cp ./models/transfer_fixed${idx}_r_w_b${ss}.pkl ./models/transfer_fixed${idx}_r_w_b${ss}_adpm.pkl


# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/transfer_fixed${idx}_r_w_b${ss}_adpm.pkl" \
# 						 result_path="./results/transfer_fixed${idx}_r_w_b${ss}_adpm-m.csv" \
# 						 cuda_device=${cuda} #\
# 						 # split=110

# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/transfer_fixed${idx}_r_w_b${ss}_adpm.pkl" \
# 						 result_path="./results/transfer_fixed${idx}_r_w_b${ss}_adpm-m.csv" \
# 						 cuda_device=${cuda}

# done