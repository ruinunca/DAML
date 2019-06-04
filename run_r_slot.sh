#!/bin/bash

set -xeu

# s=150
# ss=3
# cuda=7
# # barely test
# for idx in 1 2 3 4 5 6 7 8 9; do
# 	# barely test

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}-r_slot.csv" \
# 						 cuda_device=${cuda}

# 	# adaptation

# 	cp ./models/fixed${idx}_r_w_b${ss}.pkl ./models/fixed${idx}_r_w_b${ss}_adpr_slot.pkl


# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}_adpr_slot.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}_adpr_slot-r_slot.csv" \
# 						 cuda_device=${cuda}\
# 						 split=110



# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}_adpr_slot.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}_adpr_slot-r_slot.csv" \
# 						 cuda_device=${cuda}

# done


# for idx in 1 2 3 4 5 6 7 8 9; do
# 	cp ./models/fixed${idx}_r_w_b${ss}_adpr.pkl ./models/fixed${idx}_r_w_b${ss}_adprr_slot.pkl
# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}_adprr_slot.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}_adprr_slot-r_slot.csv" \
# 						 cuda_device=${cuda}



# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}_adprr_slot.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}_adprr_slot-r_slot.csv" \
# 						 cuda_device=${cuda}

# done









# # # # # # #for the transfer learning
# s=2
# ss=$s
# cuda=0
# for idx in 1 2 3 4 5 6 7 8 9
# do
# 	# # barely test
# 	# python model.py -mode test \
# 	# 				-model tsdf-camrest \
# 	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500.json" \
# 	# 					 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-DB.json" \
# 	# 					 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-OTGY.json" \
# 	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 	# 					 model_path="./models/fixed${idx}_r_w_b${ss}.pkl" \
# 	# 					 result_path="./results/fixed${idx}_r_w_b${ss}-r_slot.csv" \
# 	# 					 cuda_device=${cuda}

# 	# adaptation
# 	cp ../sequicity_orig/models/fixed${idx}_r_w_b${ss}.pkl ./models/orig_fixed${idx}_r_w_b${ss}_adpr_slot.pkl

# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpr_slot.pkl" \
# 						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpr_slot-r_slot.csv" \
# 						 cuda_device=${cuda} \
# 						 split=110

# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpr_slot.pkl" \
# 						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpr_slot-r_slot.csv" \
# 						 cuda_device=${cuda}

# done

s=2
ss=$s
cuda=7
for idx in 1 2 3 4 5 6 7 8 9; do
	cp ./models/orig_fixed${idx}_r_w_b${ss}_adpr.pkl ./models/orig_fixed${idx}_r_w_b${ss}_adprr_slot.pkl

	python model.py -mode adjust \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}.json" \
						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-${s}-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adprr_slot.pkl" \
						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adprr_slot-r_slot.csv" \
						 cuda_device=${cuda} \
						 split=110



	# test again
	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/rest_pitt-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adprr_slot.pkl" \
						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adprr_slot-r_slot.csv" \
						 cuda_device=${cuda}

done








