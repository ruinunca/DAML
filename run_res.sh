#!/bin/bash

set -xeu


s=15
ss=$s
cuda=5
for idx in 1 2 3 4 5 6 7 8 9
do
	# # barely test

	# python model.py -mode test \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
	# 					 model_path="./models/fixed${idx}_b_w_m${ss}.pkl" \
	# 					 result_path="./results/fixed${idx}_b_w_m${ss}-r.csv" \
	# 					 cuda_device=${cuda}

	# # # adaptation

	cp ./models/fixed${idx}_b_w_m${ss}.pkl ./models/fixed${idx}_b_w_m${ss}_adpr.pkl

	# python model.py -mode adjust \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-${s}.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-${s}-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-${s}-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
	# 					 model_path="./models/fixed${idx}_b_w_m${ss}_adpr.pkl" \
	# 					 result_path="./results/fixed${idx}_b_w_m${ss}_adpr-r.csv" \
	# 					 cuda_device=${cuda}

	python model.py -mode adjust \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
						 model_path="./models/fixed${idx}_b_w_m${ss}_adpr.pkl" \
						 result_path="./results/fixed${idx}_b_w_m${ss}_adpr-r.csv" \
						 cuda_device=${cuda}

	# test again
	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
						 model_path="./models/fixed${idx}_b_w_m${ss}_adpr.pkl" \
						 result_path="./results/fixed${idx}_b_w_m${ss}_adpr-r.csv" \
						 cuda_device=${cuda}

done

# s=15
# ss=$s
# cuda=5
# for idx in 1 2 3 4 5 6 7 8 9
# do
# 	# # barely test

# 	# python model.py -mode test \
# 	# 				-model tsdf-camrest \
# 	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
# 	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
# 	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
# 	# 					 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
# 	# 					 model_path="./models/orig_fixed${idx}_b_w_m${ss}.pkl" \
# 	# 					 result_path="./results/orig_fixed${idx}_b_w_m${ss}-r.csv" \
# 	# 					 cuda_device=${cuda}

# 	# # # adaptation

# 	cp ../sequicity_orig/models/fixed${idx}_b_w_m${ss}.pkl ./models/orig_fixed${idx}_b_w_m${ss}_adpr.pkl

# 	# python model.py -mode adjust \
# 	# 				-model tsdf-camrest \
# 	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-${s}.json" \
# 	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-${s}-DB.json" \
# 	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-${s}-OTGY.json" \
# 	# 					 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
# 	# 					 model_path="./models/orig_fixed${idx}_b_w_m${ss}_adpr.pkl" \
# 	# 					 result_path="./results/orig_fixed${idx}_b_w_m${ss}_adpr-r.csv" \
# 	# 					 cuda_device=${cuda}
# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
# 						 model_path="./models/orig_fixed${idx}_b_w_m${ss}_adpr.pkl" \
# 						 result_path="./results/orig_fixed${idx}_b_w_m${ss}_adpr-r.csv" \
# 						 cuda_device=${cuda}
# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_b_w_m_${s}r.pkl" \
# 						 model_path="./models/orig_fixed${idx}_b_w_m${ss}_adpr.pkl" \
# 						 result_path="./results/orig_fixed${idx}_b_w_m${ss}_adpr-r.csv" \
# 						 cuda_device=${cuda}

# done


