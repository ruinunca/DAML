#!/bin/bash

set -xeu


# for k in 7; do
# for idx in 1 2 3 4 5; do
# 	echo "asdf${idx}_a${k}sdf" #| grep 3 #> log.txt
# done
# done


# s=150
# cuda=5
# ss=3
# # for restaurant domain

# # barely test
# for idx in 9 #1 2 3 4 5 6 7 8 9
# do
# 	# barely test

# 	# python model.py -mode test \
# 	# 				-model tsdf-camrest \
# 	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
# 	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
# 	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
# 	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 	# 					 model_path="./models/fixed${idx}_r_w_b${ss}.pkl" \
# 	# 					 result_path="./results/fixed${idx}_r_w_b${ss}-r.csv" \
# 	# 					 cuda_device=${cuda}

# 	# # adaptation

# 	# cp ./models/fixed${idx}_r_w_b${ss}.pkl ./models/fixed${idx}_r_w_b${ss}_adpr.pkl


# 	# python model.py -mode adjust \
# 	# 				-model tsdf-camrest \
# 	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
# 	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
# 	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
# 	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 	# 					 model_path="./models/fixed${idx}_r_w_b${ss}_adpr.pkl" \
# 	# 					 result_path="./results/fixed${idx}_r_w_b${ss}_adpr-r.csv" \
# 	# 					 cuda_device=${cuda}



# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}_adpr.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}_adpr-r.csv" \
# 						 cuda_device=${cuda}

# done



# # # # # # #for the transfer learning
s=2
cuda=0
ss=$s
for idx in 1 2 3 4 5 6 7 8 9
do
	# barely test

	# python model.py -mode test \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
	# 					 model_path="./models/fixed${idx}_r_w_b${ss}.pkl" \
	# 					 result_path="./results/fixed${idx}_r_w_b${ss}-r.csv" \
	# 					 cuda_device=${cuda}

	# # adaptation

	cp ../sequicity_orig/models/fixed${idx}_r_w_b${ss}.pkl ./models/orig_fixed${idx}_r_w_b${ss}_adpr.pkl


	python model.py -mode adjust \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpr.pkl" \
						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpr-r.csv" \
						 cuda_device=${cuda} \
						 split=110



	# test again
	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/restaurant-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpr.pkl" \
						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpr-r.csv" \
						 cuda_device=${cuda}

done