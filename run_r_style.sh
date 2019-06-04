#!/bin/bash

set -xeu

s=2
ss=6
cuda=4
e_num=12
for idx in 1 2 3 4 5 6 7 8 9; do
# 	# barely test

	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/fixed${idx}_r_w_b${ss}.pkl" \
						 result_path="./results/fixed${idx}_r_w_b${ss}-r_style.csv" \
						 cuda_device=${cuda}

	# adaptation

	# cp ./models/fixed${idx}_r_w_b${ss}.pkl ./models/fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step.pkl


	# python model.py -mode adjust \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
	# 					 model_path="./models/fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step.pkl" \
	# 					 result_path="./results/fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step-r_style.csv" \
	# 					 cuda_device=${cuda}\
	# 					 split=110 \
	# 					 epoch_num=${e_num}



	# # test again
	# python model.py -mode test \
	# 				-model tsdf-camrest \
	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500.json" \
	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-DB.json" \
	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-OTGY.json" \
	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
	# 					 model_path="./models/fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step.pkl" \
	# 					 result_path="./results/fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step-r_style.csv" \
	# 					 cuda_device=${cuda}

done
# for idx in 1 2 3 4 5 6 7 8 9
# do
# 	cp ./models/fixed${idx}_r_w_b${ss}_adpr.pkl ./models/fixed${idx}_r_w_b${ss}_adprr_style.pkl
# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}_adprr_style.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}_adprr_style-r_style.csv" \
# 						 cuda_device=${cuda} \
# 						 split=110



# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_r_w_b${ss}_adprr_style.pkl" \
# 						 result_path="./results/fixed${idx}_r_w_b${ss}_adprr_style-r_style.csv" \
# 						 cuda_device=${cuda}

# # done

# s=2
# ss=$s
# cuda=1
# e_num=1
# for idx in 1 2 3 4 5 6 7 8 9
# do
# 	# # barely test
# 	# python model.py -mode test \
# 	# 				-model tsdf-camrest \
# 	# 				-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500.json" \
# 	# 					 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-DB.json" \
# 	# 					 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-OTGY.json" \
# 	# 					 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 	# 					 model_path="./models/fixed${idx}_r_w_b${ss}.pkl" \
# 	# 					 result_path="./results/fixed${idx}_r_w_b${ss}-r_style.csv" \
# 	# 					 cuda_device=${cuda}

# 	# adaptation
# 	cp ../sequicity_orig/models/fixed${idx}_r_w_b${ss}.pkl ./models/orig_fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step.pkl


# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step.pkl" \
# 						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step-r_style.csv" \
# 						 cuda_device=${cuda} \
# 						 split=110 \
# 						 epoch_num=${e_num}



# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adpr_style_${e_num}step.pkl" \
# 						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adpr_style-r_${e_num}step_style.csv" \
# 						 cuda_device=${cuda}

# done
# s=2
# ss=$s
# cuda=7
# for idx in 1 2 3 4 5 6 7 8 9
# do
# 	cp ./models/orig_fixed${idx}_r_w_b${ss}_adpr.pkl ./models/orig_fixed${idx}_r_w_b${ss}_adprr_style.pkl
# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adprr_style.pkl" \
# 						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adprr_style-r_style.csv" \
# 						 cuda_device=${cuda} \
# 						 split=110



# 	# test again
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/restaurant_style-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/orig_fixed${idx}_r_w_b${ss}_adprr_style.pkl" \
# 						 result_path="./results/orig_fixed${idx}_r_w_b${ss}_adprr_style-r_style.csv" \
# 						 cuda_device=${cuda}

# done