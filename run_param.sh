#!/bin/bash
#

set -exu

# # # fine-tuning the parameters

s=150
cuda=4


# # # # # # # # embedding size
# emb_size=200
# for idx in 0 1 2 3 4 5 6 7 8 9; do
# 	# barely test
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_emb${emb_size}.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_emb${emb_size}-m.csv" \
# 						 cuda_device=${cuda} \
# 						 embedding_size=${emb_size} \
# 						 glove_path='./data/glove/glove.6B.200d.txt'

# 	# # adaptation
# 	cp ./models/fixed${idx}_rwb${s}_emb${emb_size}.pkl ./models/fixed${idx}_rwb${s}_emb${emb_size}_adpm.pkl

# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_emb${emb_size}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_emb${emb_size}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 embedding_size=${emb_size} \
# 						 glove_path='./data/glove/glove.6B.200d.txt'

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_emb${emb_size}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_emb${emb_size}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 embedding_size=${emb_size} \
# 						 glove_path='./data/glove/glove.6B.200d.txt'
# done



# # # # # # # # hidden size
# hid_size=128
# for idx in 0 1 2 3 4 5 6 7 8 9; do
# 	# barely test
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_hid${hid_size}.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_hid${hid_size}-m.csv" \
# 						 cuda_device=${cuda} \
# 						 hidden_size=${hid_size}

# 	# adaptation
# 	cp ./models/fixed${idx}_rwb${s}_hid${hid_size}.pkl ./models/fixed${idx}_rwb${s}_hid${hid_size}_adpm.pkl

# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_hid${hid_size}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_hid${hid_size}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 hidden_size=${hid_size}

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_hid${hid_size}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_hid${hid_size}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 hidden_size=${hid_size}
# done




# # # # # # # # learning rate
# lr=0.001
# for idx in 0 1 2 3 4 5 6 7 8 9; do
# 	# barely test
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_lr${lr}.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_lr${lr}-m.csv" \
# 						 cuda_device=${cuda} \
# 						 lr=${lr}

# 	# adaptation
# 	cp ./models/fixed${idx}_rwb${s}_lr${lr}.pkl ./models/fixed${idx}_rwb${s}_lr${lr}_adpm.pkl

# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_lr${lr}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_lr${lr}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 lr=${lr}

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_lr${lr}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_lr${lr}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 lr=${lr}
# done


# # # # # # # # dropout rate
# dropout=0.3
# for idx in 0 1 2 3 4 5 6 7 8 9; do
# 	# barely test
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_dropout${dropout}.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_dropout${dropout}-m.csv" \
# 						 cuda_device=${cuda} \
# 						 dropout_rate=${dropout}

# 	# adaptation
# 	cp ./models/fixed${idx}_rwb${s}_dropout${dropout}.pkl ./models/fixed${idx}_rwb${s}_dropout${dropout}_adpm.pkl

# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_dropout${dropout}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_dropout${dropout}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 dropout_rate=${dropout}

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_dropout${dropout}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_dropout${dropout}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 dropout_rate=${dropout}
# done

# # # # # # # # embedding size & hid_size
# emb_size=200
# hid_size=128
# for idx in 0 1 2 3 4 5 6 7 8 9; do
# 	# barely test
# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}-m.csv" \
# 						 cuda_device=${cuda} \
# 						 embedding_size=${emb_size} \
# 						 hidden_size=${hid_size} \
# 						 glove_path='./data/glove/glove.6B.200d.txt'

# 	# # adaptation
# 	cp ./models/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}.pkl ./models/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}_adpm.pkl

# 	python model.py -mode adjust \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 embedding_size=${emb_size} \
# 						 hidden_size=${hid_size} \
# 						 glove_path='./data/glove/glove.6B.200d.txt'

# 	python model.py -mode test \
# 					-model tsdf-camrest \
# 					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
# 						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
# 						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
# 						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
# 						 model_path="./models/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}_adpm.pkl" \
# 						 result_path="./results/fixed${idx}_rwb${s}_emb${emb_size}_hid${hid_size}_adpm-m.csv" \
# 						 cuda_device=${cuda} \
# 						 embedding_size=${emb_size} \
# 						 hidden_size=${hid_size} \
# 						 glove_path='./data/glove/glove.6B.200d.txt'
# done

# # # # # # # temporary gradient update steps
mstep=7

for idx in 0 1 2 3 4 5 6 7 8 9; do
	# barely test
	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/fixed${idx}_rwb${s}_mstep${mstep}.pkl" \
						 result_path="./results/fixed${idx}_rwb${s}_mstep${mstep}-m.csv" \
						 cuda_device=${cuda} \
						 maml_step=${mstep}


	# # adaptation
	cp ./models/fixed${idx}_rwb${s}_mstep${mstep}.pkl ./models/fixed${idx}_rwb${s}_mstep${mstep}_adpm.pkl

	python model.py -mode adjust \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-${s}-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/fixed${idx}_rwb${s}_mstep${mstep}_adpm.pkl" \
						 result_path="./results/fixed${idx}_rwb${s}_mstep${mstep}_adpm-m.csv" \
						 cuda_device=${cuda} \
						 maml_step=${mstep}

	python model.py -mode test \
					-model tsdf-camrest \
					-cfg data="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500.json" \
						 db="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-DB.json" \
						 entity="../SimDial/1500_data_fixed_${idx}/movie-MixSpec-1500-OTGY.json" \
						 vocab_path="./vocab/vocab-fixed${idx}_r_w_b_${s}m_${s}rslot.pkl" \
						 model_path="./models/fixed${idx}_rwb${s}_mstep${mstep}_adpm.pkl" \
						 result_path="./results/fixed${idx}_rwb${s}_mstep${mstep}_adpm-m.csv" \
						 cuda_device=${cuda} \
						 maml_step=${mstep}
done

