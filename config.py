import logging
import time
import configparser

class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 0
        self.eos_m_token = 'EOS_M'       
        self.beam_len_bonus = 0.5

        self.mode = 'unknown'
        self.m = 'TSD'
        self.prev_z_method = 'none'

        self.seed = 0
  
    def init_handler(self, m):
        init_method = {
            'tsdf-camrest':self._camrest_tsdf_init,
            'tsdf-kvret':self._kvret_tsdf_init
        }
        init_method[m]()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # this is for test different parameters # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    def _camrest_tsdf_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.vocab_size = 800

        self.embedding_size = 50
        self.glove_path = './data/glove/glove.6B.' + str(self.embedding_size) + 'd.txt'
        self.hidden_size = 50
        self.lr = 0.003
        self.dropout_rate = 0.3
        self.maml_step=7

        self.split = (9, 1, 5)
        self.lr_decay = 0.5
        self.batch_size = 32
        self.z_length = 8
        self.degree_size = 5
        self.layer_num = 1
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 2
        self.cuda = True
        self.spv_proportion = 100
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True

        self.enlarge_vocab = False

        idx = 9
        size= 150
        self.data = [
                     # # # # bus
                     '../SimDial/1500_data_fixed_'+str(idx)+'/bus-MixSpec-1500.json',
                     # # # # # restaurant
                     '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500.json',
                     # # # # weather
                     '../SimDial/1500_data_fixed_'+str(idx)+'/weather-MixSpec-1500.json',
                     # # # # # movie
                     # '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500.json'
                    ]

        self.db = [
                     # # # # bus
                     '../SimDial/1500_data_fixed_'+str(idx)+'/bus-MixSpec-1500-DB.json',
                     # # # # restaurant
                     '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500-DB.json',
                     # # # weather
                     '../SimDial/1500_data_fixed_'+str(idx)+'/weather-MixSpec-1500-DB.json',
                     # # # # movie
                     # '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-DB.json'
                    ]

        self.entity = [
                         # # # # # bus
                         '../SimDial/1500_data_fixed_'+str(idx)+'/bus-MixSpec-1500-OTGY.json',
                         # # restaurant
                         '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500-OTGY.json',
                         # # # weather
                         '../SimDial/1500_data_fixed_'+str(idx)+'/weather-MixSpec-1500-OTGY.json',
                         # # # # movie
                         # '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-OTGY.json'
                        ]

        self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_w_b_'+str(size)+'m_'+str(size)+'rslot.pkl'

        # self.model_path = './models/fixed'+str(idx)+'_rwb'+str(size)+'_emb'+str(self.embedding_size)+'.pkl'
        # self.result_path = './results/fixed'+str(idx)+'_rwb'+str(size)+'_emb'+str(self.embedding_size)+'.csv'

        # self.model_path = './models/fixed'+str(idx)+'_rwb'+str(size)+'_hid'+str(self.hidden_size)+'.pkl'
        # self.result_path = './results/fixed'+str(idx)+'_rwb'+str(size)+'_hid'+str(self.hidden_size)+'.csv'

        # self.model_path = './models/fixed'+str(idx)+'_rwb'+str(size)+'_lr'+str(self.lr)+'.pkl'
        # self.result_path = './results/fixed'+str(idx)+'_rwb'+str(size)+'_lr'+str(self.lr)+'.csv'

        # self.model_path = './models/fixed'+str(idx)+'_rwb'+str(size)+'_dropout'+str(self.dropout_rate)+'.pkl'
        # self.result_path = './results/fixed'+str(idx)+'_rwb'+str(size)+'_dropout'+str(self.dropout_rate)+'.csv'

        # self.model_path = './models/fixed'+str(idx)+'_rwb'+str(size)+'_emb'+str(self.embedding_size)+'_hid'+str(self.hidden_size)+'.pkl'
        # self.result_path = './results/fixed'+str(idx)+'_rwb'+str(size)+'_emb'+str(self.embedding_size)+'_hid'+str(self.hidden_size)+'.csv'

        self.model_path = './models/fixed'+str(idx)+'_rwb'+str(size)+'_mstep'+str(self.maml_step)+'.pkl'
        self.result_path = './results/fixed'+str(idx)+'_rwb'+str(size)+'_mstep'+str(self.maml_step)+'.csv'

        # # movie
        # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500.json'
        # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-DB.json'
        # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-OTGY.json'

    #     # # movie
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-15.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-15-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-15-OTGY.json'

        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # this is test for different datasize # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # def _camrest_tsdf_init(self):
    #     self.beam_len_bonus = 0.5
    #     self.prev_z_method = 'separate'
    #     self.vocab_size = 800
    #     self.embedding_size = 50
    #     self.hidden_size = 50
    #     # self.split = (3, 1, 1)
    #     self.split = (9, 1, 5)
    #     self.lr = 0.003
    #     self.lr_decay = 0.5

    #     self.enlarge_vocab = False

    #     # self.vocab_path = './vocab/vocab-fixed_r_w_b.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed_r_w_b_150m.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed_r_w_b_150m_150rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed_r_w_b_75m_75rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed_r_w_b_15m_15rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed6_r_w_b_2m_2rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed_all.pkl'

    #     idx = 8
    #     size= 2
    #     # self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_w_b_15m_15rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_w_b_75m_75rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_w_b_2m_2rslot.pkl'
    #     self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_w_b_150m_150rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_w_b_'+str(size)+'m_'+str(size)+'rslot.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_b_m_'+str(size)+'w.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_r_w_m_'+str(size)+'b.pkl'
    #     # self.vocab_path = './vocab/vocab-fixed'+str(idx)+'_b_w_m_'+str(size)+'r.pkl'

    #     # self.model_path = "./models/fixed"+str(idx)+"_r_w_b6.pkl" 
    #     # self.result_path="./results/fixed"+str(idx)+"_r_w_b6-tmp.csv"


    #     # self.data = [
    #     #              # # # # bus
    #     #              '../SimDial/1500_data_fixed_'+str(idx)+'/bus-MixSpec-1500.json',
    #     #              # # # # # restaurant
    #     #              # '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500.json',
    #     #              # # # # weather
    #     #              '../SimDial/1500_data_fixed_'+str(idx)+'/weather-MixSpec-1500.json',
    #     #              # # # # movie
    #     #              '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500.json'
    #     #             ]

    #     # self.db = [
    #     #              # # # # bus
    #     #              '../SimDial/1500_data_fixed_'+str(idx)+'/bus-MixSpec-1500-DB.json',
    #     #              # # # # restaurant
    #     #              # '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500-DB.json',
    #     #              # # # weather
    #     #              '../SimDial/1500_data_fixed_'+str(idx)+'/weather-MixSpec-1500-DB.json',
    #     #              # # # weather
    #     #              '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-DB.json'
    #     #             ]

    #     # self.entity = [
    #     #                  # # # # # bus
    #     #                  '../SimDial/1500_data_fixed_'+str(idx)+'/bus-MixSpec-1500-OTGY.json',
    #     #                  # # restaurant
    #     #                  # '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500-OTGY.json',
    #     #                  # # # weather
    #     #                  '../SimDial/1500_data_fixed_'+str(idx)+'/weather-MixSpec-1500-OTGY.json',
    #     #                  # # # movie
    #     #                  '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-OTGY.json'
    #     #                 ]

    #     # self.model_path = './models/fixed'+str(idx)+'_b_w_m'+str(size)+'.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_b_w_m'+str(size)+'.csv'

    #     # self.model_path = './models/fixed'+str(idx)+'_r_w_m'+str(size)+'.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_m'+str(size)+'.csv'

    #     # self.model_path = './models/fixed'+str(idx)+'_r_b_m'+str(size)+'.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_b_m'+str(size)+'.csv'

    #     # # restaurant_weather_bus
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/r_w_b.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/r_w_b-OTGY.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/r_w_b-DB.json'

    #     # self.model_path = './models/transfer_fixed'+str(idx)+'_r_w_b'+str(size)+'.pkl'
    #     # self.result_path = './results/transfer_fixed'+str(idx)+'_r_w_b'+str(size)+'.csv'

    #     # self.model_path = './models/fixed'+str(idx)+'_r'+str(size)+'.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r'+str(size)+'.csv'

    #     # # restaurant
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant-MixSpec-1500-OTGY.json'

    #     # movie
    #     self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500.json'
    #     self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-DB.json'
    #     self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-1500-OTGY.json'

    #     # # restaurant Pitt
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/rest_pitt-MixSpec-1500.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/rest_pitt-MixSpec-1500-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/rest_pitt-MixSpec-1500-OTGY.json'

    #     # # restaurant style
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant_style-MixSpec-1500.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant_style-MixSpec-1500-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant_style-MixSpec-1500-OTGY.json'

    #     # # ####################################################################
    #     # # # # #fixed_r_w_b5 use vocab: 'vocab-fixed'+str(idx)+'_r_w_b_15m_15rslot.pkl'
    #     # # movie
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-15.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-15-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/movie-MixSpec-15-OTGY.json'

    #     # # restaurant Pitt
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/rest_pitt-MixSpec-15.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/rest_pitt-MixSpec-15-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/rest_pitt-MixSpec-15-OTGY.json'

    #     # # restaurant style
    #     # self.data = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant_style-MixSpec-15.json'
    #     # self.db = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant_style-MixSpec-15-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_'+str(idx)+'/restaurant_style-MixSpec-15-OTGY.json'

    #     # self.model_path = './models/fixed'+str(idx)+'_r_w_b5.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5.csv'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5-r.csv'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5-m.csv'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5-r_slot.csv'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5-r_style.csv'

    #     # self.model_path = './models/fixed'+str(idx)+'_r_w_b5_adpr.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5_adpr-r.csv'

    #     # # self.model_path = './models/fixed'+str(idx)+'_r_w_b5_adpm.pkl'
    #     # # self.result_path = './results/fixed'+str(idx)+'_r_w_b5_adpm-m.csv'

    #     # # self.model_path = './models/fixed'+str(idx)+'_r_w_b5_adpr_slot.pkl'
    #     # # self.result_path = './results/fixed'+str(idx)+'_r_w_b5_adpr_slot-r_slot.csv'
    #     # self.model_path = './models/fixed'+str(idx)+'_r_w_b5_adprr_slot.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5_adprr_slot-r_slot.csv'

    #     # # # self.model_path = './models/fixed'+str(idx)+'_r_w_b5_adpr_style.pkl'
    #     # # # self.result_path = './results/fixed'+str(idx)+'_r_w_b5_adpr_style-r_style.csv'
    #     # self.model_path = './models/fixed'+str(idx)+'_r_w_b5_adprr_style.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b5_adprr_style-r_style.csv'
        
    #     ####################################################################
    #     # # ####################################################################
    #     # # # # #fixed_r_w_b6 use vocab: 'vocab-fixed_r_w_b_2m_2rslot.pkl'
    #     # # # movie
    #     # # self.data = '../SimDial/1500_data_fixed/movie-MixSpec-2.json'
    #     # # self.db = '../SimDial/1500_data_fixed/movie-MixSpec-2-DB.json'
    #     # # self.entity = '../SimDial/1500_data_fixed/movie-MixSpec-2-OTGY.json'

    #     # # # restaurant Pitt
    #     # # self.data = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-2.json'
    #     # # self.db = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-2-DB.json'
    #     # # self.entity = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-2-OTGY.json'

    #     # # # restaurant style
    #     # # self.data = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-2.json'
    #     # # self.db = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-2-DB.json'
    #     # # self.entity = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-2-OTGY.json'

    #     # self.model_path = './models/fixed_r_w_b6.pkl'
    #     # self.result_path = './results/fixed_r_w_b6.csv'
    #     # self.result_path = './results/fixed_r_w_b6-r.csv'
    #     # self.result_path = './results/fixed_r_w_b6-m.csv'
    #     # self.result_path = './results/fixed_r_w_b6-r_slot.csv'
    #     # self.result_path = './results/fixed_r_w_b6-r_style.csv'

    #     # self.model_path = './models/fixed_r_w_b6_adpm.pkl'
    #     # self.result_path = './results/fixed_r_w_b6_adpm-m.csv'

    #     # # self.model_path = './models/fixed_r_w_b6_adpr.pkl'
    #     # # self.result_path = './results/fixed_r_w_b6_adpr-r.csv'

    #     # # self.model_path = './models/fixed_r_w_b6_adpr_slot.pkl'
    #     # # self.result_path = './results/fixed_r_w_b6_adpr_slot-r_slot.csv'
    #     # self.model_path = './models/fixed_r_w_b6_adprr_slot.pkl'
    #     # self.result_path = './results/fixed_r_w_b6_adprr_slot-r_slot.csv'

    #     # # # self.model_path = './models/fixed_r_w_b6_adpr_style.pkl'
    #     # # # self.result_path = './results/fixed_r_w_b6_adpr_style-r_style.csv'
    #     # self.model_path = './models/fixed_r_w_b6_adprr_style.pkl'
    #     # self.result_path = './results/fixed_r_w_b6_adprr_style-r_style.csv'
        
    #     # # ####################################################################
    #     # # # # #fixed_r_w_b5 use vocab: 'vocab-fixed_r_w_b_15m_15rslot.pkl'
    #     # # movie
    #     # self.data = '../SimDial/1500_data_fixed_3/movie-MixSpec-1500.json'
    #     # self.db = '../SimDial/1500_data_fixed_3/movie-MixSpec-1500-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed_3/movie-MixSpec-1500-OTGY.json'

    #     # # # movie
    #     # # self.data = '../SimDial/1500_data_fixed/movie-MixSpec-15.json'
    #     # # self.db = '../SimDial/1500_data_fixed/movie-MixSpec-15-DB.json'
    #     # # self.entity = '../SimDial/1500_data_fixed/movie-MixSpec-15-OTGY.json'

    #     # # # restaurant Pitt
    #     # # self.data = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-15.json'
    #     # # self.db = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-15-DB.json'
    #     # # self.entity = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-15-OTGY.json'

    #     # # # restaurant style
    #     # # self.data = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-15.json'
    #     # # self.db = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-15-DB.json'
    #     # # self.entity = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-15-OTGY.json'

    #     # self.model_path = './models/fixed_r_w_b5.pkl'
    #     # self.result_path = './results/fixed_r_w_b5.csv'
    #     # self.result_path = './results/fixed_r_w_b5-r.csv'
    #     # self.result_path = './results/fixed_r_w_b5-m.csv'
    #     # self.result_path = './results/fixed_r_w_b5-r_slot.csv'
    #     # self.result_path = './results/fixed_r_w_b5-r_style.csv'

    #     # self.model_path = './models/fixed3_r_w_b6_adpm.pkl'
    #     # self.result_path = './results/fixed3_r_w_b6_adpm-m-tmp.csv'

    #     # # self.model_path = './models/fixed_r_w_b5_adpr.pkl'
    #     # # self.result_path = './results/fixed_r_w_b5_adpr-r.csv'

    #     # # self.model_path = './models/fixed_r_w_b5_adpr_slot.pkl'
    #     # # self.result_path = './results/fixed_r_w_b5_adpr_slot-r_slot.csv'
    #     # self.model_path = './models/fixed_r_w_b5_adprr_slot.pkl'
    #     # self.result_path = './results/fixed_r_w_b5_adprr_slot-r_slot.csv'

    #     # # # self.model_path = './models/fixed_r_w_b5_adpr_style.pkl'
    #     # # # self.result_path = './results/fixed_r_w_b5_adpr_style-r_style.csv'
    #     # self.model_path = './models/fixed_r_w_b5_adprr_style.pkl'
    #     # self.result_path = './results/fixed_r_w_b5_adprr_style-r_style.csv'
        
    #     ####################################################################
    #     # # # # # #fixed_r_w_b4 use vocab: 'vocab-fixed_r_w_b_75m_75rslot.pkl'
    #     # # # movie
    #     # # self.data = '../SimDial/1500_data_fixed/movie-MixSpec-75.json'
    #     # # self.db = '../SimDial/1500_data_fixed/movie-MixSpec-75-DB.json'
    #     # # self.entity = '../SimDial/1500_data_fixed/movie-MixSpec-75-OTGY.json'

    #     # # restaurant Pitt
    #     # self.data = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-75.json'
    #     # self.db = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-75-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-75-OTGY.json'

    #     # # restaurant style
    #     # self.data = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-75.json'
    #     # self.db = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-75-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-75-OTGY.json'

    #     # self.model_path = './models/fixed'+str(idx)+'_r_w_b4.pkl'
    #     # self.result_path = './results/fixed'+str(idx)+'_r_w_b4.csv'
    #     # self.result_path = './results/fixed_r_w_b4-r.csv'
    #     # self.result_path = './results/fixed_r_w_b4-m.csv'
    #     # self.result_path = './results/fixed_r_w_b4-r_slot.csv'
    #     # # self.result_path = './results/fixed_r_w_b4-r_style.csv'

    #     # self.model_path = './models/fixed_r_w_b4_adpm.pkl'
    #     # self.result_path = './results/fixed_r_w_b4_adpm-m.csv'

    #     # self.model_path = './models/fixed_r_w_b4_adpr.pkl'
    #     # self.result_path = './results/fixed_r_w_b4_adpr-r.csv'

    #     # self.model_path = './models/fixed_r_w_b4_adpr_slot.pkl'
    #     # self.result_path = './results/fixed_r_w_b4_adpr_slot-r_slot.csv'
    #     # self.model_path = './models/fixed_r_w_b4_adprr_slot.pkl'
    #     # self.result_path = './results/fixed_r_w_b4_adprr_slot-r_slot.csv'

    #     # self.model_path = './models/fixed_r_w_b4_adpr_style.pkl'
    #     # self.result_path = './results/fixed_r_w_b4_adpr_style-r_style.csv'
    #     # self.model_path = './models/fixed_r_w_b4_adprr_style.pkl'
    #     # self.result_path = './results/fixed_r_w_b4_adprr_style-r_style.csv'

    #     ###################################################################
    #     # # # # #fixed_r_w_b3 use vocab: 'vocab-fixed_r_w_b_150m_150rslot.pkl'
    #     # # movie
    #     # self.data = '../SimDial/1500_data_fixed/movie-MixSpec-150.json'
    #     # self.db = '../SimDial/1500_data_fixed/movie-MixSpec-150-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed/movie-MixSpec-150-OTGY.json'

    #     # # restaurant Pitt
    #     # self.data = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-150.json'
    #     # self.db = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-150-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed/rest_pitt-MixSpec-150-OTGY.json'

    #     # # restaurant style
    #     # self.data = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-150.json'
    #     # self.db = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-150-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed/restaurant_style-MixSpec-150-OTGY.json'

    #     self.model_path = './models/fixed0_r_w_b3.pkl'
    #     self.result_path = './results/fixed0_r_w_b3.csv'
    # #     # self.result_path = './results/fixed_r_w_b3-r.csv'
    #     # self.result_path = './results/fixed_r_w_b3-m.csv'
    #     # self.result_path = './results/fixed_r_w_b3-r_slot.csv'
    #     # self.result_path = './results/fixed_r_w_b3-r_style.csv'

    #     # self.model_path = './models/fixed_r_w_b3_adpm.pkl'
    #     # self.result_path = './results/fixed_r_w_b3_adpm-m.csv'

    #     # # self.model_path = './models/fixed_r_w_b3_adpmnrand.pkl'
    #     # # self.result_path = './results/fixed_r_w_b3_adpmnrand-m.csv'

    #     # # self.model_path = './models/fixed_r_w_b3_adpr.pkl'
    #     # # self.result_path = './results/fixed_r_w_b3_adpr-r.csv'
    #     # # # self.model_path = './models/fixed_r_w_b3_adprrand.pkl'
    #     # # # self.result_path = './results/fixed_r_w_b3_adprrand-r.csv'

    #     # self.model_path = './models/fixed_r_w_b3_adpr_slot.pkl'
    #     # self.result_path = './results/fixed_r_w_b3_adpr_slot-r_slot.csv'
    #     # self.model_path = './models/fixed_r_w_b3_adprr_slot.pkl'
    #     # self.result_path = './results/fixed_r_w_b3_adprr_slot-r_slot.csv'

    #     # self.model_path = './models/fixed_r_w_b3_adpr_style.pkl'
    #     # self.result_path = './results/fixed_r_w_b3_adpr_style-r_style.csv'
    #     # self.model_path = './models/fixed_r_w_b3_adprr_style.pkl'
    #     # self.result_path = './results/fixed_r_w_b3_adprr_style-r_style.csv'


    #     # # # # #fixed_r_w_b2 use vocab: 'vocab-fixed_r_w_b_150m.pkl'
    #     # # self.model_path = './models/fixed_r_w_b2.pkl'
    #     # # # self.result_path = './results/fixed_r_w_b2-r.csv'
    #     # # # self.result_path = './results/fixed_r_w_b2-m.csv'
    #     # # # self.result_path = './results/fixed_r_w_b2-r_slot.csv'
    #     # # self.result_path = './results/fixed_r_w_b2-r_style.csv'

    #     # # self.model_path = './models/fixed_r_w_b2_adpr.pkl'
    #     # # self.result_path = './results/fixed_r_w_b2_adpr-r.csv'

    #     # # self.model_path = './models/fixed_r_w_b2_adpm.pkl'
    #     # # self.result_path = './results/fixed_r_w_b2_adpm-m.csv'
    #     # # self.model_path = './models/fixed_r_w_b2_adpm2.pkl'
    #     # # self.result_path = './results/fixed_r_w_b2_adpm2-m2.csv'

    #     # # self.model_path = './models/fixed_r_w_b2_adpr_slot.pkl'
    #     # # self.result_path = './results/fixed_r_w_b2_adpr_slot-r_slot.csv'

    #     # # self.model_path = './models/fixed_r_w_b2_adpr_style.pkl'
    #     # # self.result_path = './results/fixed_r_w_b2_adpr_style-r_style.csv'

    #     # self.model_path = './models/fixed_r_w_b2_adpm-tmp.pkl'
    #     # self.result_path = './results/fixed_r_w_b2_adpm-m-tmp.csv'




    #     # # self.model_path = './models/fixed_r_w_b_150m.pkl'
    #     # # self.result_path = './results/fixed_r_w_b_150m-m.csv'

    #     # # self.model_path = './models/fixed_r_w_b_150m_adpm.pkl'
    #     # # self.result_path = './results/fixed_r_w_b_150m_adpm-m.csv'

    #     # # self.model_path = './models/fixed_all.pkl'
    #     # # # self.result_path = './results/fixed_all-r.csv'
    #     # # self.result_path = './results/fixed_all-m.csv'

    #     # # self.model_path = './models/fixed_all_adpm.pkl'
    #     # # self.result_path = './results/fixed_all_adpm-m.csv'

    #     #######################################
    #     ##########     add domain     #########
    #     #######################################

    #     # self.data = [
    #     #              # # # # bus
    #     #              '../SimDial/1500_data_fixed_dom/bus-MixSpec-1500.json',
    #     #              # # # # restaurant
    #     #              '../SimDial/1500_data_fixed_dom/restaurant-MixSpec-1500.json',
    #     #              # # # # weather
    #     #              '../SimDial/1500_data_fixed_dom/weather-MixSpec-1500.json',
    #     #              # # # # movie
    #     #              # '../SimDial/1500_data_fixed_dom/movie-MixSpec-150x10.json'
    #     #             ]

    #     # self.db = [
    #     #              # # # bus
    #     #              '../SimDial/1500_data_fixed_dom/bus-MixSpec-1500-DB.json',
    #     #              # # # # restaurant
    #     #              '../SimDial/1500_data_fixed_dom/restaurant-MixSpec-1500-DB.json',
    #     #              # # # weather
    #     #              '../SimDial/1500_data_fixed_dom/weather-MixSpec-1500-DB.json',
    #     #              # # # movie
    #     #              # '../SimDial/1500_data_fixed_dom/movie-MixSpec-150-DB.json'
    #     #             ]

    #     # self.entity = [
    #     #                  # # # # bus
    #     #                  '../SimDial/1500_data_fixed_dom/bus-MixSpec-1500-OTGY.json',
    #     #                  # # restaurant
    #     #                  '../SimDial/1500_data_fixed_dom/restaurant-MixSpec-1500-OTGY.json',
    #     #                  # # # weather
    #     #                  '../SimDial/1500_data_fixed_dom/weather-MixSpec-1500-OTGY.json',
    #     #                  # # movie
    #     #                  # '../SimDial/1500_data_fixed_dom/movie-MixSpec-150-OTGY.json'
    #     #                 ]

    #     # self.vocab_path = './vocab/vocab-dom_fixed_r_w_b_150m_150rslot.pkl'

    #     # # self.model_path = './models/dom_fixed_r_w_b3.pkl'
    #     # # self.result_path = './results/dom_fixed_r_w_b3.csv'

    #     # # movie
    #     # self.data = '../SimDial/1500_data_fixed/movie-MixSpec-150.json'
    #     # self.db = '../SimDial/1500_data_fixed/movie-MixSpec-150-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed/movie-MixSpec-150-OTGY.json'
    #     # # movie
    #     # self.data = '../SimDial/1500_data_fixed/movie-MixSpec-1500.json'
    #     # self.db = '../SimDial/1500_data_fixed/movie-MixSpec-1500-DB.json'
    #     # self.entity = '../SimDial/1500_data_fixed/movie-MixSpec-1500-OTGY.json'

    #     # self.model_path = './models/dom_fixed_r_w_b3_adpm.pkl'
    #     # self.result_path = './results/dom_fixed_r_w_b3_adpm-m.csv'
    #     # # self.model_path = './models/dom_fixed_r_w_b3_adpmrand.pkl'
    #     # # self.result_path = './results/dom_fixed_r_w_b3_adpmrand-m.csv'



    #     self.glove_path = './data/glove/glove.6B.50d.txt'
    #     self.batch_size = 32
    #     self.z_length = 8
    #     self.degree_size = 5
    #     self.layer_num = 1
    #     self.dropout_rate = 0.5
    #     self.epoch_num = 100 # triggered by early stop
    #     self.rl_epoch_num = 2
    #     self.cuda = True
    #     self.spv_proportion = 100
    #     self.max_ts = 40
    #     self.early_stop_count = 5
    #     self.new_vocab = True



    #     self.teacher_force = 100
    #     self.beam_search = False
    #     self.beam_size = 10
    #     self.sampling = False
    #     self.unfrz_attn_epoch = 0
    #     self.skip_unsup = False
    #     self.truncated = False
    #     self.pretrain = False

    # def _camrest_tsdf_init(self):
    #     self.beam_len_bonus = 0.5
    #     self.prev_z_method = 'separate'
    #     self.vocab_size = 800
    #     self.embedding_size = 50
    #     self.hidden_size = 50
    #     # self.split = (3, 1, 1)
    #     self.split = (9, 1, 5)
    #     self.lr = 0.003
    #     self.lr_decay = 0.5

    #     # self.vocab_path = ''
    #     self.enlarge_vocab = False

    #     # self.vocab_path = './vocab/vocab-camrest.pkl'
    #     # self.data = [
    #     #              # # # camrest
    #     #              # './data/CamRest676/CamRest676.json',
    #     #              './data/CamRest676/CamRest676.json'
    #     #             ]
    #     # # 
    #     # self.db = [
    #     #              # # camrest
    #     #              # './data/CamRest676/CamRestDB.json',
    #     #              './data/CamRest676/CamRestDB.json'
    #     #             ]

    #     # self.entity = [
    #     #                  # # camrest
    #     #                  # './data/CamRest676/CamRestOTGY.json',
    #     #                  './data/CamRest676/CamRestOTGY.json'
    #     #                 ]
    #     # self.model_path = './models/camrest_single.pkl'
    #     # # self.model_path = './models/camrest_single_0.0015.pkl'
    #     # # self.model_path = './models/camrest_single_weight_decay.pkl'
    #     # # self.model_path = './models/camrest_single_weight_decay_adjust.pkl'
    #     # # self.model_path = './models/camrest_single_weight_decay_metalr_decay.pkl'
    #     # # self.model_path = './models/camrest_single_weight_decay_metalr_decay_adjust.pkl'
    #     # self.result_path = './results/camrest.csv'


    #     self.vocab_path = 'vocab/vocab-rest_weat_bus_littlemovie.pkl'
    #     # self.vocab_path = 'vocab/vocab-rest_weat_bus.pkl'
    #     # self.vocab_path = './vocab/vocab-rest_weat_bus.pkl'
    #     # # # self.vocab_path = './vocab/vocab-simdial_rest.pkl'
    #     # self.vocab_path = './vocab/vocab-all.pkl'

    #     self.data = [
    #                  # # # # # bus
    #                  # '../SimDial/1500_data/bus-MixSpec-1500.json',
    #                  # # # # # restaurant
    #                  # '../SimDial/1500_data/restaurant-MixSpec-1500.json',
    #                  # # # # # weather
    #                  # '../SimDial/1500_data/weather-MixSpec-1500.json',
    #                  # # # movie
    #                  # '../SimDial/1500_data/movie-MixSpec-150.json',
    #                  # # # # movie
    #                  '../SimDial/1500_data/movie-MixSpec-1500.json',

    #                  # # # restaurant Pitt
    #                  # '../SimDial/1500_data/rest_pitt-MixSpec-1500.json',
    #                  # restaurant style
    #                  # '../SimDial/1500_data/restaurant_style-MixSpec-1500.json'
    #                 ]

    #     self.db = [
    #     #              # # # bus
    #     #              '../SimDial/1500_data/bus-MixSpec-1500-DB.json',
    #     #              # # # # restaurant
    #                  # '../SimDial/1500_data/restaurant-MixSpec-1500-DB.json',
    #     #              # # # weather
    #                  # '../SimDial/1500_data/weather-MixSpec-1500-DB.json',
    #                  # # # movie
    #                  # '../SimDial/1500_data/movie-MixSpec-150-DB.json',
    #                  # # # movie
    #                  '../SimDial/1500_data/movie-MixSpec-1500-DB.json',
    #                  # # # restaurant Pitt
    #                  # '../SimDial/1500_data/rest_pitt-MixSpec-1500-DB.json',
    #                  # restaurant style
    #                  # '../SimDial/1500_data/restaurant_style-MixSpec-1500-DB.json'
    #                 ]

    #     self.entity = [
    #                      # # # # bus
    #                      # '../SimDial/1500_data/bus-MixSpec-1500-OTGY.json',
    #                      # # # restaurant
    #                      # '../SimDial/1500_data/restaurant-MixSpec-1500-OTGY.json',
    #                      # # # # weather
    #                      # '../SimDial/1500_data/weather-MixSpec-1500-OTGY.json',
    #                      # # # movie
    #                      # '../SimDial/1500_data/movie-MixSpec-150-OTGY.json',
    #                      # # movie
    #                      '../SimDial/1500_data/movie-MixSpec-1500-OTGY.json',
    #                      # # restaurant Pitt
    #                      # '../SimDial/1500_data/rest_pitt-MixSpec-1500-OTGY.json',
    #                      # restaurant style
    #                      # '../SimDial/1500_data/restaurant_style-MixSpec-1500-OTGY.json'
    #                     ]

    #     # self.model_path = './models/simdial_rest.pkl'
    #     # self.result_path = './results/simdial_rest.csv'

    #     # self.model_path = './models/rest_weat_bus3.pkl'
    #     # # self.result_path = './results/rest_weat_bus3.csv'
    #     # self.result_path = './results/rest_weat_bus3-m.csv'
    #     # # self.result_path = './results/rest_weat_bus3-r.csv'
    #     # # self.result_path = './results/rest_weat_bus3-r_slot.csv'
    #     # # self.result_path = './results/rest_weat_bus3-r_style.csv'

    #     # self.model_path = './models/rest_weat_bus3_adaptm.pkl'
    #     # self.result_path = './results/rest_weat_bus3_adaptm-m.csv'

    #     # self.model_path = './models/rest_weat_bus2.pkl'
    #     # # self.result_path = './results/rest_weat_bus2.csv'
    #     # self.result_path = './results/rest_weat_bus2-m.csv'
    #     # # self.result_path = './results/rest_weat_bus2-r.csv'
    #     # # self.result_path = './results/rest_weat_bus2-r_slot.csv'
    #     # # self.result_path = './results/rest_weat_bus2-r_style.csv'

    #     # self.model_path = './models/rest_weat_bus2_adaptr.pkl'
    #     # self.result_path = './results/rest_weat_bus2_adaptr-r.csv'
    #     # # self.model_path = './models/rest_weat_bus2_adaptm.pkl'
    #     # # self.result_path = './results/rest_weat_bus2_adaptm-m.csv'
    #     # # self.model_path = './models/rest_weat_bus2_adaptrslot.pkl'
    #     # # self.result_path = './results/rest_weat_bus2_adaptrslot-r_slot.csv'
    #     # # self.model_path = './models/rest_weat_bus2_adaptrstyle.pkl'
    #     # # self.result_path = './results/rest_weat_bus2_adaptrstyle-r_style.csv'

    #     # self.model_path = './models/rest_weat_bus_adaptresstyle.pkl'
    #     # # self.result_path = './results/rest_weat_bus_adaptres.csv'
    #     # self.result_path = './results/rest_weat_bus_adaptresstyle-r_style.csv'

    #     # self.model_path = './models/rest_weat_bus_littlemovie.pkl'
    #     # # self.result_path = './results/rest_weat_bus_littlemovie.csv'
    #     # self.result_path = './results/rest_weat_bus_littlemovie-rest.csv'
    #     # # self.result_path = './results/rest_weat_bus_littlemovie-movie.csv'
    #     # # self.result_path = './results/rest_weat_bus_littlemovie-rest_slot.csv'
    #     # # self.result_path = './results/rest_weat_bus_littlemovie-rest_style.csv'

    #     # self.model_path = './models/rest_weat_bus_littlemovie_adaptmov.pkl'
    #     # # self.result_path = './results/rest_weat_bus_littlemovie_adaptmov.csv'
    #     # self.result_path = './results/rest_weat_bus_littlemovie_adaptmov-m.csv'
    #     # # self.result_path = './results/rest_weat_bus_littlemovie_adaptmov-r.csv'

    #     # self.model_path = './models/r_w_b_littlemx10.pkl'
    #     # # self.result_path = './results/r_w_b_littlemx10.csv'
    #     # # self.result_path = './results/r_w_b_littlemx10-r.csv'
    #     # self.result_path = './results/r_w_b_littlemx10-m.csv'
    #     # # self.result_path = './results/r_w_b_littlemx10-r_slot.csv'
    #     # self.result_path = './results/r_w_b_littlemx10-r_style.csv'

    #     # # # self.model_path = './models/r_w_b_littlemx10_adaptr.pkl'
    #     # # # self.result_path = './results/r_w_b_littlemx10_adaptr-r.csv'
    #     self.model_path = './models/r_w_b_littlemx10_adaptm.pkl'
    #     self.result_path = './results/r_w_b_littlemx10_adaptm-m-tmp.csv'
    #     # # # self.model_path = './models/r_w_b_littlemx10_adaptrslot.pkl'
    #     # # # self.result_path = './results/r_w_b_littlemx10_adaptrslot-r_slot.csv'
    #     # # self.model_path = './models/r_w_b_littlemx10_adaptrstyle.pkl'
    #     # # self.result_path = './results/r_w_b_littlemx10_adaptrstyle-r_style.csv'

    #     # self.model_path = './models/r_w_b_littlemx10_adaptr2.pkl'
    #     # self.result_path = './results/r_w_b_littlemx10_adaptr2-r.csv'

    #     # self.model_path = './models/r_w_b_littlemx10_adaptr_1grad.pkl'
    #     # self.result_path = './results/r_w_b_littlemx10_adaptr_1grad-r.csv'

    #     self.glove_path = './data/glove/glove.6B.50d.txt'
    #     self.batch_size = 32
    #     self.z_length = 8
    #     self.degree_size = 5
    #     self.layer_num = 1
    #     self.dropout_rate = 0.5
    #     self.epoch_num = 100 # triggered by early stop
    #     self.rl_epoch_num = 2
    #     self.cuda = True
    #     self.spv_proportion = 100
    #     self.max_ts = 40
    #     self.early_stop_count = 5
    #     self.new_vocab = True



    #     self.teacher_force = 100
    #     self.beam_search = False
    #     self.beam_size = 10
    #     self.sampling = False
    #     self.unfrz_attn_epoch = 0
    #     self.skip_unsup = False
    #     self.truncated = False
    #     self.pretrain = False

    # def _camrest_tsdf_init(self):
    #     self.beam_len_bonus = 0.5
    #     self.prev_z_method = 'separate'
    #     self.vocab_size = 800
    #     self.embedding_size = 50
    #     self.hidden_size = 50
    #     self.split = (3, 1, 1)
    #     self.lr = 0.003
    #     self.lr_decay = 0.5

    #     self.enlarge_vocab = False

    #     self.vocab_path = './vocab/vocab-camrest.pkl'
    #     self.data = './data/CamRest676/CamRest676.json'
    #     self.entity = './data/CamRest676/CamRestOTGY.json'
    #     self.db = './data/CamRest676/CamRestDB.json'

    #     self.glove_path = './data/glove/glove.6B.50d.txt'
    #     self.batch_size = 32
    #     self.z_length = 8
    #     self.degree_size = 5
    #     self.layer_num = 1
    #     self.dropout_rate = 0.5
    #     self.epoch_num = 100 # triggered by early stop
    #     self.rl_epoch_num = 2
    #     self.cuda = True
    #     self.spv_proportion = 100
    #     self.max_ts = 40
    #     self.early_stop_count = 3
    #     self.new_vocab = True
    #     self.model_path = './models/camrest.pkl'
    #     self.result_path = './results/camrest-rl.csv'

    #     self.teacher_force = 100
    #     self.beam_search = False
    #     self.beam_size = 10
    #     self.sampling = False
    #     self.unfrz_attn_epoch = 0
    #     self.skip_unsup = False
    #     self.truncated = False
    #     self.pretrain = False

    def _kvret_tsdf_init(self):
        self.prev_z_method = 'separate'
        self.intent = 'all'
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = None
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-kvret.pkl'
        self.train = './data/kvret/kvret_train_public.json'
        self.dev = './data/kvret/kvret_dev_public.json'
        self.test = './data/kvret/kvret_test_public.json'
        self.entity = './data/kvret/kvret_entities.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.degree_size = 5
        self.z_length = 8
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.cuda = True
        self.spv_proportion = 100
        self.alpha = 0.0
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        # self.model_path = './models/kvret.pkl'
        # self.result_path = './results/kvret.csv'

        self.model_path = './models/fixed_r_w_b3.pkl'
        self.result_path = './results/fixed_r_w_b3.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False
        self.oov_proportion = 100

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

global_config = _Config()

