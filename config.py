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

        self.model_path = './models/fixed'+str(idx)+'_rwb'+str(size)+'_mstep'+str(self.maml_step)+'.pkl'
        self.result_path = './results/fixed'+str(idx)+'_rwb'+str(size)+'_mstep'+str(self.maml_step)+'.csv'

        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False


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

