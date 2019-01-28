import logging
import time
import configparser

class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 7       
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
    def _camrest_tsdf_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = (3, 1, 1)
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-simdial.pkl'
        self.data = [
                     # camrest
                     './data/CamRest676/CamRest676.json',
                     # restaurant
                     '../SimDial/data/restaurant-CleanSpec-1000.json',
                     # # restaurant noised
                     # '../SimDial/data/restaurant-MixSpec-1000.json',
                     # # bus
                     # '../SimDial/data/bus-MixSpec-1000.json',
                     # # weather
                     # '../SimDial/data/weather-MixSpec-1000.json',
                     # # movie
                     # '../SimDial/data/movie-MixSpec-1000.json',


                     # # restaurant Pitt
                     # '../SimDial/data/rest_pitt-MixSpec-1000.json',
                     # # restaurant style
                     # '../SimDial/data/restaurant_style-MixSpec-1000.json'
                    ]
        # 
        self.db = [
                     # camrest
                     './data/CamRest676/CamRestDB.json',
                     # restaurant
                     '../SimDial/data/restaurant-CleanSpec-1000-DB.json',
                     # restaurant noised
                     '../SimDial/data/restaurant-MixSpec-1000-DB.json',
                     # bus
                     '../SimDial/data/bus-MixSpec-1000-DB.json',
                     # weather
                     '../SimDial/data/weather-MixSpec-1000-DB.json',
                     # movie
                     '../SimDial/data/movie-MixSpec-1000-DB.json',
                     # restaurant Pitt
                     '../SimDial/data/rest_pitt-MixSpec-1000-DB.json',
                     # restaurant style
                     '../SimDial/data/restaurant_style-MixSpec-1000-DB.json'
                    ]

        # self.entity = './data/CamRest676/CamRestOTGY.json'
        # self.entity = '../SimDial/train/simdialOTGY.json'
        self.entity = [
                         # camrest
                         './data/CamRest676/CamRestOTGY.json',
                         # restaurant
                         '../SimDial/data/restaurant-CleanSpec-1000-OTGY.json',
                         # restaurant noised
                         '../SimDial/data/restaurant-MixSpec-1000-OTGY.json',
                         # bus
                         '../SimDial/data/bus-MixSpec-1000-OTGY.json',
                         # weather
                         '../SimDial/data/weather-MixSpec-1000-OTGY.json',
                         # movie
                         '../SimDial/data/movie-MixSpec-1000-OTGY.json',
                         # restaurant Pitt
                         '../SimDial/data/rest_pitt-MixSpec-1000-OTGY.json',
                         # restaurant style
                         '../SimDial/data/restaurant_style-MixSpec-1000-OTGY.json'
                        ]

        # # # added data for maml
        # # restaurant
        # self.data_sim1 = '../SimDial/data/restaurant-MixSpec-1000.json'
        # self.db_sim1 = '../SimDial/data/restaurant-MixSpec-1000-DB.json'

        # # restaurant style
        # self.data_sim2 = '../SimDial/data/restaurant_style-MixSpec-1000.json'
        # self.db_sim2 = '../SimDial/data/restaurant_style-MixSpec-1000-DB.json'

        # # bus
        # self.data_sim3 = '../SimDial/data/bus-MixSpec-1000.json'
        # self.db_sim3 = '../SimDial/data/bus-MixSpec-1000-DB.json'

        # # weather
        # self.data_sim4 = '../SimDial/data/weather-MixSpec-1000.json'
        # self.db_sim4 = '../SimDial/data/weather-MixSpec-1000-DB.json'

        # # movie
        # self.data_sim5 = '../SimDial/data/movie-MixSpec-1000.json'
        # self.db_sim5 = '../SimDial/data/movie-MixSpec-1000-DB.json'

        # # restaurant Pitt
        # self.data_sim6 = '../SimDial/data/rest_pitt-MixSpec-1000.json'
        # self.db_sim6 = '../SimDial/data/rest_pitt-MixSpec-1000-DB.json'


        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.z_length = 8
        self.degree_size = 5
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 2
        self.cuda = True
        self.spv_proportion = 100
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/simdial1.pkl'
        self.result_path = './results/simdial1.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    # def _camrest_tsdf_init(self):
    #     self.beam_len_bonus = 0.5
    #     self.prev_z_method = 'separate'
    #     self.vocab_size = 800
    #     self.embedding_size = 50
    #     self.hidden_size = 50
    #     self.split = (3, 1, 1)
    #     self.lr = 0.003
    #     self.lr_decay = 0.5
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
        self.vocab_size = 1400
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
        self.cuda = False
        self.spv_proportion = 100
        self.alpha = 0.0
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/kvret.pkl'
        self.result_path = './results/kvret.csv'
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

