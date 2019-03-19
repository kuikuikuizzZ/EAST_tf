import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from keras.utils.data_utils import OrderedEnqueuer
from keras.engine.training_utils import iter_sequence_infinite
import time
import keras

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size', 5, '')
tf.app.flags.DEFINE_integer('workers',1, '')
tf.app.flags.DEFINE_integer('max_queue_size', 40, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../EAST/model/east_font_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_boolean('use_multiprocessing', True, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_string('train_txt_dir', '/clever/dataset/tezign-synth/background_image/merge_h_num4_h_num6_v4/train_txt_images/', '')

import model
import icdar

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))

class EAST_generator(keras.utils.Sequence):
    ''' generator for EAST model'''
    def __init__(self,batch_size,nums,shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nums = list(range(nums))
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.nums)) / self.batch_size)
    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.nums))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_image_files_temp = [self.nums[k] for k in indexes]

        # Generate data
        data = list_image_files_temp
        
        return data

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    print('gpu id',FLAGS.gpu_list)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        print('FLAGS.train_txt_dir',FLAGS.train_txt_dir)
        generator = EAST_generator(batch_size=FLAGS.batch_size,nums=50)
        
        if FLAGS.workers > 0:
            ''' load data with multiprocessing   
            '''
            enqueuer = OrderedEnqueuer(
                generator,
                use_multiprocessing=FLAGS.use_multiprocessing,
                shuffle=True)
            enqueuer.start(workers=FLAGS.workers, max_queue_size=FLAGS.max_queue_size)
            output_generator = enqueuer.get()
            print('workers ',FLAGS.workers)
        else:
            output_generator = iter_sequence_infinite(generator)
                
        start = time.time()
        step_print =FLAGS.save_checkpoint_steps//10
        for epoch in range(3):
            print(generator.indexes)
            num_list = []
            for step in range(30):
                data = next(output_generator)
                insect = set(data) & set(num_list)
                if insect:
                    print(insect)
                num_list.extend(data)
                print('worker',data)
                time.sleep(1)
            generator.on_epoch_end()
            print(generator.indexes)



if __name__ == '__main__':
    tf.app.run()
