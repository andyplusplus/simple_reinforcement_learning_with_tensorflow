import os
import tensorflow as tf


load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
saver = None

def get_saver():
    global saver
    if saver == None:
        saver = tf.train.Saver()
    return saver

#Make a path for our model to be saved in.
def load_model(sess):
    saver = get_saver()
    ckpt = None
    if not os.path.exists(path):
        os.makedirs(path)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    return ckpt

def save_model_i(sess, i):
    saver.save(sess, path+'/model-'+str(i)+'.ckpt')
    print("Saved Model")

