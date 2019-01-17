import os

path = "./dqn" #The path to save our model to.
#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
