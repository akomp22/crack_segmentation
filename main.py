from model import *
from data import *
from train_add import *






df =  creat_train_data_frame()
val_read = val_name_list()


data_gen_args = dict(rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.01,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(1,df,data_gen_args,save_to_dir = None)


callback_sch = tf.keras.callbacks.LearningRateScheduler(scheduler)



model = unet()

model.compile(optimizer='adam',loss = weighted_cross_entropy(1))

filepath = '/content/drive/My Drive/ML/Skynet keypoint/checkpoint1/'
save_callback=tf.keras.callbacks.ModelCheckpoint(filepath, monitor='train_loss', verbose=0,
                                                 save_best_only=False,save_weights_only=True, mode='auto', save_freq=1000)



hist = model.fit_generator(myGene,steps_per_epoch=50,epochs=500,callbacks=[save_callback,callback_sch])