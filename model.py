from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,MaxPool2D,Dropout
from tensorflow.keras.layers import Cropping2D,ZeroPadding2D,BatchNormalization,Concatenate,Flatten,Dense
import tensorflow as tf



def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = Concatenate([conv_1x1, conv_3x3,conv_5x5,pool_proj])
    
    return output


def get_crop_shape(target,refer):
    w = target.get_shape()[2] - refer.get_shape()[2]
    assert( w>=0)
    if w%2!=0:
        w1,w2 = int(w/2),int(w/2+1)
    else :
        w1,w2 = int(w/2),int(w/2)

    h = target.get_shape()[1] - refer.get_shape()[1]
    assert( h>=0)
    if h%2!=0:
        h1,h2 = int(h/2),int(h/2+1)
    else :
        h1,h2 = int(h/2),int(h/2)



    return(h1,h2),(w1,w2)






def crop_concat(layer,skip_con):
    h,w = get_crop_shape(skip_con,layer)
    skip_con_crop = Cropping2D(cropping=(h,w))(skip_con)
    out = Concatenate([layer, skip_con_crop])
    return out







def unet(im_size = (1080, 768, 3)):

    inputs = Input(shape=im_size)


    conv1 = Conv2D(64, 7, activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(64, 7, activation = 'relu', padding = 'same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(3, 3))(conv1)
    #pool1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(64, 5, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(64, 5, activation = 'relu', padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D(pool_size=(3, 3))(conv2)
    #pool2 = Dropout(0.2)(pool2)

    conv3 = inception_module(pool2,32,32,32,32,32,32)
    conv3 = BatchNormalization()(conv3)
    conv3 = inception_module(conv3,64,64,64,64,64,64)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPool2D(pool_size=(3, 3))(conv3)
    #pool3 = Dropout(0.2)(pool3)

    conv4 = inception_module(pool3,64,64,64,64,64,64)
    conv4 = BatchNormalization()(conv4)
    conv4 = inception_module(conv4,128,128,128,128,128,128)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPool2D(pool_size=(3, 3))(conv4)
    #pool4 = Dropout(0.2)(pool4)

    conv5 = inception_module(pool4,128,128,128,128,128,128)
    conv5 = BatchNormalization()(conv5)
    conv5 = inception_module(conv5,256,256,256,256,256,256)
    conv5 = BatchNormalization()(conv5)
    #conv5 = Dropout(0.2)(conv5)
    #pool5 = MaxPool2D(pool_size=(3, 3))(conv5)

    up_conv1 = Conv2DTranspose(512, kernel_size=(3,3), strides=3, padding='valid',activation='relu')(conv5)
    conv4 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(2,2))(conv4)
    conv4 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(3,3))(conv4)
    up_conv1 = crop_concat(up_conv1,conv4)
    up_conv1 = inception_module(up_conv1,256,256,256,256,256,256)
    up_conv1 = BatchNormalization()(up_conv1)
    up_conv1 = inception_module(up_conv1,128,128,128,128,128,128)
    up_conv1 = BatchNormalization()(up_conv1)
    #up_conv1 = Dropout(0.2)(up_conv1)

    up_conv2 = Conv2DTranspose(256, kernel_size=(3,3), strides=3, padding='same',activation='relu')(up_conv1 )
    conv3 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(2,2))(conv3)
    conv3 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(3,3))(conv3)
    up_conv2 = crop_concat(up_conv2,conv3)
    up_conv2 = inception_module(up_conv2,128,128,128,128,128,128)
    up_conv2 = BatchNormalization()(up_conv2)
    up_conv2 = inception_module(up_conv2,64,64,64,64,64,64)
    up_conv2 = BatchNormalization()(up_conv2)
    #up_conv2 = Dropout(0.2)(up_conv2)


    up_conv3 = Conv2DTranspose(128, kernel_size=(3,3), strides=3, padding='same',activation='relu')(up_conv2)
    conv2 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(2,2))(conv2)
    conv2 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(3,3))(conv2)
    up_conv3 = crop_concat(up_conv3,conv2)
    up_conv3 = inception_module(up_conv3,64,64,64,64,64,64)
    up_conv3 = BatchNormalization()(up_conv3)
    up_conv3 = inception_module(up_conv3,64,64,64,64,64,64)
    up_conv3 = BatchNormalization()(up_conv3)
    #up_conv3 = Dropout(0.2)(up_conv3)


    up_conv4 = Conv2DTranspose(64, kernel_size=(3,3), strides=3, padding='same',activation='relu')(up_conv3)
    h,w = get_crop_shape(inputs,up_conv4)
    up_conv4 = ZeroPadding2D(padding=(h,w))(up_conv4)
    h,w = get_crop_shape(conv1,up_conv4)
    conv1 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(2,2))(conv1)
    conv1 = Conv2D(128, kernel_size=(3,3), strides=1, padding='same',activation='relu',dilation_rate=(3,3))(conv1)
    conv1_crop = Cropping2D(cropping=(h,w))(conv1)

    #conv1_crop = inception_module(conv1_crop,64,64,64,64,64,64)
    #conv1_crop = BatchNormalization()(conv1_crop)
    concat = tf.keras.layers.Concatenate()
    up_conv4 = concat([up_conv4, conv1_crop])
    #up_conv4 = tf.matmul(up_conv4,th)
    up_conv4 = inception_module(up_conv4,64,64,64,64,64,64)
    up_conv4 = BatchNormalization()(up_conv4)
    up_conv4 = inception_module(up_conv4,32,32,32,32,32,32)
    up_conv4 = BatchNormalization()(up_conv4)
    #up_conv4 = Dropout(0.2)(up_conv4)



    out = Conv2D(1,kernel_size=3,strides=1,padding='same',activation='sigmoid')(up_conv4)
    #out = tf.matmul(out,th)






    return tf.keras.Model(inputs=inputs, outputs=out )