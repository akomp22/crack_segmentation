from tensorflow.keras.preprocessing.image import ImageDataGenerator



def creat_train_data_frame(dir_mask = '/content/drive/My Drive/ML/Skynet keypoint/masks/',\
                           dir_im = '/content/drive/My Drive/ML/Skynet keypoint/initial photo for train/'):
    names = []
    for file in os.listdir(dir_mask):
        names.append(file)

    mask_read = []
    im_read = []
    for i in range(len(names)):
        mask_read.append(os.path.join(dir_mask, names[i]))
        im_read.append(os.path.join(dir_im, names[i]))



    im_df = pd.DataFrame(im_read,columns = ['image path'])
    mask_df = pd.DataFrame(mask_read,columns = ['mask path'])

    df = pd.DataFrame(data = np.column_stack([im_df,mask_df]),columns =  ['image path','mask path'])

    return(df)




def val_name_list(dir_validation = '/content/drive/My Drive/ML/Skynet keypoint/validation/'):

    val_names = []
    for file in os.listdir(dir_validation):
        val_names.append(file)

    val_read = []
    for i in range(len(val_names)):
        val_read.append(os.path.join(dir_validation, val_names[i]))
    
    return(val_read)




def disp(im,mask):
  mask = np.concatenate((mask,mask,mask),axis = 3)
  im2 = im.copy()
  im2[mask==0] = im2[mask==0]-0.2
  im2[mask==1] = im2[mask==1]+0.3


  im = im[0,:,:,:]
  mask = mask[0,:,:,:]
  im2 = im2[0,:,:,:]
  
  plt.figure(figsize=(20,20))
  plt.imshow(im2)
  plt.show()









def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img / 255
        w = img.shape[2]
        w1 = int(w*0.3)
        w2 = int(w*0.7)
        img = img[:,:,w1:w2,:]
        mask = mask[:,:,w1:w2,:]
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)





def trainGenerator(batch_size,df,aug_dict,image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (1080,1920),seed = 1):

    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    

    image_generator = image_datagen.flow_from_dataframe(
        df, 
        directory=None, 
        x_col='image path', 
        y_col=None,
        target_size=target_size, 
        color_mode='rgb',
        class_mode=None, 
        batch_size=batch_size, 
        seed=seed,
        save_to_dir=save_to_dir, 
        save_prefix='image', )
    



    mask_generator = image_datagen.flow_from_dataframe(
        df, 
        directory=None, 
        x_col='mask path', 
        y_col=None,
        target_size=target_size, 
        color_mode='grayscale',
        class_mode=None, 
        batch_size=batch_size, 
        seed=seed,
        save_to_dir=save_to_dir, 
        save_prefix='mask', )




    train_generator = zip(image_generator, mask_generator)
    

    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)
