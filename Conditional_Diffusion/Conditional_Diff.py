#self package
from Utils.BuildModel2 import *
from Utils.network_utils import *
from Utils.Unet_utils import *
from Utils.utils import *
import pickle as pickle
from datetime import datetime

def one_hot_MPT(y,depth):
# #one-hot
  y=tf.one_hot(y,depth=depth,dtype=tf.int32)
  # y=y[:,0:3]# 前面一列删除
  return y

#self difined
def train_on_step(image_batch,y):
    diffusion_loss=train_loss(image_batch,y)
    return diffusion_loss

def train_loss(image_batch,y):#-1,134,1
    t = tf.random.uniform(minval=0, maxval=timesteps, shape=(image_batch.shape[0],), dtype=tf.int64)
    with tf.GradientTape() as tape:
        noise=tf.random.normal(shape=tf.shape(image_batch),dtype=image_batch.dtype)
        image_noise=gdf_util.q_sample(image_batch,t,noise)
        pred_noise = ddpm([image_noise, t,y], training=True)
        difusion_loss = mseloss(noise, pred_noise[:,:1024])
    gradients = tape.gradient(difusion_loss,
                              ddpm.trainable_weights)
    opt.apply_gradients(zip(gradients, ddpm.trainable_weights))
    return difusion_loss.numpy()

def generate_images_condition(y,num_images=1000):
    y_generate = tf.one_hot(np.repeat(y,num_images,axis=0), depth=6, dtype=tf.int32)
    #
    # # 1. Randomly sample noise (starting point for reverse process)
    samples = tf.random.normal(
        shape=(num_images, img_size), dtype=tf.float32
    )
    # # # 2. Sample from the model iteratively
    for t in reversed(range(0, timesteps)):
        tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        pred_noise = ddpm.predict([samples, tt,y_generate], verbose=0, batch_size=num_images)
        pred_noise = pred_noise[:,:1024]
        samples = gdf_util.p_sample(
            pred_noise, samples, tt, clip_denoised=True
        )


    # Return generated samples
    return samples

# data input
batch_size=32
num_epochs=10000
timesteps = 1000
norm_groups=8
learning_rate=2e-4
img_size=1024
# img_channels=1
first_conv_channels=16
channel_multipier=[4,2,1,1/2]
widths=[first_conv_channels* mult for mult in channel_multipier]
has_attention=[False,False,False,False]
num_res_blocks = 2

#build dataset x and y
types=['A','E','F','G','K','L']
dataset=[]
for type in types:
    path = './Data_GS/S10/S3/S10_GSS3_'+ type + '.pkl'
    with open(path, 'rb') as f:
        data_train,_ = pickle.load(f)
    data_train=data_train.astype(np.float32)
    dataset.extend(data_train)
dataset=np.array(dataset)
num=10
dataset_y=np.array([0]*num+[1]*num+[2]*num+[3]*num+[4]*num+[5]*num)
train_ds=tf.data.Dataset.from_tensor_slices((dataset,dataset_y)).shuffle(675).batch(batch_size)

label_dim=len(types)
#
# # Build model
image_input = layers.Input(shape=( img_size), name="generator_input") #128 1
time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")  # *(None,)
label_onehot = keras.Input(shape=(label_dim,))

ddpm_x = build_model(
    input=image_input,
    time_input=time_input,
    label_input=label_onehot,
    widths=widths,
    has_attention=has_attention,
    first_conv_channels=first_conv_channels,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)

ddpm=keras.Model([image_input,time_input,label_onehot],ddpm_x)

gdf_util = GaussianDiffusion(timesteps=timesteps)
opt=keras.optimizers.Adam(learning_rate=learning_rate)
mseloss = keras.losses.MeanSquaredError()
# print('Network Summary-->')
# ddpm.summary()

start_time = datetime.now()
print(f"开始时间: {start_time}")

# Training
type_loss_dict = {type: [] for type in types}  # Initialize a dictionary to hold loss for each type

for epoch in range(num_epochs):
    epoch_loss = {type: 0 for type in types}  # Initialize epoch loss for each type
    total_samples = {type: 0 for type in types}  # Initialize counter for each type's samples

    for (images_batch, y) in train_ds:
        data_y = one_hot_MPT(y, len(types))
        diffusion_loss = train_on_step(images_batch, data_y)

        # 计算每个类型的损失
        for i, label in enumerate(y):
            label_type = types[label]
            epoch_loss[label_type] += diffusion_loss
            total_samples[label_type] += 1  # Increment the counter for this label type

    # 计算每个类型的平均损失并记录
    for label_type in types:
        average_loss = epoch_loss[label_type] / total_samples[label_type]
        type_loss_dict[label_type].append(average_loss)

    if epoch % 200 == 0:
        # Save the model weights
        save_dir = './Data_GS/Gen_condition_1024/cmodels/model_'+'all'+'/'
        os.makedirs(save_dir, exist_ok=True)
        ddpm.save_weights(save_dir + 'cmodel_' + str(epoch) + '.ckpt')

    print(f'epoch {epoch}, diffusion loss: {diffusion_loss}')

# 保存每个type的loss值
for label_type in types:
    type_loss_list = np.array(type_loss_dict[label_type])  # Get the loss list for the current type
    np.savetxt(f'./Data_GS/Gen_condition_1024/loss_S3_{label_type}.txt', type_loss_list)
#
# # Save the trained model weights (last step)
save_dir = './Data_GS/Gen_condition_1024/cmodels/model_'+'all'+'/'
ddpm.save_weights(save_dir + 'cmodel_last_' + str(epoch) + '.ckpt')

# # # Load the model weights
dir='./Data_GS/Gen_condition_1024/cmodels/model_all/cmodel_last_9999.ckpt'
print('Load weights from ',dir)
ddpm.load_weights(dir)
print('load weights successfully!!!')
# ddpm.summary()
#
# # Generate the data from trained model
print('start generate')

# Save the generated data to the directory
import os

# Ensure the directory exists
os.makedirs('.\\Data_GS\\Gen_condition_1024', exist_ok=True)

def save_generated_data(y, samples, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)
        print(f"Generated data for label {y} saved successfully at {file_path}.")
    except Exception as e:
        print(f"Error saving generated data for label {y}: {e}")

def generate_and_save(y, num_images, save_path):
    samples = generate_images_condition(y, num_images=num_images)
    save_generated_data(y, samples, save_path)

# 数据生成与保存
labels = [0, 1, 2, 3, 4, 5]
file_paths = [
    f'./Data_GS/Gen_condition_1024/Gen_GSS3_{label}.pkl'
    for label in ['A', 'E', 'F', 'G', 'K', 'L']
]
for y, file_path in zip(labels, file_paths):
    generate_and_save(np.array([y]), 1000, file_path)


end_time = datetime.now()
print(f"结束时间: {end_time}")
elapsed_time = end_time - start_time
print(f"代码运行时长: {elapsed_time}")



