

from __future__ import absolute_import
import numpy as np
from glob import glob
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Subtract,Input, Dense, Conv2D, concatenate,Layer,Dense, Dropout, LayerNormalization,Embedding
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.activations import softmax
from tensorflow.image import extract_patches
import tensorflow.keras.backend as K
from Swin_Transformer import *
def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0.2 # Droupout after each MLP layer
    attn_drop_rate = 0.2 # Dropout after Swin-Attention
    proj_drop_rate = 0.2 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0.2 # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = SwinTransformerBlock(dim=embed_dim, 
                                             num_patch=num_patch, 
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             shift_size=shift_size_temp, 
                                             num_mlp=num_mlp, 
                                             qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate, 
                                             attn_drop=attn_drop_rate, 
                                             proj_drop=proj_drop_rate, 
                                             drop_path_prob=drop_path_rate, 
                                             name='name{}'.format(i))(X)
    return X


def swin_unet_2d_base(input_tensor,input_tensor1, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):
    '''
    The base of Swin-UNET.
    
    The general structure:
    
    1. Input images --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head
    
    '''
    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = 64//patch_size[0]
    num_patch_y = 64//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    conv11=Conv2D(filters=64, kernel_size=(7,7), kernel_initializer="he_normal",
               padding="same")(X)
    b11=BatchNormalization()(conv11)
    b11 = Activation("relu")(b11)
    b11 = MaxPooling2D((2,2),strides=(2,2))(b11)
    # Patch extraction
    X = patch_extract(patch_size)(b11)
    
    # Embed patches to tokens
    X = patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, 
                               stack_num=stack_num_down, 
                               embed_dim=embed_dim, 
                               num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], 
                               window_size=window_size[0], 
                               num_mlp=num_mlp, 
                               shift_window=shift_window, 
                               name='{}_swin_down0'.format(name))
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_down, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], 
                                   window_size=window_size[i+1], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_down{}'.format(name, i+1))
        # Store tensors for concat
        X_skip.append(X)

    # input_size1 = input_tensor1.shape.as_list()[1:]
    # num_patch_x1 = input_size1[0]//patch_size[0]
    # num_patch_y1 = input_size1[1]//patch_size[1]
    input_size1 = input_tensor1.shape.as_list()[1:]
    num_patch_x1 = 64//patch_size[0]
    num_patch_y1 = 64//patch_size[1]
    # Number of Embedded dimensions
    embed_dim1 = filter_num_begin
    
    depth1_ = depth

    X1_skip = []

    X1 = input_tensor1
    conv1=Conv2D(filters=64, kernel_size=(7,7), kernel_initializer="he_normal",
               padding="same")(X1)
    b1=BatchNormalization()(conv1)
    b1 = Activation("relu")(b1)
    b1 = MaxPooling2D((2,2),strides=(2,2))(b1)
    # Patch extraction
    X1 = patch_extract(patch_size)(b1)
    # Embed patches to tokens
    X1 = patch_embedding(num_patch_x1*num_patch_y1, embed_dim1)(X1)
    
    # The first Swin Transformer stack
    X1 = swin_transformer_stack(X1, 
                               stack_num=stack_num_down, 
                               embed_dim=embed_dim1, 
                               num_patch=(num_patch_x1, num_patch_y1), 
                               num_heads=num_heads[0], 
                               window_size=window_size[0], 
                               num_mlp=num_mlp, 
                               shift_window=shift_window, 
                               name='{}_swin_down1'.format(name))
    X1_skip.append(X1)
    
    # Downsampling blocks
    for i in range(depth1_-1):
        
        # Patch merging
        X1 = patch_merging((num_patch_x1, num_patch_y1), embed_dim=embed_dim1, name='down1{}'.format(i))(X1)
        
        # update token shape info
        embed_dim1 = embed_dim1*2
        num_patch_x1 = num_patch_x1//2
        num_patch_y1 = num_patch_y1//2
        
        # Swin Transformer stacks
        X1 = swin_transformer_stack(X1, 
                                   stack_num=stack_num_down, 
                                   embed_dim=embed_dim1, 
                                   num_patch=(num_patch_x1, num_patch_y1), 
                                   num_heads=num_heads[i+1], 
                                   window_size=window_size[i+1], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_down3{}'.format(name, i+1))
        # Store tensors for concat
        X1_skip.append(X1)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    X1_skip = X1_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    X1 = X1_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    X1_decode = X1_skip[1:]
    depth_decode = len(X_decode)
    

    
    for i in range(depth_decode):

        # Patch expanding
        X = patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                  embed_dim=embed_dim, 
                                  upsample_rate=2, 
                                  return_vector=True)(X)
        
        X1 = patch_expanding(num_patch=(num_patch_x1, num_patch_y1), 
                                  embed_dim=embed_dim1, 
                                  upsample_rate=2, 
                                  return_vector=True)(X1)
        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2

        embed_dim1 = embed_dim1//2
        num_patch_x1 = num_patch_x1*2
        num_patch_y1 = num_patch_y1*2
        
    
        X = concatenate([X,X_decode[i],X1_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_up, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i], 
                                   window_size=window_size[i], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_up{}'.format(name, i))
        X1 = concatenate([X1,X1_decode[i],X_decode[i]], axis=-1, name='{}_concat1_{}'.format(name, i))
        X1 = Dense(embed_dim1, use_bias=False, name='{}_concat_linear_proj1_{}'.format(name, i))(X1)
        # Swin Transformer stacks
        X1 = swin_transformer_stack(X1, 
                                   stack_num=stack_num_up, 
                                   embed_dim=embed_dim1, 
                                   num_patch=(num_patch_x1, num_patch_y1), 
                                   num_heads=num_heads[i], 
                                   window_size=window_size[i], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_up1{}'.format(name, i))
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    
    X = patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                        embed_dim=embed_dim, 
                        upsample_rate=patch_size[0], 
                        return_vector=False)(X)

    X1 = patch_expanding(num_patch=(num_patch_x1, num_patch_y1), 
                        embed_dim=embed_dim1, 
                        upsample_rate=patch_size[0], 
                        return_vector=False)(X1)

    Xf=tf.abs(Subtract()([X,X1]))
    Xf1=tf.abs(Subtract()([b11,b1]))
    up11= keras.layers.UpSampling2D(size=(1,1), data_format=None, interpolation='bilinear')(Xf)
    up11 = concatenate([up11,Xf1])
    up12=Conv2DTranspose(64, (7, 7),  padding='same')(up11)
    up13=BatchNormalization()(up12)
    up13 = Activation("relu")(up13)
    up11= keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(up13)
    

    return up11

filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
depth =4                 # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
stack_num_down = 2         # number of Swin Transformers per downsampling level
stack_num_up = 2           # number of Swin Transformers per upsampling level
patch_size = (4,4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
num_mlp = 512              # number of MLP nodes within the Transformer
shift_window=True          # Apply window shifting, i.e., Swin-MSA

# Input section
input_size = (128, 128, 3)
IN = Input(input_size)
input_size1 = (128, 128, 3)
IN1 = Input(input_size1)

# Base architecture
X = swin_unet_2d_base(IN,IN1, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, 
                      shift_window=shift_window, name='swin_unet')

# Output section
n_labels = 1
OUT = Conv2D(n_labels, kernel_size=1, activation='sigmoid')(X)

# Model configuration
model = Model(inputs=[IN,IN1], outputs=[OUT])


