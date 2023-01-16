# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:36:02 2021

@author: Ådne
"""
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPool3D, UpSampling3D, Concatenate, Dropout, Conv3DTranspose
from tensorflow.keras.models import Model

# Create model blocks
def conv_block(inputs, filters, maxpool=True, poolZ=True, dropout=0.1):
    x = Conv3D(filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)
    if maxpool == True and poolZ ==True:
        p = MaxPool3D(pool_size=(2,2,2))(x)
        return x, p
    elif maxpool == True and poolZ == False:
        p = MaxPool3D(pool_size=(2,2,1))(x)
        return x, p
    else:
        return x
    
def build_unet(shape, num_classes):
    inputs = Input(shape)
    
    # Encoder
    x1, p1 = conv_block(inputs, 16, maxpool=True, dropout=0.1)
    x2, p2 = conv_block(p1, 32, maxpool=True, dropout=0.1)
    x3, p3 = conv_block(p2, 64, maxpool=True, poolZ = False, dropout=0.2)
    x4, p4 = conv_block(p3, 128, maxpool=True, poolZ = False, dropout=0.2)
    
    # Bridge
    b1 = conv_block(p4, 256, maxpool=False, dropout=0.3)
    
    # Decoder
    u1 = Conv3DTranspose(128, (2,2,2), strides=(2,2,1), padding='same')(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 128, maxpool=False, dropout=0.2)
    
    u2 = Conv3DTranspose(64, (2,2,2), strides=(2,2,1), padding='same')(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 64, maxpool=False, dropout=0.2)
    
    u3 = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding='same')(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, maxpool=False, dropout=0.1)
    
    u4 = Conv3DTranspose(16, (2,2,2), strides=(2,2,2), padding='same')(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, maxpool=False, dropout=0.1)
    
    # Output layer
    output = Conv3D(num_classes, kernel_size = 1, padding="same", activation="softmax")(x8)
    return Model(inputs, output)

if __name__ == '__main__':
    shape = (256, 256, 16, 1)
    num_classes = 3
    model = build_unet(shape, num_classes)
    model.summary()
