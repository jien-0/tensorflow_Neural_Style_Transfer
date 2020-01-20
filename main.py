
# coding: utf-8

# In[2]:


import tensorflow as tf
import os
import image
import model
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
content_path = 'input/content/a.jpg'
style_path = 'input/style/timg.jpg'

# content = image.loadimg(content_path).astype('uint8')
# style = image.loadimg(style_path).astype('uint8')

# Content layer where will pull our feature maps
content_layers = ['block4_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]
# i,j = 0,0
# for c_img in os.listdir('input/content/'):
#     c_path = os.path.join('input/content/',c_img)
#     os.makedirs(f'output/{i}/')
    
#     for s_img in os.listdir('input/style/'):

#         s_path = os.path.join('input/style/',s_img)
#         best, best_loss = model.run_nst(c_path,s_path,iteration=1000)
#         image.saveimg(best, f'output/{i}/{j}.jpg')
#         j += 1
#     i += 1


if __name__ == "__main__":
    best, best_loss = model.run_nst(content_path, style_path, iteration=1000)
    image.saveimg(best, 'output/output10.jpg')
