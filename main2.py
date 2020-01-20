# -*- coding: utf-8 -*-
import model
content_path = r'E:\NST\input\content\2234.jpg'
style_path = r'E:\NST\input\style\timg.jpg'
best_img, best_loss = model.run_nst(content_path,style_path,iteration = 1000,content_weight = 1e3,style_weight = 1)
print(type(best_img))
print(type(best_loss))
import scipy.misc
scipy.misc.imsave('outfile2.jpg', best_img)