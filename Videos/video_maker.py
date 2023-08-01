# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:56:08 2023

@author: Pierre PAJUELO
@subject: Create videos
"""
# Importing modules
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Defining functions


#Principal program
if __name__=='__main__':
    # PARAMETERS
    folder = 'D:/Documents/GitHub/chitosangels/Videos'
    days = [f.path for f in os.scandir(folder) if f.is_dir()]# if os.path.isdir(f)]
    
    plt.close('all')
    # for day in days:
    day = days[5]
    first_pictures = [f for f in os.listdir(day) if "0000" in f]
    total_number = len(os.listdir(day))
    index_file = 0
    # for first in first_pictures:
    first = first_pictures[0]
    images = []
    index = first.index("0000")
    time = first[index+14:index+22]
    hour = int(time[0:2])
    minute = int(time[3:5])
    index_number = 0
    search = True
    while search:
        images.append(first)
        if minute==59:
            hour+=1
            minute=0
        else:
            minute+=1
        index_number += 1 
        # print(first[:index+1] + "%03d"%(index_number) + first[index+4:index+14] + str(hour)+'-'+"%02d"%(minute)+"*")
        search_file = [f for f in os.listdir(day) if first[:index+1] + "%03d"%(index_number) + first[index+4:index+14] + "%02d"%(hour)+'-'+"%02d"%(minute) in f]
        # print(search_file)
        if len(search_file)==0:
            search=False
        else:
            first = search_file[0]
    snapshots = [cv2.imread(day+"/"+images[j], cv2.IMREAD_GRAYSCALE) for j in range(len(images))]
    a = snapshots[0]
    fig, ax = plt.subplots()
    def animate_func(i):        
        im.set_array(snapshots[i])
        txt.set_text('Image %s/%s'%(i,len(images)))
        return [im,txt]
    im = ax.imshow(a,cmap='gray')
    txt = ax.text(0.1,0.9,'Image %s/%s'%(0,len(images)),transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.5))
    anim = animation.FuncAnimation(fig, 
                                animate_func, 
                                frames=len(images), 
                                interval=1)
    anim.save(folder+'/%s.gif'%first, writer='pillow')