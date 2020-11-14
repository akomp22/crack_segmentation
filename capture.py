import cv2
from datetime import datetime
import time
from datetime import timedelta 
import os
import numpy as np
import tkinter as tk


master=tk.Tk()
e=tk.Entry(master)
e.pack()

def callback():
    global time_interval
    time_interval=int(e.get())
    master.destroy()
    
b=tk.Button(master, text='Ok',width=10,command=callback)
b.pack()
tk.mainloop()



cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

#print(cam0.get(3),cam0.get(4))
#print(cam1.get(3),cam1.get(4))
cam0.set(3, 40000)
cam0.set(4, 40000)

cam1.set(3, 40000)
cam1.set(4, 40000)

#print(cam0.get(3),cam0.get(4))
#print(cam1.get(3),cam1.get(4))

a=1
nowo=datetime.now()
b=False
while True: 
    
    
    ret_val0, img0 = cam0.read()
    ret_val1, img1 = cam1.read()

    cv2.imshow('And said God let there be light, and there was light', img0)
    cv2.imshow('What goes up must come down. Isaac Newton ', img1)
    key=cv2.waitKey(1)
    
    nown=datetime.now()
    if key==ord('p'):
        b=True
    if key==ord('s'):
        b=False
    
    if  b==True:
        nowc=nowo+timedelta(seconds=time_interval*a)
        if nowc<=nown:

            current_timen=datetime.now()
            name1=time.strftime("%m-%d-%Y  %H;%M;%S")
            name00=name1+'.png'
            name11='2  '+name1+'.png'
            cv2.imwrite(name00, img0)
            cv2.imwrite(name11, img1)
                
    

    
    
                
            a=a+1
            
            
    
    if key==ord('q'):
        break
    
cam0.release()
cam1.release()
cv2.destroyWindow('And said God let there be light, and there was light')
cv2.destroyWindow('What goes up must come down. Isaac Newton ')