
# coding: utf-8

# In[71]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


# In[72]:


img_path = input("Enter the path of the image\n")


# In[73]:


img = cv2.imread(img_path)


# In[75]:


user_input = int(input("Please enter the option number you want to try \n1)Crop the image \n2)Grey scale image \n3)Enhance the image \n4)Sharpen the image \n5)Blur the image \n"))


# In[76]:


save_option = int(input("1)Only display the image\n2)Display and save the image\n"))


# In[77]:


def save_image(img, save_option):
    if save_option == 1:
        cv2.imshow("Target image", img)
    else :
        if user_input==1:
            modified = "Cropped_image"
        elif user_input==2:
            modified = "Gray_image"
        elif user_input==3:
            modified = "reflection_removed_image"
        elif user_input==4:
            modified = "Sharpned_image"
        elif user_input==5:
            modified = "Enhanced_image"
        path_target = input("Enter the path to save the image")
        cv2.imshow("Target image", img)
        path_target = path_target + str("/") + str(modified) + str(".png")
        print(path_target)
        cv2.imwrite(path_target,img)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


# In[78]:


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, refPoint
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            


# In[79]:


cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
oriImage = img.copy()
def crop_img(image, save_option):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
    
    while True:
        i = image.copy()
        if not cropping:
            cv2.imshow("image", image)

        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    if len(refPoint) == 2: #when two points were found
        roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
        save_image(roi, save_option)


# In[80]:


def grey_img(img, save_option):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image(gray, save_option)


# In[117]:


def sharpen_img(img, save_option):
    k = int(input("1)Normal sharping \n2)Excessive sharping\n3)Edge enhancement\n"))
    if k == 1:
        kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    elif k==2:
        kernel_sharpening = np.array([[1,1,1], [1,-7,1], [1,1,1]])
    else:
        kernel_sharpening = np.array([[-1,-1,-1,-1,-1],
                               [-1,2,2,2,-1],
                               [-1,2,8,2,-1],
                               [-2,2,2,2,-1],
                               [-1,-1,-1,-1,-1]])/8.0
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    save_image(img, save_option)


# In[118]:


def blur(img, save_option):
    k = int(input("Enter the level for blur(3/5/7/9/13/15)"))
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    save_image(blurred, save_option)


# In[119]:


def enhance_img(img, save_option):
    gamma = float(input("Enter the level of enhancement of an image(0(dark)-4(bright))"))
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    enhanced_img = cv2.LUT(img, table)
    save_image(enhanced_img,save_option)


# In[121]:


if user_input==1:
    crop_img(img, save_option)
elif user_input==2:
    grey_img(img, save_option)
elif user_input==3:
    blur(img, save_option)
elif user_input==4:
    sharpen_img(img, save_option)
elif user_input==5:
    enhance_img(img, save_option)
else:
    print("Invalid option")

