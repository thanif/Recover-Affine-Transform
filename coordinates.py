# importing the module
import cv2
import numpy as np
# function to display the coordinates of
# of the points clicked on the image

coord1 = []
coord2 = []

def click_event(event, x, y, flags, params):



    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        coord1.append([np.float32(x),np.float32(y)])

        # displaying the coordinates
        # on the image window


        cv2.imshow('image 1', img1)

    # checking for right mouse clicks
    if event==cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        coord1.append([np.float32(x),np.float32(y)])

        # displaying the coordinates
        # on the image window



        cv2.imshow('image 1', img1)

def click_event2(event, x, y, flags, params):



    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        coord2.append([np.float32(x),np.float32(y)])

        # displaying the coordinates
        # on the image window


        cv2.imshow('image 2', img2)

    # checking for right mouse clicks
    if event==cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        coord2.append([np.float32(x),np.float32(y)])

        # displaying the coordinates
        # on the image window

    
        cv2.imshow('image 2', img2)


def complete_inverse_transform(I, T):

    T = np.linalg.inv(T)

    h,w = I.shape[:2]

    if T[0][0] < 2 and T[0][1] < 2 and T[1][0] < 2 and T[1][1] < 2:

        new_height  = round(abs(I.shape[0]*T[0][0])+abs(I.shape[1]*T[1][0]))+1
        new_width  = round(abs(I.shape[1]*T[0][0])+abs(I.shape[0]*T[1][0]))+1

        if len(I.shape) == 3:

            T_I = np.zeros((new_height, new_width, I.shape[2]),dtype='u1')

        else:

            T_I = np.zeros((new_height, new_width),dtype='u1')

        # Find the centre of the image about which we have to rotate the image
        original_centre_height   = round(((I.shape[0]+1)/2)-1)    #with respect to the original image
        original_centre_width    = round(((I.shape[1]+1)/2)-1)    #with respect to the original image

        # Find the centre of the new image that will be obtained
        new_centre_height= round(((new_height+1)/2)-1)        #with respect to the new image
        new_centre_width= round(((new_width+1)/2)-1)          #with respect to the new image

        for i in range(h):
            for j in range(w):

                origin_y = I.shape[0]-1-i-original_centre_height
                origin_x = I.shape[1]-1-j-original_centre_width

                origin_xy = np.array([origin_x,origin_y,1])

                new_xy = np.dot(T,origin_xy)
                new_x = new_xy[0]
                new_y = new_xy[1]

                new_y = new_centre_height-new_y
                new_x = new_centre_width-new_x


                if 0<new_x < w and 0<new_y < h:
                    T_I[int(new_y), int(new_x)]  = get_bilinear_intensity(I,i, j)


    elif T[0][0] >= 0 and T[1][1] >= 0 and T[0][1] == 0 and T[1][0] == 0:

        if len(I.shape) == 3:

            T_I = np.zeros((int(I.shape[0]*T[0][0]), int(I.shape[1]*T[1][1]), I.shape[2]),dtype='u1')

        elif len(I.shape) == 2:

            T_I = np.zeros((int(I.shape[0]*T[0][0]), int(I.shape[1]*T[1][1])),dtype='u1')

        h,w = T_I.shape[:2]

        for i in range(h):
            for j in range(w):
                origin_x = j
                origin_y = i
                origin_xy = np.array([origin_x,origin_y,1])

                new_xy = np.dot(T,origin_xy)
                new_x = new_xy[0]
                new_y = new_xy[1]

                if 0<new_x < w and 0<new_y < h:
                    T_I[int(new_y), int(new_x)]  = get_bilinear_intensity(I,i, j)


    else:

        if len(I.shape) == 3:

            T_I = np.zeros((int(I.shape[0]), int(I.shape[1]), I.shape[2]),dtype='u1')

        elif len(I.shape) == 2:

            T_I = np.zeros((int(I.shape[0]), int(I.shape[1])),dtype='u1')

        h,w = T_I.shape[:2]

        for i in range(h):
            for j in range(w):
                origin_x = j
                origin_y = i
                origin_xy = np.array([origin_x,origin_y,1])

                new_xy = np.dot(T,origin_xy)
                new_x = new_xy[0]
                new_y = new_xy[1]

                if 0<new_x < w and 0<new_y < h:
                    T_I[int(new_y), int(new_x)]  = get_bilinear_intensity(I,i, j)



    return T_I


def transform(I, T):

    h,w = I.shape[:2]

    T_I = np.zeros(I.shape,dtype='u1')

    if T[0][0] < 2 and T[0][1] < 2 and T[1][0] < 2 and T[1][1] < 2:

        new_height  = round(abs(I.shape[0]*T[0][0])+abs(I.shape[1]*T[1][0]))+1
        new_width  = round(abs(I.shape[1]*T[0][0])+abs(I.shape[0]*T[1][0]))+1

        if len(I.shape) == 3:

            T_I = np.zeros(I.shape,dtype='u1')

            #T_I = np.zeros((new_height, new_width, I.shape[2]),dtype='u1')

        else:

            T_I = np.zeros(I.shape,dtype='u1')

            #T_I = np.zeros((new_height, new_width),dtype='u1')

        # Find the centre of the image about which we have to rotate the image
        original_centre_height   = round(((I.shape[0]+1)/2)-1)    #with respect to the original image
        original_centre_width    = round(((I.shape[1]+1)/2)-1)    #with respect to the original image

        # Find the centre of the new image that will be obtained
        new_centre_height= round(((new_height+1)/2)-1)        #with respect to the new image
        new_centre_width= round(((new_width+1)/2)-1)          #with respect to the new image

        for i in range(h):
            for j in range(w):

                origin_y = I.shape[0]-1-i-original_centre_height
                origin_x = I.shape[1]-1-j-original_centre_width

                origin_xy = np.array([origin_x,origin_y,1])

                new_xy = np.dot(T,origin_xy)
                new_x = new_xy[0]
                new_y = new_xy[1]

                new_y = new_centre_height-new_y
                new_x = new_centre_width-new_x


                if 0<new_x < w and 0<new_y < h:
                    T_I[int(new_y), int(new_x)]  = I[i,j]



    else:

        T_I = np.zeros(I.shape,dtype='u1')


        for i in range(h):
            for j in range(w):
                origin_x = j
                origin_y = i
                origin_xy = np.array([origin_x,origin_y,1])

                new_xy = np.dot(T,origin_xy)
                new_x = new_xy[0]
                new_y = new_xy[1]

                if 0<new_x < w and 0<new_y < h:
                    T_I[int(new_y), int(new_x)]  = I[i,j]

    return T_I


def get_bilinear_intensity(img, y, x):

    # extract corner coordinates
    y_top = int(y)  # top in image representation is low in y value
    x_left = int(x)
    y_bottom = min(y_top+1, img.shape[0]-1)
    x_right = min(x_left+1, img.shape[1]-1)

    # extract fractional component
    y_frac = y - y_top
    x_frac = x - x_left

    # pixel values at corner
    intensity_top_left = img[y_top, x_left]
    intensity_top_right = img[y_top, x_right]
    intensity_bottom_left = img[y_bottom, x_left]
    intensity_bottom_right = img[y_bottom, x_right]

    # bilinear interpolation of intensity values based on fractional components
    intensity_interpolated_top = intensity_top_left * (1-x_frac) \
        + intensity_top_right * x_frac
    intensity_interpolated_bottom = intensity_bottom_left * (1-x_frac) \
        + intensity_bottom_right * x_frac
    intensity_interpolated = intensity_interpolated_top * (1-y_frac) \
        + intensity_interpolated_bottom * y_frac

    return intensity_interpolated


# driver function
if __name__=="__main__":

    name1 = input("Enter first input image name: ")

    name2 = input("Enter second input image name: ")

    # reading the image
    img1 = cv2.imread(name1, 1)

    # displaying the image
    cv2.imshow('image 1', img1)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image 1', click_event)

    # reading the image
    img2 = cv2.imread(name2, 1)

    # displaying the image
    cv2.imshow('image 2', img2)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image 2', click_event2)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    rows,cols = img2.shape[:2]

    h,w = img2.shape[:2]

    # Affine Transformation
    A = [[coord1[0][0],coord1[0][1],1,0,0,0],[0,0,0,coord1[0][0],coord1[0][1],1],[coord1[1][0],coord1[1][1],1,0,0,0],[0,0,0,coord1[1][0],coord1[1][1],1],[coord1[2][0],coord1[2][1],1,0,0,0],[0,0,0,coord1[2][0],coord1[2][1],1]]

    B = [coord2[0][0],coord2[0][1],coord2[1][0],coord2[1][1],coord2[2][0],coord2[2][1]]

    X = np.dot(np.linalg.inv(A),B)

    dst = cv2.warpAffine(img1, np.reshape(X,(2,3)), (w, w), flags=cv2.INTER_LINEAR)




    cv2.imshow("Input Transformed Image", img2)

    cv2.imshow("Original Image Transformed using matrix from given points", dst)

    cv2.imwrite("output.png", dst)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
