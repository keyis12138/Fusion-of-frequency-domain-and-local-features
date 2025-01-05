import PIL.Image
import cv2
import numpy as np
import random
from PIL import Image
from torchvision.transforms.functional import crop
import torchvision.transforms as transforms
import concurrent.futures


def img_to_patches(img) -> tuple:
    """
    Returns 32x32 patches of a resized 256x256 images,
    it returns 64 patches on grayscale and 64*3 patches
    on the RGB color scale
    --------------------------------------------------------
    ## parameters:
    - input_path: Accepts input path of the image
    """
    img = transforms.ToPILImage()(img)
    patch_size = 32
    grayscale_imgs = []
    imgs = []
    # channels,height, width  = img.shape
    height, width = img.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (i, j, patch_size, patch_size)
            img_color = np.asarray(crop(img, *box))
            img_color = img_color.astype(np.uint8)
            # img_color =crop(img,*box).numpy()
            # grayscale_image = img_color[2]*0.299+img_color[1]*0.587+img_color[0]*0.114
            grayscale_image = img_color[:, :, 0] * 0.299 + img_color[:, :, 1] * 0.587 + img_color[:, :, 2] * 0.114
            grayscale_image = grayscale_image.astype(np.uint8)
            grayscale_imgs.append(grayscale_image)
            imgs.append(img_color)
    return grayscale_imgs,imgs



def get_l1(v,x,y):
    l1 = v[:,1:y]
    # 1 to m, 1 to m-1   
    return np.sum(np.abs(v[:,0:y-1]-l1))

def get_l2(v,x,y):
    l2 = v[1:x]
    # 1 to m-1, 1 to m
    return np.sum(np.abs(v[0:x-1]-l2))

def get_l3l4(v,x,y):
    l3 = v[1:x,1:y]
    l4 = v[0:x-1,1:y]
    # 1 to m-1, 1 to m-1
    return np.sum(np.abs(v[0:x-1,0:x-1]-l3))+np.sum(np.abs(v[1:x,0:y-1]-l4))

def get_pixel_var_degree_for_patch(patch:np.array)->int:
    """
    gives pixel variation for a given patch
    ---------------------------------------
    ## parameters:
    - patch: accepts a numpy array format of the patch of an image
    """
    x,y = patch.shape
    l1 = get_l1(patch,x,y)
    l2 = get_l2(patch,x,y)
    l3l4 = get_l3l4(patch,x,y)
    return  l1+l2+l3l4


def extract_rich_and_poor_textures(variance_values:list, patches:list):
    """
    returns a list of rich texture and poor texture patches respectively
    --------------------------------------------------------------------
    ## parameters:
    - variance_values: list of values that are pixel variances of each patch
    - color_patches: coloured patches of the target image
    """
    threshold = np.mean(variance_values)
    rich_texture_patches = []
    poor_texture_patches = []
    for i,j in enumerate(variance_values):
        if j >= threshold:
            rich_texture_patches.append(patches[i])
        else:
            poor_texture_patches.append(patches[i])
    
    return rich_texture_patches, poor_texture_patches



def get_complete_image(patches:list, coloured=True):
    """
    Develops complete 265x256 image from rich and poor texture patches
    ------------------------------------------------------------------
    ## parameters:
    - patches: Takes a list of rich or poor texture patches
    """
    random.shuffle(patches)
    p_len = len(patches)
    while len(patches)<64:
        patches.append(patches[random.randint(0, p_len-1)])
    if(coloured==True):
        grid = np.asarray(patches).reshape((8,8,32,32,3))
    else:
        grid = np.asarray(patches).reshape((8,8,32,32))


    # joins columns to only leave rows
    rows = [np.concatenate(grid[i,:], axis=1) for i in range(8)]

    # joins the rows to create the final image
    img = np.concatenate(rows,axis=0)
    return img


def smash_n_reconstruct(input, coloured=True):
    """
    Performs the SmashnReconstruct part of preprocesing
    reference: [link](https://arxiv.org/abs/2311.12397)

    return rich_texture,poor_texture
    
    ----------------------------------------------------
    ## parameters:
    - input_path: Accepts input path of the image
    """
    gray_scale_patches, color_patches = img_to_patches(input)
    pixel_var_degree = []
    for patch in gray_scale_patches:
        pixel_var_degree.append(get_pixel_var_degree_for_patch(patch))
    
    # r_patch = list of rich texture patches, p_patch = list of poor texture patches
    if(coloured):
        r_patch,p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree,patches=color_patches)
    else:
        r_patch,p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree,patches=gray_scale_patches)
    rich_texture = get_complete_image(r_patch, coloured)
    poor_texture = get_complete_image(p_patch, coloured)
    # poor_texture = None
    rich_texture = transforms.ToTensor()(rich_texture)
    return rich_texture, poor_texture

