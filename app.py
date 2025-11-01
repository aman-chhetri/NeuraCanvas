import os
import streamlit as st

import torch
import numpy as np
import cv2
from PIL import Image

from main import neural_style_transfer

@st.cache_data
def prepare_imgs(content_im, style_im, RGB=False):
    """ Return scaled RGB images as numpy array of type np.uint8 """    
    # check sizes in order to avoid huge computation times:
    h,w,c = content_im.shape
    ratio = 1.
    if h > 512:
        ratio = 512./h
    if (w > 512) and (w>h):
        ratio = 512./w
    content_im = cv2.resize(content_im, dsize=None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)        
    # reshape style_im to match the content_im shape    
    # (method followed in Gatys et al. paper):
    style_im = cv2.resize(style_im, content_im.shape[1::-1], cv2.INTER_CUBIC)
    
    # pass from BGR (OpenCV) to RGB:
    if not RGB:
        content_im = cv2.cvtColor(content_im, cv2.COLOR_BGR2RGB)
        style_im   = cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB)
    
    return content_im, style_im

    
def print_info_NST():
    """ Print basic information about Neural Style Transfer within the app.
    """    
    st.markdown("""

                :blue[**Neural Style Transfer**] (*NST*) is a technique which combines two images (content image for the object of the image and the style image from which only the style is extracted) into a third target image.

                *Basically, in Neural Style Transfer we have two images- style and content. We need to copy the style from the style image and apply it to the content image. By, style I basically mean, the patterns, the brushstrokes, etc.*""")
    
    st.write("---")

    st.markdown("Let's have a look to an example: (The left-most image is the content, central image is the style and the right-most image is the output.)")
    

    st.image('assets/cover_img.jpeg', width='stretch')

    st.markdown("""
                Neural Style Transfer (NST) integrates several crucial components within its architecture to achieve the blending of artistic styles:

                1. :green[**Content Image**]: This is the image with the main subject or scene you want to change. It holds the basic structure and layout you want to keep.
                2. :green[**Style Image**]: The image with the artistic feel, like colors, textures and patterns. You want to use this style to change how your content image looks.
                3. :green[**Generated Image**]: The final output image that combines the content of the first image with the style of the second.
                ---
                """)
    
    # st.write("---") 
    
    st.write("#### :red[NST Architecture]")
    st.image('assets/nst_architecture.jpeg', width='stretch')

    st.markdown(""" 
                The Basic concept is simple: 

                * Take two losses and minimize them as much as possible. 
                * The first loss describes the distance in content between the content image and the target image.
                * While the second loss describes the distance in style between the style image and the style of the input image. 
                ---
                """)
    

    st.write("#### How to tune the parameters? (for better result)")

    # ---- Dropdown for Weights ----
    with st.expander("Weights"):
        st.markdown("##### Weights of the Loss Function:")
        st.latex(r"""
            \mathcal{L}(\lambda_{\text{content}}, 
            \lambda_{\text{style}}, \lambda_{\text{variation}}) =
            \lambda_{\text{content}}\mathcal{L}_{\text{content}} +
            \lambda_{\text{style}}\mathcal{L}_{\text{style}} +
            \lambda_{\text{variation}}\mathcal{L}_{\text{variation}}
        """)
        st.markdown("""
        - :green[**Content**]: Higher value increases the influence of the *Content* image  
        - :green[**Style**]: Higher value increases the influence of the *Style* image  
        - :green[**Variation**]: Higher value makes the resulting image look smoother
        """)

    # ---- Dropdown for Number of iterations ----
    with st.expander("Number of Iterations"):
        st.markdown("""
        ~ Its value defines the duration of the optimization process, A higher number will make the optimization process longer.  
        ~ If the image looks unoptimized, try increasing this number (or tune the weights of the loss function).
        """)

    # ---- Dropdown for Results ----
    with st.expander("Save Image"):
        st.markdown("""
        If this option is checked, then once the optimization finishes, the image will be saved to your computer (in the same folder where the **app.py** file of this project is located).
        """)

if __name__ == "__main__":
    
    # app title and sidebar:
    st.title('NeuraCanvas 🎨')
    st.markdown("""
                > :orange[**An AI-powered art generation app that transforms your photos by blending their content with unique artistic styles to create stunning, one-of-a-kind images.**]
                
                ---""")
    # st.header("This is a header with a divider", divider="gray")

    # Select what to do:
    # st.sidebar.title('Configuration')
    st.sidebar.title('Select the pages')
    options = ['Homepage', 'Click here to run app!']
    app_mode = st.sidebar.selectbox('Dropdown 👇',
                                    options
                                    )
    
    # Set parameters to tune at the sidebar:
    st.sidebar.title('Parameters')
    #Weights of the loss function
    st.sidebar.subheader('Weights')
    step=1e-1
    cweight = st.sidebar.number_input("Content", value=1e-3, step=step, format="%.5f")
    sweight = st.sidebar.number_input("Style", value=1e-1, step=step, format="%.5f")
    vweight = st.sidebar.number_input("Variation", value=0.0, step=step, format="%.5f")
    # number of iterations
    st.sidebar.subheader('Number of Iterations')
    niter = st.sidebar.number_input('Iterations', min_value=1, max_value=1000, value=20, step=1)
    # save or not the image:
    st.sidebar.subheader('Save the output image?')
    save_flag = st.sidebar.checkbox('Save result')
    
    # Show the page of the selected page:
    if app_mode == options[0]:
        print_info_NST()
        
    elif app_mode == options[1]:        
        st.markdown("#### Upload the content and style images!")        
        col1, col2 = st.columns(2)
        im_types = ["png", "jpg", "jpeg"]
        
        # Create file uploaders in a two column layout, as well as
        # placeholder to later show the images uploaded:
        with col1:
            file_c = st.file_uploader("Choose CONTENT Image", type=im_types)
            imc_ph = st.empty()            
        with col2: 
            file_s = st.file_uploader("Choose STYLE Image", type=im_types)
            ims_ph = st.empty()
        
        # if both images have been uploaded then preprocess and show them:
        if all([file_s, file_c]):
            # preprocess:
            im_c = np.array(Image.open(file_c).convert('RGB'))
            im_s = np.array(Image.open(file_s).convert('RGB'))
            im_c, im_s = prepare_imgs(im_c, im_s, RGB=True)
            
            # Show images:
            imc_ph.image(im_c, width='stretch')
            ims_ph.image(im_s, width='stretch') 
        
        st.markdown("""
                    ##### Once your images are ready, Click START to generate image!
                    """)
        
        # button for starting the stylized image:
        start_flag = st.button("START", help="Start the optimization process")
        bt_ph = st.empty() # Possible message above the button
    
        if start_flag:
            if not all([file_s, file_c]):
                bt_ph.markdown("You need to **upload the images** first! :)")
            elif start_flag:
                bt_ph.markdown("Optimizing the model ⌛.....Take a rest, have some water 💧 or coffee ☕")
                
        if start_flag and all([file_s, file_c]):
            # Create progress bar:
            progress = st.progress(0.)
            # Create place-holder for the stylized image:
            res_im_ph = st.empty()
            # config the NST function:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # parent directory of this file:
            parent_dir = os.path.dirname(__file__)
            out_img_path = os.path.join(parent_dir, "app_stylized_image.jpg")
            cfg = {
                'output_img_path' : out_img_path,
                'style_img' : im_s,
                'content_img' : im_c,
                'content_weight' : cweight,
                'style_weight' : sweight,
                'tv_weight' : vweight,
                'optimizer' : 'lbfgs',
                'model' : 'vgg19',
                'init_metod' : 'random',
                'running_app' : True,
                'res_im_ph' : res_im_ph,
                'save_flag' : save_flag,
                'st_bar' : progress,
                'niter' : niter
                }
            
            result_im = neural_style_transfer(cfg, device)
            # res_im_ph.image(result_im, channels="BGR")
            bt_ph.markdown("This is the resulting **stylized image**!")