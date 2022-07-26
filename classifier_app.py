from cv2 import circle
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image
import pickle


mp_drawing=mp.solutions.drawing_utils
map_face_mesh=mp.solutions.face_mesh

DEMO_IMG='demo.jpg'

st.title('Expression classifier using Mediapipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
       width:350px
   }
    [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
    width:350px
    margin-left:-350px
   }
   </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Classifier Sidebar')
st.sidebar.subheader('Select Parameters')

@st.cache()
def image_resize(image,width=400,height=400,inter=cv.INTER_AREA):
    dim=None
    (h,w)=image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r=width/float(w)
        dim=(int(w*r),height)

    else:
        r=width/float(w)
        dim=(width,int(h*r))

    #resize the image
    resized=cv.resize(image,dim,interpolation=inter)
    return resized

app_mode=st.sidebar.selectbox('Choose the App mode',['Run on video','Run on image','About App'])


if app_mode=='About App':
    st.markdown('Facial Expression classification using **Mediapipe** and **SVM**')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
       width:350px
   }
    [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
    width:350px
    margin-left:-350px
   }
   </style>
    """,
    unsafe_allow_html=True,
)

    st.markdown(
            '''
            **About Me**\n
            Hey this is ** Suhas RK ** from ** KritiKal Solutions **.\n
            If you are interested in building more Computer Vision apps like this one then visit the ** My Github page ** at
            https://github.com/SuhasRK\n
            Also check me out on Social Media
                [YouTube](https://www.youtube.com/c/HackToLearn)
                [LinkedIn](https://www.linkedin.com/in/suhasrk233/)
            
            
            
            '''
    )


elif app_mode=='Run on image':


    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
       width:350px
   }
    [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
    width:350px
    margin-left:-350px
   }
   </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown('**Detected Faces**')
    det_face_text=st.markdown('0')

    # to change the number of face detected 
    max_faces=1

    # max_faces=st.sidebar.number_input("Maximum number of faces",value=1,min_value=1,max_value=max_face_detect)

    # st.sidebar.markdown('---')

    detection_confidence=st.sidebar.slider("Detection Confidence",value=0.5,min_value=0.0,max_value=1.0)

    img_file_buffer=st.sidebar.file_uploader("Upload the image",type=["jpg","jpeg","png"])

    if img_file_buffer is not None:
        image=np.array(Image.open(img_file_buffer))

    else:
        demo_image=DEMO_IMG
        image=np.array(Image.open(demo_image))

    image=image_resize(image)
    st.sidebar.text('Original Image')
    st.sidebar.image(image)


    # Landmark Detection 
    mesh_coord=[]

    def landmarksDetection(img,results):
        height,width=img.shape[:2]
        # list[(x1,y1),(x2,y2)]
        mesh_coord=[[int(point.x * width),int(point.y * height)] for point in results.multi_face_landmarks[0].landmark]
        # mesh_coord=[[point.x,point.y,point.z] for point in results.multi_face_landmarks[0].landmark]



        return mesh_coord

    with map_face_mesh.FaceMesh(max_num_faces=max_faces,min_detection_confidence=detection_confidence,min_tracking_confidence=0.5) as face_mesh:
    
        image = image
        #   image=cv.resize(image,(400,400))

        bgr_image=cv.cvtColor(image,cv.COLOR_RGB2BGR)
        results=face_mesh.process(bgr_image)
            # print(type(results))
        face_count=1
        out_image=image.copy()
        circleDrawingSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
        lineDrawingSpec = mp_drawing.DrawingSpec(thickness=1, color=(0,255,0))
        if results.multi_face_landmarks:
            mp_drawing.draw_landmarks(out_image,results.multi_face_landmarks[0],map_face_mesh.FACE_CONNECTIONS,circleDrawingSpec, lineDrawingSpec)
            mesh_cords=landmarksDetection(image,results)
            mesh_cords=np.array(mesh_cords)
            mesh_cords=mesh_cords.flatten()
        st.markdown('**Detected Landmarks**')
        st.markdown(len(mesh_cords)//2)



        det_face_text.write(f"<h1 style='text-align:center;color:red'>{face_count}</h1>",unsafe_allow_html=True)
        st.subheader("Output Image")
        st.image(out_image)

        # Model Loading 
        filename='SVM_optimised_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))


        predict=loaded_model.predict([mesh_cords.tolist()])

        # print(predict)

        st.markdown('---')
        st.markdown('**Prediction**')
        pred_class=st.markdown('')
        status=("Attentive" if predict[0]==1 else "Bored")

        
        pred_class.write(f"<h1 style='text-align:center;color:red'>{status}</h1>",unsafe_allow_html=True)

elif app_mode=='Run on video':

    DEMO_VIDEO='demo_video.mp4'

    st.set_option('deprecation.showfileUploaderEncoding',False)

    use_webcam=st.sidebar.checkbox('Use Webcam')
    record=st.sidebar.button('Record Video')

    if record:
        st.checkbox("Recording",value=True)

    
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
       width:350px
   }
    [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
    width:350px
    margin-left:-350px
   }
   </style>
    """,
    unsafe_allow_html=True,
    )


    st.markdown('**Detected Faces**')
    det_face_text=st.markdown('0')

    # to change the number of face detected 
    max_faces=1

    detection_confidence=st.sidebar.slider("Detection Confidence",value=0.5,min_value=0.0,max_value=1.0)
    st.sidebar.markdown('---')

    tracking_confidence=st.sidebar.slider("Tracking Confidence",value=0.5,min_value=0.0,max_value=1.0)
    st.sidebar.markdown('---')

    st.markdown('## Output')

    stframe=st.empty
    video_file_buffer=st.sidebar.file_uploader("Upload Video",type=['mp4','mov','avi','asf','m4v'])
    tfile=tempfile.NamedTemporaryFile(delete=False)

    # getting input video 
    if not video_file_buffer:
        if use_webcam:
            vid=cv.VideoCapture(0)
        else:
            vid=cv.VideoCapture(DEMO_VIDEO)
            tfile.name=DEMO_VIDEO
    else:   
        tfile.write(video_file_buffer.read())
        vid=cv.VideoCapture(tfile.name)

    width=int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height=int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps_input = int(vid.get(cv.CAP_PROP_FPS))


    # Recording part 
    codec = cv.VideoWriter_fourcc('M','J','P','G')
    out = cv.VideoWriter('output.mp4',codec,fps_input,(width,height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfile.name)




    
