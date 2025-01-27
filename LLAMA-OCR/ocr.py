import time
import base64
from dotenv import load_dotenv
import io
from PIL import Image
from together import Together
import streamlit as st

load_dotenv()

uploaded_img = st.file_uploader("Upload Image",type=['jpg','jpeg','png'])
if uploaded_img is not None:
    img = Image.open(uploaded_img)

    st.sidebar.markdown("#### Preview image")
    st.sidebar.image(img)

    def encode_image(img):
        buffered = io.BytesIO()
        if img.mode in ("RGBA", "LA"):
            format = "PNG"  
        else:
            format = "JPEG"
        img.save(buffered, format=format)  
        
        byte_data = buffered.getvalue()
        
        return base64.b64encode(byte_data).decode('utf-8')
        
    try:
        base_image = encode_image(img)
        
        client = Together()

        prompt = "Extract the content from the image in markdown form, display a short description of only one line about the image also provide a heading for it as ### Description and extract only the content after the description exact as in the image and assign it heading ### Generated Content"

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role":"user",
                    "content":[
                        {"type":"text","text":prompt},
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":f"data:image/jpeg;base64,{base_image}",
                            },
                        }
                    ]
                }
            ],
        )
        def generate(string: str):
            for i in string:
                yield i + ""
                time.sleep(0.005)
        st.write_stream(generate(response.choices[0].message.content))
    except Exception as e:
        st.error(f"Error occured:{e}")
