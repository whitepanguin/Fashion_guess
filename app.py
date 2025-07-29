# Gradio frontend
import gradio as gr
import requests
import io

def classify_with_backend(image):
    url = "http://127.0.0.1:8000/classify"
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    response = requests.post(url, files={"file": ("image.png", image_bytes, "image/png")})
    if response.status_code == 200:
        return response.json().get("label", "Error")
    else:
        return "Error"

iface = gr.Interface(
    fn=classify_with_backend,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="패션 이미지 분류하기",
    description="패션 이미지를 넣어주세요 !!"
)

if __name__ == "__main__":
    iface.launch()