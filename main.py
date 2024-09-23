from typing import Optional

from fastapi import FastAPI, UploadFile

from sam_utils import segment_image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    with open("/test_images/0279.jpg", "rb") as file:
        image_bytes = file.read()
        s = segment_image(image_bytes)
        print('asd')
    return {"item_id": item_id, "q": q}


@app.post("/predict")
def predict(file: UploadFile):
    img_bytes = file.file.read()
    s = segment_image(img_bytes)
    print('asd')
    # class_id, class_name = get_prediction(image_bytes=img_bytes)
    return {"class_id": 1, "class_name": 1}