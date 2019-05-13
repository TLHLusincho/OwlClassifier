from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
import uvicorn


from fastai.vision import *

app = Starlette()
path = Path(__file__).parent


model_path = Path('data')
learn = load_learner(model_path)


@app.route('/')
def index(request):
    html = path/'views'/'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': str(learn.predict(img)[0])})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8559)