from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from ocr_preproc import process_path
from inference import ner_infer, llm_extract
import tempfile, pathlib

app = FastAPI(title='LST-DocAI')

@app.post('/extract_info')
async def extract_info(file: UploadFile = File(...)):
    suffix = pathlib.Path(file.filename).suffix or '.txt'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    text = process_path(tmp_path)
    entities = ner_infer(text)
    llm_out = llm_extract(text)
    return JSONResponse({'entities': entities, 'llm': llm_out})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
