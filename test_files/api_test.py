from fastapi import FastAPI, UploadFile, BackgroundTasks
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

from plus import process_path


app = FastAPI(timeout=1000)

tokenizer = AutoTokenizer.from_pretrained("bigbird-pegasus-large-pubmed-tokenizer")
model = BigBirdPegasusForConditionalGeneration.from_pretrained("bigbird-pegasus-large-pubmed-model")

# Define a root `/` endpoint
@app.post('/')
async def extract(file: UploadFile, background_tasks: BackgroundTasks):
    with open('local.pdf', 'wb') as f:
        f.write(file.file.read())
    background_tasks.add_task(process_path, 'local.pdf', model, tokenizer, api=True)
    # summary = process_path('local.pdf', model, tokenizer, api=True)
    return {'summary': "started processing..."}

@app.get('/retrieve')
def retrieve():
    with open('all_summaries.txt', 'r') as f:
        summary = f.read()
    if summary:
        open('all_summaries.txt', 'w').close()
        return {'summary': summary}
    else:
        return {'summary': "No summary available yet."}

# from fastapi import FastAPI, UploadFile
# from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
# from main import process_path

# app = FastAPI()

# # Initialize tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bigbird-pegasus-large-pubmed")
# model = BigBirdPegasusForConditionalGeneration.from_pretrained("bigbird-pegasus-large-pubmed")

# @app.post('/')
# async def extract(file: UploadFile):
#     # Save uploaded file locally
#     with open('local.pdf', 'wb') as f:
#         contents = await file.read()
#         f.write(contents)

#     # Process the saved file and generate a summary
#     summary = process_path('local.pdf', model, tokenizer, api=True)

#     # Clean up: Close file and delete if necessary
#     await file.close()

#     return {'summary': summary}
