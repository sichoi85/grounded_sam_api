from fastapi import FastAPI
#import yourclass

app = FastAPI()

# Load the ML model when the server starts
# model = yourclass.load()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get()
