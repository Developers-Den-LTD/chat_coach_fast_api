from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

from pydantic import BaseModel
import torch


app = FastAPI()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


tokenizer = AutoTokenizer.from_pretrained("Devden/Generalized-Persona",  use_auth_token=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)
model = AutoModelForCausalLM.from_pretrained("Devden/Generalized-Persona",  use_auth_token=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)
    


class Item(BaseModel):
    input: str



@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.post("/get_response")
def get_response(item : Item): 
    new_user_input_ids = tokenizer.encode(item.input + tokenizer.eos_token, return_tensors='pt')   
    chat_history_ids = model.generate(
        new_user_input_ids,
        max_new_tokens=1024,
        no_repeat_ngram_size=3,
        num_beams=3,
        top_k=3,
        top_p=0.75,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response



    
