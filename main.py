# from openai import OpenAI
# from dotenv import load_dotenv
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset

os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
    chunk_length_s=10,
    stride_length_s=2,
)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]
sample = "./audio/lsy_audio_2023_58s.mp3"

result = pipe(sample)
# print(result["text"])

start_end_text = []

for chunk in result["chunks"]:
    start = chunk["timestamp"][0]
    end = chunk["timestamp"][1]
    text = chunk["text"]
    start_end_text.append([start, end, text])

import pandas as pd
df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
df.to_csv("lsy_audio_2023_58.csv", index=False, sep="|")
display(df)

print(result)
