import os
import sys
from subprocess import Popen, PIPE
import torch 
import whisper
import winsound
import torchaudio
from tortoise import api
from tortoise.utils.audio import load_audio, load_voice, load_voices
import json
from TTS.api import TTS
import nltk
import numpy as np
import scipy.io.wavfile as wavfile

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

from llama_cpp import Llama

MODEL_DIR = "C:\\users\\Angelo\\language-models"
MODEL_NAME = "TheBloke_Samantha-7B-GGML/Samantha-7B.ggmlv3.q4_1.bin"
#"pygmalion-13b-superhot-8k.ggmlv3.q4_K_S.bin"
#"pygmalion-13b-ggml-q4_1.bin" 
#"TheBloke_Samantha-7B-GGML/Samantha-7B.ggmlv3.q4_1.bin"

MY_NAME = "You"
ASSISTANT_NAME = "Samantha"

whisper_model = None
tts = None
llm = None
voice_samples = None
conditioning_latents = None 

#CHAT_PROMPT_CONTEXT = chara['char_name']+chara['char_persona']+chara['example_dialogue']+chara['description']+chara['personality']+chara['scenario']
CHAT_PROMPT_CONTEXT = 'This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information. It will respond in one short sentence. It will not reply with more than 20 words. Long responses are bad. No greeting before a response.'
#context = ("{ctx} {user}: Hello, {assistant}. {assistant}: What can I do for you today?".format(ctx=CHAT_PROMPT_CONTEXT, user=MY_NAME, assistant=ASSISTANT_NAME))

MAX_CTX = 2048

def load_models(TRANSCRIBE, LLM_INFERENCE, TTS_INFERENCE):
    global whisper_model, tts, llm
    if TTS_INFERENCE:
        #load_fast_tortoise()
        #load_TTS()
        preload_models(text_use_small=True,coarse_use_small=True, fine_use_small=True) #bark
    if TRANSCRIBE:
        whisper_model = whisper.load_model("tiny.en", device="cpu")

    if LLM_INFERENCE:
        llm = Llama(model_path=(MODEL_DIR+"\\"+ MODEL_NAME), n_threads=6, n_gpu_layers=15, verbose=False, n_ctx=MAX_CTX, low_vram=True)
    
def load_fast_tortoise():
    global tts, voice_samples, conditioning_latents
    tts = api.TextToSpeech(device='cuda', high_vram=False, kv_cache=True)
    voice = 'train_dotrice' #@param {type:"string"}
    voice_samples, conditioning_latents = load_voice(voice)

def load_TTS():
    global tts,modelno
    
    #model_name = TTS.list_models()[25]
    model_name = TTS.list_models()[modelno]
    
    tts = TTS(model_name)

def process_input_speech(speech_path : str) -> str :
    result = whisper_model.transcribe(speech_path)
    return result['text']

def generate_LLM_response(spoken_text : str) -> str :
    global context

    output = llm(prompt=context, repeat_penalty=1.3)

    response = output['choices'][0]['text']
    return response

def speech_synthesis_fast_tortoise(assistant_dialogue : str):
    global voice_samples, conditioning_latents, tts
    
    preset = 'single_sample'
    gen = tts.tts_with_preset(assistant_dialogue, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                            preset=preset,half=True)
    torchaudio.save('out.wav', gen.squeeze(0).cpu(), 24000)
    winsound.PlaySound('out.wav', winsound.SND_FILENAME)

modelno = 0
def speech_synthesis_TTS(assistant_dialogue : str):
    global tts, model_name

    #multi speaker
    if hasattr(tts,'speaker'):
        print("Multi")
        for x in tts.speakers:
            print(x)
        speaker = 'male-en-2'
        speaker = tts.speakers[0]
        tts.tts_to_file(text=assistant_dialogue, speaker=speaker, language=tts.languages[0], file_path="out.wav")
    #single speaker
    else:
        print("Single")
        #tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)
        tts.tts_to_file(text=assistant_dialogue, file_path="out.wav")

    winsound.PlaySound('out.wav', winsound.SND_FILENAME)

def bark_gen(speech:str):

    # script = """
    # Hey, have you heard about this new text-to-audio model called "Bark"? 
    # Apparently, it's the most realistic and natural-sounding text-to-audio model 
    # out there right now. People are saying it sounds just like a real person speaking. 
    # I think it uses advanced machine learning algorithms to analyze and understand the 
    # nuances of human speech, and then replicates those nuances in its own speech output. 
    # It's pretty impressive, and I bet it could be used for things like audiobooks or podcasts. 
    # In fact, I heard that some publishers are already starting to use Bark to create audiobooks. 
    # It would be like having your own personal voiceover artist. I really think Bark is going to 
    # be a game-changer in the world of text-to-audio technology.
    # """.replace("\n", " ").strip()
    script = speech.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(script)

    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_9"
#"v2/en_speaker_6"
    silence = np.zeros(int(0.1 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in sentences:
        sentence =  "[WOMAN] "+sentence
        words_ct = len(sentence.split(" "))
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.03,  # this controls how likely the generation is to end]
            #max_gen_duration_s=words_ct*0.5            
        )

        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    print(pieces)
    arr = np.concatenate(pieces)
    # Normalize the audio array to the range [-1, 1]
    arr /= np.max(np.abs(arr))

    # Convert the audio array to 16-bit signed integer
    arr = (arr * 32767).astype(np.int16)

    # Set the sample rate (e.g., 44100 for CD-quality audio)

    # Save the audio array as a WAV file
    wavfile.write('out.wav', SAMPLE_RATE, arr)
    #torchaudio.save('out.wav',arr, 22000 , format='wav')
    winsound.PlaySound('out.wav', winsound.SND_FILENAME)

def main():
    global context

    TRANSCRIBE = False
    LLM_INFERENCE = True
    TTS_INFERENCE = True

    # with open("../characters/character.txt", 'r') as f:
    #     CHAT_PROMPT_CONTEXT = f.read().rstrip() 

    context = CHAT_PROMPT_CONTEXT

    print("Cuda "+"available." if torch.cuda.is_available() else "not available.")

    #for using python TTS lib
    global modelno 
    modelno = 0
    if len(sys.argv) >= 2:
        modelno = int(sys.argv[1])

    load_models(TRANSCRIBE, LLM_INFERENCE, TTS_INFERENCE)
    
    print("Models loaded.")

    print(context)

    with open("last-conversation-log.txt",'w') as log:
        log.write(context)
        while(True):
            if TRANSCRIBE:
                spoken = process_input_speech("speech.wav")
            else:
                spoken = input(">")

            context+="\n"+MY_NAME+":"+spoken+"\n"+ASSISTANT_NAME+": "
            log.write("\n"+MY_NAME+":"+spoken+"\n"+ASSISTANT_NAME+": ")

            print("ctx:",len(context))

            if LLM_INFERENCE:
                response = generate_LLM_response(spoken)
                print("RESPONSE:",response)
            else:
                response = "no response"

            context+=response
            log.write(response)

            if TTS_INFERENCE:
                bark_gen(response)
                #speech_synthesis_fast_tortoise(response)
                #speech_synthesis_TTS(response)

main()