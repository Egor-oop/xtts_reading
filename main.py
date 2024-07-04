import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# /Users/egorgulido/Library/"Application Support"/tts/tts_models--multilingual--multi-dataset--xtts_v2

xtts_path = '/Users/egorgulido/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2'
print('Loading model...')
config = XttsConfig()
config.load_json(xtts_path + '/config.json')
model = Xtts.init_from_config(config)
model.load_checkpoint(config,
                      checkpoint_dir=xtts_path,
                      use_deepspeed=True if torch.cuda.is_available() else False)
if torch.cuda.is_available():
    model.cuda()

print('Computing speaker latents...')
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=['audio/yegor_en.wav'])

print('Inference...')
print('First')
out = model.inference(
    'It took me quite a long time to develop a voice and now that I have it I am not going to be silent.',
    'en',
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7
)
torchaudio.save('output.wav', torch.tensor(out['wav']).unsqueeze(0), 24000)

print('Second')
out = model.inference(
    'It doesn\'t bother me! But what\'s your name?',
    'en',
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7
)
torchaudio.save('output2.wav', torch.tensor(out['wav']).unsqueeze(0), 24000)
