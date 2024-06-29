from TTS.api import TTS

# Model is located here on mac
# /Users/egorgulido/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2
print('Started loading model')
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

print('Model has been just installed\nStarting generating response')

tts.tts_to_file(text="Trump received a Bachelor of Science in economics from the University of Pennsylvania in 1968. His father named him president of his real estate business in 1971. Trump renamed it the Trump Organization and reoriented the company toward building and renovating skyscrapers, hotels, casinos, and golf courses. After a series of business failures in the late twentieth century, he launched successful side ventures, mostly licensing the Trump name. From 2004 to 2015, he co-produced and hosted the reality television series The Apprentice. He and his businesses have been plaintiffs or defendants in more than 4,000 legal actions, including six business bankruptcies.",
                file_path="output.wav",
                speaker_wav='audio/caseoh.wav',
                language="en")

print('DONE!')