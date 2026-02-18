"""
    -audio
    -load
    -normalize
    -segment 
    -sauvegarder
"""

import logging 
import librosa 
import soundfile
import tqdm 
import os 


OUTPUT_DIR="./results"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)
#logging.basicConfig(filename="",level=logging.INFO)
logger.info('Started')
logger.info("Loadin audio file")
array,sr=librosa.load('./data_test/tyla_segment.wav')
duration=librosa.get_duration(y=array,sr=sr)
logger.debug("Sampling rate: %d duration: %d",sr,duration)

logger.info('Normalizing Audio')
norm_arr=librosa.util.normalize(array)
norm_duration=librosa.get_duration(y=norm_arr,sr=sr)
logger.debug('Noralized Audio duration: %d',norm_duration)

segments=librosa.util.frame(array,frame_length=3,hop_length=2)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR,exist_ok=True)

for i,segment in tqdm.tqdm(enumerate(segments)):
    logger.info("Saving %d segment ...",i)
    soundfile.write(f"{OUTPUT_DIR}/{i}_seg.wav",segment,samplerate=sr)

