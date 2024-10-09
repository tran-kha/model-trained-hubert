import os
import librosa
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_audio_duration(file_path):
    try:
        duration = librosa.get_duration(filename=file_path)
        return duration
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    wav_folder = "wav_folder"
    
    if not os.path.exists(wav_folder):
        logging.error(f"The directory {wav_folder} does not exist.")
        return

    audio_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
    
    if not audio_files:
        logging.info(f"No WAV files found in {wav_folder}")
        return

    logging.info(f"Found {len(audio_files)} WAV files in {wav_folder}")

    for audio_file in audio_files:
        file_path = os.path.join(wav_folder, audio_file)
        duration = get_audio_duration(file_path)
        if duration is not None:
            logging.info(f"{audio_file}: {duration:.2f} seconds")

if __name__ == "__main__":
    main()