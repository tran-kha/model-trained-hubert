import os
import wave
import contextlib

def check_wav_files(folder_path):
    print(f"Kiểm tra các file trong thư mục: {folder_path}")
    print("-" * 50)

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                with contextlib.closing(wave.open(file_path, 'r')) as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    
                    print(f"File: {filename}")
                    print(f"  Số kênh: {channels}")
                    print(f"  Tần số lấy mẫu: {frame_rate}Hz")
                    print(f"  Độ sâu bit: {sample_width * 8}-bit")
                    
                    if channels == 1 and frame_rate == 16000 and sample_width == 2:
                        print("  Kết quả: Đạt tiêu chuẩn ✅")
                    else:
                        print("  Kết quả: Không đạt tiêu chuẩn ❌")
                        if channels != 1:
                            print("    - Cần 1 kênh (mono)")
                        if frame_rate != 16000:
                            print("    - Cần tần số 16000Hz")
                        if sample_width != 2:
                            print("    - Cần độ sâu 16-bit")
            except Exception as e:
                print(f"  Lỗi khi đọc file: {str(e)}")
            
            print("-" * 50)

if __name__ == "__main__":
    wav_folder = "wav_folder"  # Đường dẫn đến thư mục chứa các file WAV
    check_wav_files(wav_folder)

    import os
import librosa
import numpy as np

def check_audio_files(folder_path):
    print("Checking audio files in:", folder_path)
    print("-" * 50)

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Load the audio file
                y, sr = librosa.load(file_path, sr=None)
                
                # Check if the audio is completely silent (all zeros)
                if np.allclose(y, 0):
                    print(f"WARNING: {filename} is completely silent (all zeros)")
                else:
                    # Check for partially zeroed sections
                    zero_sections = np.where(np.abs(y) < 1e-6)[0]
                    if len(zero_sections) > 0:
                        longest_zero_section = np.max(np.diff(np.where(np.abs(np.diff(zero_sections)) > 1)[0]))
                        if longest_zero_section > sr * 0.1:  # If there's a silent section longer than 0.1 seconds
                            print(f"WARNING: {filename} has a silent section of {longest_zero_section/sr:.2f} seconds")
                        else:
                            print(f"OK: {filename}")
                    else:
                        print(f"OK: {filename}")
                
                # Print some statistics
                print(f"  Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
                print(f"  Mean: {np.mean(y):.6f}")
                print(f"  Std: {np.std(y):.6f}")
                print(f"  Min: {np.min(y):.6f}")
                print(f"  Max: {np.max(y):.6f}")
                
            except Exception as e:
                print(f"ERROR: Could not process {filename}: {str(e)}")
            
            print("-" * 50)

# Use the function
check_audio_files("wav_folder")