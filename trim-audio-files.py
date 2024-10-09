import os
import math
from pydub import AudioSegment

def split_audio_into_segments(input_folder, output_folder, segment_length_ms=10000):
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Duyệt qua tất cả các file trong thư mục input
    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):  # Thêm các định dạng audio khác nếu cần
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]

            # Đọc file audio
            audio = AudioSegment.from_file(input_path)

            # Chuyển đổi sang tần số 16000Hz
            audio = audio.set_frame_rate(16000)

            # Tính số lượng đoạn 10s
            num_segments = math.ceil(len(audio) / segment_length_ms)

            for i in range(num_segments):
                # Cắt đoạn 10s
                start = i * segment_length_ms
                end = start + segment_length_ms
                segment = audio[start:end]

                # Tạo tên file cho đoạn
                output_filename = f"{base_name}_segment_{i+1:03d}.wav"
                output_path = os.path.join(output_folder, output_filename)

                # Xuất file đã cắt và chuyển đổi sang định dạng WAV
                segment.export(output_path, format="wav", parameters=["-ac", "1"])  # Mono channel

                print(f"Processed: {filename} -> {output_filename}")

if __name__ == "__main__":
    input_folder = "input_audio"  # Thay đổi thành đường dẫn thư mục chứa file audio
    output_folder = "wav_folder"  # Thay đổi thành đường dẫn thư mục bạn muốn lưu file đã cắt
    
    split_audio_into_segments(input_folder, output_folder)
    print("All files processed.")