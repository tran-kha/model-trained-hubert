import os
import torch
import torchaudio
import numpy as np
from transformers import HubertConfig, HubertForCTC, Trainer, TrainingArguments, TrainerCallback
from datasets import IterableDataset
import argparse
import logging
import psutil
import traceback
import librosa
import gc
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.table import Table

# Tạo một console để in các thông báo đẹp hơn
console = Console()

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Định nghĩa các thư mục
INPUT_DIR = "wav_folder"
OUTPUT_DIR = "fine_tuned_model"

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_memory_usage():
    """In ra thông tin về việc sử dụng bộ nhớ."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    console.print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def load_and_preprocess_audio(file_path, target_sample_rate=16000, chunk_length=5, max_duration=60):
    """Đọc và tiền xử lý file âm thanh."""
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Chuyển đổi tốc độ lấy mẫu nếu cần
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    
    # Đảm bảo âm thanh là mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Chuẩn hóa waveform
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
    
    # Cắt hoặc pad để đạt độ dài tối đa
    max_length = max_duration * target_sample_rate
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    elif waveform.shape[1] < max_length:
        pad_length = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    # Chia thành các đoạn
    chunk_samples = chunk_length * target_sample_rate
    chunks = torch.split(waveform, chunk_samples, dim=1)
    
    return [chunk.squeeze().numpy() for chunk in chunks if chunk.numel() > 0]

def audio_generator(audio_files, chunk_length=5, max_duration=60):
    """Generator để tạo dữ liệu âm thanh."""
    for file_path in audio_files:
        try:
            audio_chunks = load_and_preprocess_audio(file_path, chunk_length=chunk_length, max_duration=max_duration)
            for chunk in audio_chunks:
                if len(chunk) > 0 and np.all(np.isfinite(chunk)):
                    # Tạo nhãn ngẫu nhiên cho mỗi đoạn
                    random_labels = np.random.randint(0, 32, size=len(chunk), dtype=np.int64)
                    yield {
                        "input_values": chunk.astype(np.float32),
                        "labels": random_labels
                    }
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

def prepare_dataset(audio_files, chunk_length=5, max_duration=60):
    """Chuẩn bị dataset từ các file âm thanh."""
    return IterableDataset.from_generator(lambda: audio_generator(audio_files, chunk_length, max_duration))

class CustomTrainer(Trainer):
    """Trainer tùy chỉnh để xử lý các trường hợp đặc biệt trong quá trình training."""
    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            logging.error(f"Error in compute_loss: {str(e)}")
            return torch.tensor(0.0, device=self.args.device)

class ProgressCallback(TrainerCallback):
    def __init__(self, progress):
        self.progress = progress
        self.task = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.task is None:
            self.task = self.progress.add_task("[cyan]Training", total=state.max_steps)

    def on_step_end(self, args, state, control, **kwargs):
        self.progress.update(self.task, completed=state.global_step)

def fine_tune_hubert(dataset, num_epochs=10, batch_size=2, learning_rate=1e-5, audio_files=None, args=None):
    """Hàm chính để fine-tune mô hình HuBERT."""
    console.print(Panel("Initializing model and trainer", style="bold magenta"))
    
    # Cấu hình mô hình
    config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
    config.vocab_size = 32
    model = HubertForCTC(config)
    
    # Ước tính số bước training
    total_audio_duration = sum(librosa.get_duration(path=f) for f in audio_files)
    estimated_samples = total_audio_duration / args.chunk_length
    steps_per_epoch = estimated_samples / batch_size
    max_steps = int(steps_per_epoch * num_epochs)
    
    # Thiết lập các tham số training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=1000,
        evaluation_strategy="no",
        save_total_limit=2,
        max_steps=max_steps,
    )
    
    # Khởi tạo trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    console.print(Panel("Starting fine-tuning process", style="bold green"))
    
    # Bắt đầu quá trình training
    with Progress() as progress:
        trainer.add_callback(ProgressCallback(progress))
        trainer.train()
    
    console.print(Panel("Fine-tuning completed. Saving the model", style="bold green"))
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

def main():
    """Hàm chính của chương trình."""
    parser = argparse.ArgumentParser(description="Fine-tune HuBERT model on audio files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--chunk_length", type=int, default=5, help="Length of audio chunks in seconds")
    parser.add_argument("--max_duration", type=int, default=60, help="Maximum duration of audio files in seconds")
    args = parser.parse_args()

    console.print(Panel("Starting the fine-tuning process", style="bold cyan"))

    # Hiển thị các tham số
    table = Table(title="Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Epochs", str(args.epochs))
    table.add_row("Batch Size", str(args.batch_size))
    table.add_row("Learning Rate", str(args.learning_rate))
    table.add_row("Chunk Length", f"{args.chunk_length}s")
    table.add_row("Max Duration", f"{args.max_duration}s")
    console.print(table)

    # Tìm các file WAV
    audio_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
    console.print(f"Found [bold]{len(audio_files)}[/bold] WAV files for fine-tuning")

    # Logging thời lượng của mỗi file
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing audio files", total=len(audio_files))
        for file in audio_files:
            duration = librosa.get_duration(path=file)
            logging.info(f"{os.path.basename(file)}: {duration:.2f} seconds")
            progress.update(task, advance=1)

    # Chuẩn bị dataset
    dataset = prepare_dataset(audio_files, chunk_length=args.chunk_length, max_duration=args.max_duration)

    # Hiển thị thông tin về bộ nhớ
    console.print("Initial memory usage:")
    log_memory_usage()

    # Bắt đầu quá trình fine-tuning
    fine_tune_hubert(dataset, num_epochs=args.epochs, batch_size=args.batch_size, 
                     learning_rate=args.learning_rate, audio_files=audio_files, args=args)

    # Hiển thị thông tin về bộ nhớ sau khi hoàn thành
    console.print("Final memory usage:")
    log_memory_usage()

    console.print(Panel("Fine-tuning completed. Model saved in the output directory.", style="bold green"))

if __name__ == "__main__":
    main()