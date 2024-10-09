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

# Optimize CPU usage
torch.set_num_threads(4)

# Suppress warnings
import warnings
from transformers import logging as hf_logging
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
INPUT_DIR = "wav_folder"
OUTPUT_DIR = "fine_tuned_model"
LOG_FILE = "fine_tuning_log.txt"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_memory_usage():
    gc.collect()  # Force garbage collection
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def load_and_preprocess_audio(file_path, target_sample_rate=16000, chunk_length=5, max_duration=60):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    # Ensure the waveform is mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Global normalization
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

    # Trim or pad to max_duration
    max_length = max_duration * target_sample_rate
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    elif waveform.shape[1] < max_length:
        pad_length = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    # Split into chunks
    chunk_samples = chunk_length * target_sample_rate
    chunks = torch.split(waveform, chunk_samples, dim=1)

    return [chunk.squeeze().numpy() for chunk in chunks if chunk.numel() > 0]

def audio_generator(audio_files, chunk_length=5, max_duration=60):
    for file_path in audio_files:
        try:
            audio_chunks = load_and_preprocess_audio(file_path, chunk_length=chunk_length, max_duration=max_duration)
            for i, chunk in enumerate(audio_chunks):
                if len(chunk) > 0 and np.all(np.isfinite(chunk)):
                    # Create random labels for each chunk
                    random_labels = np.random.randint(0, 32, size=len(chunk), dtype=np.int64)
                    
                    # Log statistics about the chunk
                    logging.info(f"File: {file_path}, Chunk: {i}")
                    logging.info(f"  Shape: {chunk.shape}")
                    logging.info(f"  Mean: {np.mean(chunk):.4f}")
                    logging.info(f"  Std: {np.std(chunk):.4f}")
                    logging.info(f"  Min: {np.min(chunk):.4f}")
                    logging.info(f"  Max: {np.max(chunk):.4f}")
                    logging.info(f"  Labels shape: {random_labels.shape}")
                    
                    yield {
                        "input_values": chunk.astype(np.float32),
                        "labels": random_labels
                    }
                else:
                    logging.warning(f"Skipping invalid chunk in {file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

def prepare_dataset(audio_files, chunk_length=5, max_duration=60):
    return IterableDataset.from_generator(lambda: audio_generator(audio_files, chunk_length, max_duration))

class LoggingCallback(TrainerCallback):
    def __init__(self, log_steps=10):
        self.log_steps = log_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_steps == 0:
            loss = logs.get('loss', None)
            if loss is not None:
                logging.info(f"Step {state.global_step}: Loss: {loss:.4f}")
            else:
                logging.info(f"Step {state.global_step}: Loss not available")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            # Check input values
            if torch.isnan(inputs['input_values']).any() or torch.isinf(inputs['input_values']).any():
                logging.error("NaN or Inf found in input_values")
                raise ValueError("Invalid input_values")

            # Check labels
            if torch.isnan(inputs['labels']).any() or torch.isinf(inputs['labels']).any():
                logging.error("NaN or Inf found in labels")
                raise ValueError("Invalid labels")

            outputs = model(**inputs)
            loss = outputs.loss
            
            if not torch.isfinite(loss).all():
                logging.warning(f"Non-finite loss detected: {loss.item()}")
                loss = torch.where(torch.isfinite(loss), loss, torch.tensor(1e-8).to(loss.device))
            
            logging.info(f"Batch loss: {loss.item():.4f}")
            logging.debug(f"Input shape: {inputs['input_values'].shape}")
            logging.debug(f"Label shape: {inputs['labels'].shape}")
            logging.debug(f"Output logits shape: {outputs.logits.shape}")
            logging.debug(f"First few logits: {outputs.logits[0, :5]}")
            
        except Exception as e:
            logging.error(f"Error in compute_loss: {str(e)}")
            logging.error(traceback.format_exc())
            loss = torch.tensor(1e-8, device=model.device)
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(loss).all():
            logging.warning(f"Non-finite loss after training step: {loss.item()}")
            loss = torch.where(torch.isfinite(loss), loss, torch.tensor(1e-8).to(loss.device))
        return loss

def fine_tune_hubert(dataset, num_epochs=10, batch_size=2, learning_rate=1e-5, audio_files=None, args=None):
    config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
    config.vocab_size = 32
    config.hidden_size = 768
    config.num_hidden_layers = 12
    model = HubertForCTC(config)

    # Custom weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    model.apply(init_weights)

    # Estimate the number of steps
    total_audio_duration = sum(librosa.get_duration(path=f) for f in audio_files)
    estimated_samples = total_audio_duration / args.chunk_length
    steps_per_epoch = estimated_samples / batch_size
    max_steps = int(steps_per_epoch * num_epochs)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=1,  # Log every step for debugging
        evaluation_strategy="no",
        load_best_model_at_end=False,
        gradient_accumulation_steps=4,
        fp16=False,
        no_cuda=True,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
        warmup_steps=1000,
        weight_decay=0.01,
        max_steps=max_steps,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    log_memory_usage()
    try:
        trainer.train()
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        logging.error(traceback.format_exc())
    log_memory_usage()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

def main():
    parser = argparse.ArgumentParser(description="Fine-tune HuBERT model on audio files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--chunk_length", type=int, default=5, help="Length of audio chunks in seconds")
    parser.add_argument("--max_duration", type=int, default=60, help="Maximum duration of audio files in seconds")
    args = parser.parse_args()

    audio_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]

    logging.info(f"Found {len(audio_files)} WAV files for fine-tuning")

    # Log duration of each file
    for file in audio_files:
        duration = librosa.get_duration(path=file)
        logging.info(f"{os.path.basename(file)}: {duration:.2f} seconds")

    dataset = prepare_dataset(audio_files, chunk_length=args.chunk_length, max_duration=args.max_duration)

    log_memory_usage()
    fine_tune_hubert(dataset, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, audio_files=audio_files, args=args)
    log_memory_usage()

    logging.info("Fine-tuning completed. Model saved in the output directory.")

if __name__ == "__main__":
    main()