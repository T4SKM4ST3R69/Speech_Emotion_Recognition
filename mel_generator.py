import os
import numpy as np
import librosa
import random
import argparse
from tqdm import tqdm

# Shift audio in time
def time_shift(audio, shift_factor):
    shift = int(len(audio) * shift_factor)
    if shift > 0:
        return np.pad(audio, (0, shift), mode='constant')[shift:]
    else:
        return np.pad(audio, (-shift, 0), mode='constant')[:shift]

# Shift pitch by n semitones
def pitch_shift(audio, sr, n_steps):

    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

#Stretch audio by rate factor
def time_stretch(audio, rate):

    return librosa.effects.time_stretch(audio, rate=rate)

# Add random noise to audio
def add_noise(audio, noise_factor):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


# Extract emotion from [ActorID]_[UtteranceID]_[Emotion]_[Intensity].wav
def get_emotion(filename):
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[2]
    return None


# Generate and save mel spectrogram as .npy file
def create_mel_spectrogram(audio, sr, output_path, filename, augmentation_type="original",
                           n_fft=2048, hop_length=512, n_mels=128, fmax=8000):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0,
        n_mels=n_mels,
        fmax=fmax
    )

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Create filename for the spectrogram
    base_filename = os.path.splitext(filename)[0]
    save_filename = f"{base_filename}_{augmentation_type}.npy"
    save_path = os.path.join(output_path, save_filename)

    # Save as numpy file
    np.save(save_path, log_mel_spec)
    return save_path

#Process all audio files and create mel spectrograms with augmentation
def process_audio_files(input_dir, output_dir, apply_augmentation=True, n_augmentations=3):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for each emotion
    emotion_dirs = {}
    for emotion in ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']:
        emotion_path = os.path.join(output_dir, emotion)
        os.makedirs(emotion_path, exist_ok=True)
        emotion_dirs[emotion] = emotion_path

    # Get all wav files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    # Define available augmentation techniques
    augmentation_techniques = [
        ("time_shift_pos", lambda a, s: time_shift(a, 0.1)),
        ("time_shift_neg", lambda a, s: time_shift(a, -0.1)),
        ("pitch_up", lambda a, s: pitch_shift(a, s, 2.0)),
        ("pitch_down", lambda a, s: pitch_shift(a, s, -2.0)),
        ("stretch_fast", lambda a, s: time_stretch(a, 1.2)),
        ("stretch_slow", lambda a, s: time_stretch(a, 0.8)),
        ("noise_low", lambda a, s: add_noise(a, 0.005)),
        ("noise_med", lambda a, s: add_noise(a, 0.01))
    ]

    for filename in tqdm(audio_files, desc="Processing audio files"):
        emotion = get_emotion(filename)
        if not emotion or emotion not in emotion_dirs:
            continue

        # Determine output path based on emotion
        output_path = emotion_dirs[emotion]

        try:
            # Load audio file
            audio_path = os.path.join(input_dir, filename)
            y, sr = librosa.load(audio_path, sr=None)  # Use original sample rate

            # Create mel spectrogram for original audio
            create_mel_spectrogram(y, sr, output_path, filename)

            # Apply augmentations if specified
            if apply_augmentation:
                # Select random augmentation techniques
                selected_augmentations = random.sample(
                    augmentation_techniques,
                    min(n_augmentations, len(augmentation_techniques))
                )

                # Apply selected augmentations
                for aug_name, aug_func in selected_augmentations:
                    try:
                        augmented_audio = aug_func(y, sr)
                        create_mel_spectrogram(augmented_audio, sr, output_path,
                                               filename, aug_name)
                    except Exception as e:
                        print(f"Error applying {aug_name} to {filename}: {e}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mel spectrograms from CREMA-D dataset")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing CREMA-D audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save mel spectrograms")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--n_augmentations", type=int, default=3,
                        help="Number of augmentations per audio file")

    args = parser.parse_args()

    # Process files
    process_audio_files(
        args.input_dir,
        args.output_dir,
        apply_augmentation=not args.no_augment,
        n_augmentations=args.n_augmentations
    )

    print(f"All mel spectrograms saved to {args.output_dir}")
