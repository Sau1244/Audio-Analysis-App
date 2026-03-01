import argparse
import logging
from dataclasses import fields
from pathlib import Path

import torch

from AI_dio.audio.audio_file_reader import (
    compute_log_mel_spectrogram,
    get_sound_parameters,
    plot_melspectrogram,
    plot_waveform,
    read_sound,
)
from AI_dio.audio.microphone_input import microphone_input
from AI_dio.inference import predict_audio, predict_file

logging.getLogger().setLevel(logging.CRITICAL)

ROOT = Path(__file__).parents[3].resolve()
CHECKPOINT_PATH = ROOT / "checkpoints/model_best.pt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick audio analysis")

    source_group = parser.add_mutually_exclusive_group(required=True)

    source_group.add_argument(
        "-f", "--file", type=str, help="Audio file to analyse, .mp3 and .wav supported"
    )

    source_group.add_argument(
        "-m",
        "--microphone",
        action="store_true",
        help="Record sound from the microphone instead of using a file",
    )

    parser.add_argument(
        "-wv",
        "--plot_waveform",
        action="store_true",
        help="Generate and save a waveform of given audio",
    )
    parser.add_argument(
        "-wvf",
        "--waveform_file",
        type=str,
        default="waveform.png",
        help="Filename for the waveform. Default: waveform.png",
    )
    parser.add_argument(
        "-sp",
        "--plot_spectrogram",
        action="store_true",
        help="Generate and save a Mel spectrogram of the given audio",
    )
    parser.add_argument(
        "-spf",
        "--spectrogram_file",
        type=str,
        default="spectrogram.png",
        help="Filename for the spectrogram. Default: spectrogram.png",
    )
    parser.add_argument(
        "-ms",
        "--microphone_seconds",
        type=int,
        default=5,
        help="Specify the numbers of seconds to record when using a microphone. Default: 5",
    )
    parser.add_argument(
        "-msr",
        "--microphone_sample_rate",
        type=int,
        default=44100,
        help="Specify the sample rate when using a microphone. Default: 44100",
    )
    parser.add_argument(
        "-np",
        "--no_parameters",
        action="store_true",
        help="Skip printing sound parameters",
    )
    parser.add_argument(
        "-ai",
        "--ai_analysis",
        action="store_true",
        help="Analyse the audio sample with AI, to check if it's real.",
    )
    parser.add_argument(
        "-chck",
        "--checkpoint",
        type=Path,
        help="Path to the model (only valid with --ai_analysis). Default: checkpoints/model_best.pt",
    )

    args = parser.parse_args()

    if args.checkpoint and not args.ai_analysis:
        parser.error("--checkpoint can only be used with --ai_analysis")

    if args.file:
        audio, parameters = read_sound(Path(args.file))
    elif args.microphone:
        audio, rate = microphone_input(
            args.microphone_seconds, rate=args.microphone_sample_rate
        )
        parameters = get_sound_parameters(audio, rate)

    if not args.no_parameters:
        for key, value in parameters.items():
            print(f"{key}: {value}")

    if args.ai_analysis:
        checkpoint_path = args.checkpoint if args.checkpoint else CHECKPOINT_PATH

        if args.file:
            result = predict_file(checkpoint=checkpoint_path, wav=args.file)
        elif args.microphone:
            audio_tensor = torch.tensor(audio)
            result = predict_audio(
                checkpoint=checkpoint_path, audio=audio_tensor, sample_rate=rate
            )

        print("AI Result:")
        for field in fields(result):
            if field.name == "wav":
                continue
            print(f"{field.name}:", getattr(result, field.name))

    if args.plot_waveform:
        plot_waveform(audio, Path(args.waveform_file))
        print(f"Waveform saved to {args.waveform_file}")

    if args.plot_spectrogram:
        spectrogram = compute_log_mel_spectrogram(audio, parameters["sample_rate"])
        plot_melspectrogram(
            spectrogram,
            parameters["sample_rate"],
            output_path=Path(args.spectrogram_file),
        )
        print(f"Spectogram saved to {args.spectrogram_file}")
