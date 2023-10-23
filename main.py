import whisper
import argparse

parser = argparse.ArgumentParser(description="Naver Cafe crawling operator")

parser.add_argument(
    "--type",
    required=True,
    help="madow whisper decode, file",
    default=None,
)

parser.add_argument(
    "--file",
    required=True,
    help="madow whisper file path",
    default=None,
)

parser.add_argument(
    "--model",
    required=True,
    help="madow whisper model",
    default='small',
)

args = parser.parse_args()

def progress_segments(segments, segment_size, left_segment, rate):
    print(segment_size, left_segment, rate)
    return True

if __name__ == "__main__":
    if args.type == "decode" or args.type == None:
        model = whisper.load_model(args.model)
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(args.file)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        # decode the audio
        options = whisper.DecodingOptions(fp16 = False)
        result = whisper.decode(model, mel, options)
        # print the recognized text
        print(result.text)
    elif args.type =='file':
        print(args)
        model = whisper.load_model(args.model)
        result = whisper.transcribe(model = model, audio=args.file, progress=progress_segments)
        print("end")