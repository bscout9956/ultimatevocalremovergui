import argparse
import os

import cv2
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils

# Variable manipulation and command line text parsing
import torch
import tkinter as tk
import traceback  # Error Message Recent Calls


class Namespace:
    """
    Replaces ArgumentParser
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(window: tk.Wm, input_paths: list, gpu: bool = -1,
         model: str = 'models/baseline.pth', sr: int = 44100, hop_length: int = 1024,
         window_size: int = 512, n_fft: int = 2048,
         export_path: str = '', loops: int = 1,
         # Other Variables (Tkinter)
         progress_var: tk.Variable = None, button_widget: tk.Button = None, command_widget: tk.Text = None,
         ):
    def reconstruct(X, window_size, model, device, roll=False):
        l, r, roi_size = dataset.make_padding(X.shape[2], window_size, model.offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')

        if roll:
            X_pad = np.roll(X_pad, roi_size // 2, axis=2)

        model.eval()
        with torch.no_grad():
            preds = []
            for i in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
                start = i * roi_size
                X_window = torch.from_numpy(np.asarray([
                    X_pad[:, :, start:start + window_size],
                ])).to(device)

                pred = model.predict(X_window)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)[:, :, :X.shape[2]]

        if roll:
            pred = np.roll(pred, -roi_size // 2, axis=2)

        return pred  

    def load_model():
        args.command_widget.write('Loading model...\n')  # nopep8 Write Command Text
        device = torch.device('cpu')
        model = nets.CascadedASPPNet(args.n_fft)
        model.load_state_dict(torch.load(args.model, map_location=device))
        if torch.cuda.is_available() and args.gpu >= 0:
            device = torch.device('cuda:{}'.format(args.gpu))
            model.to(device)
        args.command_widget.write('Done!\n')  # nopep8 Write Command Text

        return model, device

    def load_wave_source():
        args.command_widget.write(base_text + 'Loading wave source...\n')  # nopep8 Write Command Text
        X, sr = librosa.load(music_file,
                             args.sr,
                             False,
                             dtype=np.float32,
                             res_type='kaiser_fast')
        args.command_widget.write(base_text + 'Done!\n')  # nopep8 Write Command Text

        return X, sr

    def stft_wave_source(X):
        args.command_widget.write(base_text + 'Stft of wave source...\n')  # nopep8 Write Command Text
        X = spec_utils.get_spectrogram(X, args.hop_length, args.n_fft)
        X_mag, X_phase = np.abs(X), np.angle(X)

        pred = reconstruct(X_mag, args.window_size, model, device)
        pred_roll = reconstruct(X_mag, args.window_size, model, device, roll=True)
        pred = (pred + pred_roll) / 2
        args.command_widget.write(base_text + 'Inverse stft of instruments and vocals...\n')
        y_spec = pred * np.exp(1.j * X_phase)
        v_spec = np.clip(X_mag - pred, 0, np.inf) * np.exp(1.j * X_phase)

        wav_instrument = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length) # nopep8
        wav_vocals = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length) # nopep8

        args.command_widget.write(base_text + 'Saving Files...\n')  # nopep8 Write Command Text
        sf.write(f'{export_path}/{base_name}_(Instrumental).wav',
                 wav_instrument.T, sr)
        if cur_loop == 0:
            sf.write(f'{export_path}/{base_name}_(Vocals).wav',
                     wav_vocals.T, sr)
        if (cur_loop == (args.loops - 1) and
                args.loops > 1):
            sf.write(f'{export_path}/{base_name}_(Last_Vocals).wav',
                     wav_vocals.T, sr)

        args.command_widget.write(base_text + 'Done!\n')  # nopep8 Write Command Text

    args = Namespace(input=input_paths, gpu=gpu, model=model,
                     sr=sr, hop_length=hop_length, window_size=window_size,
                     n_fft=n_fft, export=export_path,
                     loops=loops,
                     # Other Variables (Tkinter)
                     window=window, progress_var=progress_var,
                     button_widget=button_widget, command_widget=command_widget,
                     )
    args.command_widget.clear()  # Clear Command Text
    args.button_widget.configure(state=tk.DISABLED)  # Disable Button
    total_files = len(args.input)  # Used to calculate progress

    model, device = load_model()

    for file_num, music_file in enumerate(args.input, start=1):
        try:
            base_name = f'{file_num}_{os.path.splitext(os.path.basename(music_file))[0]}'
            for cur_loop in range(args.loops):
                if cur_loop > 0:
                    args.command_widget.write(f'File {file_num}/{total_files}:  ' + 'Next Pass!\n')  # nopep8 Write Command Text
                    music_file = f'{export_path}/{base_name}_(Instrumental).wav'
                base_progress = 100 / \
                    (total_files*args.loops) * \
                    ((file_num*args.loops)-((args.loops-1) - cur_loop)-1)
                base_text = 'File {file_num}/{total_files}:{loop} '.format(
                    file_num=file_num,
                    total_files=total_files,
                    loop='' if args.loops <= 1 else f' ({cur_loop+1}/{args.loops})')
                max_progress = 100 / (total_files*args.loops)
                progress_var.set(base_progress + max_progress * 0.05)  # nopep8 Update Progress

                X, sr = load_wave_source()
                progress_var.set(base_progress + max_progress * 0.1)  # nopep8 Update Progress

                X_phase = stft_wave_source(X)
                progress_var.set(base_progress + max_progress * 0.7)  # nopep8 Update Progress

            args.command_widget.write(base_text + 'Completed Seperation!\n\n')  # nopep8 Write Command Text
        except Exception as e:
            traceback_text = ''.join(traceback.format_tb(e.__traceback__))
            print(traceback_text)
            print(type(e).__name__, e)
            tk.messagebox.showerror(master=args.window,
                                    title='Untracked Error',
                                    message=f'Traceback Error: "{traceback_text}"\n{type(e).__name__}: "{e}"\nFile: {music_file}\n\nPlease contact the creator and attach a screenshot of this error with the file which caused it!')
            args.button_widget.configure(state=tk.NORMAL)  # Enable Button
            return

    progress_var.set(100)  # Update Progress
    args.command_widget.write(f'Conversion(s) Completed and Saving all Files!')  # nopep8 Write Command Text
    args.button_widget.configure(state=tk.NORMAL)  # Enable Button
