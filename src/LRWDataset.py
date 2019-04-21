import torch
import imageio
from python_speech_features import mfcc, delta, logfbank
import librosa
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class LRWDataset(Dataset):
    COL_MP4 = 'mp4'
    COL_MP3 = 'mp3'
    COL_TXT = 'txt'
    
    def __init__(self, root_dir, clean_files_path, is_train=True, is_dev=False):
        self.root_dir = root_dir
        self.df = self._get_files(root_dir, clean_files_path, is_train, is_dev)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        mp4, mp3, txt = self._get_records(idx)
        
        reversed_mp3 = self._get_mp3_as_tensor(self.root_dir + mp3)
        reversed_txt = self._get_txt_as_tensor(self.root_dir + txt)
        reversed_mp4 = self._get_frames_as_tensors(self.root_dir + mp4)
        
        return reversed_mp4, reversed_mp3, reversed_txt

    def _get_files(self, root_dir, file_path, is_train=True, is_dev=False):
        df = pd.read_csv(root_dir + file_path)
        if is_dev:
            return df
        if is_train:
            return df[df['is_train'] == 1]
        else:
            return df[df['is_train'] == 0]
    
    def _get_records(self, idx):
        record = self.df.iloc[idx]
        mp4 = record[LRWDataset.COL_MP4]
        mp3 = record[LRWDataset.COL_MP3]
        txt = record[LRWDataset.COL_TXT]
        
        return mp4, mp3, txt

    def _get_reversed_txt_as_tensor(self, txt_file):
        ascii = self._get_txt_as_tensor(txt_file)
        rev_ascii = torch.flip(ascii, [0])
        return rev_ascii

    def _get_txt_as_tensor(self, txt_file):
        with open(txt_file, 'r') as f:
             content = f.readline()
        ascii = np.array([ord(c) - 32 for c in content.replace('Text:', '').strip()])
        ascii = torch.autograd.Variable(torch.from_numpy(ascii.astype(int)).int())
        return ascii
    
    def _get_reversed_frames_as_tensors(self, mp4_file):
        frames = self._get_frames_as_tensors(mp4_file)
        rev_frames = torch.flip(frames, [0])
        return rev_frames

    def _get_frames_as_tensors(self, mp4_file):
        reader = imageio.get_reader(mp4_file)
        imgs = np.array(reader.get_data(0))
        imgs = imgs.reshape(1, *imgs.shape)
        count = reader.count_frames()
        for i in range(1, count):
            frame = np.array(reader.get_data(i))
            frame = frame.reshape(1, *frame.shape)
            imgs = np.vstack((imgs, frame))
        frames = torch.from_numpy(imgs)
        return frames

    def _get_reversed_mp3_as_tensor(self, mp3_file, dim=13, window_size=25, stride=10, method='psf'):
        windows = self._get_mp3_as_tensor(mp3_file)
        rev_windows = torch.flip(windows, [1])
        return rev_windows

    def _get_mp3_as_tensor(self, mp3_file, dim=13, window_size=25, stride=10, method='psf'):
        if method == 'psf':
            feat = self._get_audio_feat_psf(mp3_file, dim, window_size, stride)
        else:
            feat = self._get_audio_feat_librosa(mp3_file, dim, window_size, stride)
        mfcc = zip(*feat)
        mfcc = np.stack([np.array(i) for i in mfcc])
        #cc = np.expand_dims(np.expand_dims(mfcc, axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(mfcc.astype(float)).float())
        return cct

    def _get_audio_feat_psf(self, mp3_file, dim=13, window_size=25, stride=10):
        sig, rate = librosa.load(mp3_file, sr=None)
        feat = mfcc(sig, samplerate=rate, numcep=dim, winlen=window_size/1000, winstep=stride/1000)
        return feat

    def _get_audio_feat_librosa(self, mp3_file, dim=13, window_size=25, stride=10,
                               feature='mfcc', cmvn=False, delta=False, delta_delta=False, save_feature=None):
        y, sr = librosa.load(mp3_file, sr=None)
        ws = int(sr * 0.001 * window_size)
        st = int(sr * 0.001 * stride)
        if feature == 'fbank':  # log-scaled
            feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                                  n_fft=ws, hop_length=st)
            feat = np.log(feat + 1e-6)
        elif feature == 'mfcc':
            feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                        n_fft=ws, hop_length=st)
            feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws)

        else:
            raise ValueError('Unsupported Acoustic Feature: ' + feature)

        feat = [feat]
        if delta:
            feat.append(librosa.feature.delta(feat[0]))

        if delta_delta:
            feat.append(librosa.feature.delta(feat[0], order=2))
        feat = np.concatenate(feat, axis=0)
        if cmvn:
            feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
        if save_feature is not None:
            tmp = np.swapaxes(feat, 0, 1).astype('float32')
            np.save(save_feature, tmp)
            return len(tmp)
        else:
            return np.swapaxes(feat, 0, 1).astype('float32')