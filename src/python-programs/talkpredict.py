from scipy.spatial import distance
import librosa
import time
import numpy as np
import torch
from torch.autograd import Variable
from model import Model
from decoder import GreedyDecoder
import scipy.signal
import dlib
import cv2
import imutils
import pandas as pd
import math
import random

class talkPredictor:

	FACE_MODEL_PATH = './model/shape_predictor_68_face_landmarks.dat'
	RNN_MODEL_PATH = './model/rnn_model.pth'

	def __init__(self, purge=True):
		self.facedetector = dlib.get_frontal_face_detector()
		self.facepredictor = dlib.shape_predictor(talkPredictor.FACE_MODEL_PATH)
		self.start_time = int(round(time.time() * 1000))
		self.log = pd.DataFrame(data=[], columns=['ts','key','value'])
		self.log.set_index(['ts', 'key'])
		self.purge=purge
		self.procdevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.talkmodel = Model.load_model(talkPredictor.RNN_MODEL_PATH)  # , map_location='cpu'
		self.talkmodel.eval()
		self.talklabels = Model.get_labels(self.talkmodel)
		self.talkdecoder = GreedyDecoder(self.talklabels, blank_index=self.talklabels.index('_'))
		self.audio_conf = Model.get_audio_conf(self.talkmodel)
		self.samplerate = 16000
		self.framerate = None
		self.video_queue = {}
		self.audio_queue = []
		self.pred_queue = []

	# Used the following code as reference: http://dlib.net/face_landmark_detection.py.html
	def detect_faces(self,  frame, resize_to_width=None, use_gray=True):
		# Faster prediction when frame is resized
		if resize_to_width is not None:
			frame = imutils.resize(frame, width=resize_to_width)
		# If use_gray = True then convert frame used for detection in to grayscale
		if use_gray:
			dframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			dframe = frame

		#Detect faces in frame
		faces_loc = self.facedetector(dframe, 0)

		self.purge_from_log(10000, 'numfaces')
		self.push_to_log('numfaces', len(faces_loc))

		return faces_loc

	def dlib_shape_to_points(self, shape, dtype=np.int32):
		points = np.zeros((68, 2), dtype=dtype)

		for j in range(0, 68):
			points[j] = (shape.part(j).x,shape.part(j).y)

		return points

	def pred_points_on_face(self, frame, face_loc, face=0, use_gray=True):
		# If use_gray = True then convert frame used for prediction in to grayscale
		if use_gray:
			pframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			pframe = frame

		shape = self.facepredictor(pframe, face_loc)
		points = self.dlib_shape_to_points(shape)

		self.purge_from_log(10000, ('fleft'+str(face)))
		self.push_to_log(('fleft'+str(face)), face_loc.left())
		self.purge_from_log(10000, ('ftop'+str(face)))
		self.push_to_log(('ftop'+str(face)), face_loc.top())
		self.purge_from_log(10000, ('fright'+str(face)))
		self.push_to_log(('fright'+str(face)), face_loc.right())
		self.purge_from_log(10000, ('fbottom'+str(face)))
		self.push_to_log(('fbottom'+str(face)), face_loc.bottom())

		return points

	def draw_points_on_face(self, frame, points, color):
		for (x, y) in points:
			cv2.circle(frame, (x, y), 1, color, -1)
		return frame

	def pad_tensor(self, vec, pad, dim):
		pad_size = list(vec.shape)
		pad_size[dim] = pad - vec.size(dim)
		return torch.cat([vec.float(), torch.zeros(*pad_size)], dim=dim)

	def get_current_ts(self):
		ts = int(round(time.time() * 1000)) - self.start_time
		return ts

	def push_to_log(self, key, value):
		ts = self.get_current_ts()
		self.log = self.log.append({'ts': ts, 'key':key, 'value':value}, ignore_index=True)
		return self.log['ts'].count()

	def purge_from_log(self, ts_threshold, key):
		if self.purge:
			ts = self.get_current_ts() - ts_threshold
			self.log = self.log.drop(self.log[(self.log.ts < ts) & (self.log.key == key)].index)
		return self.log['ts'].count()

	def fetch_log(self, key=None, ts_threshold=None):
		log = None
		if ts_threshold is None:
			if key is None:
				log = self.log
			else:
				log = self.log[self.log.key == key]
		else:
			ts = self.get_current_ts() - ts_threshold
			if key is None:
				log = self.log[(self.log.ts < ts)]
			else:
				log = self.log[(self.log.ts < ts) & (self.log.key == key)]
		return log

	def est_mouth_openess(self, points, face=0):
		lip_top_bottom = distance.euclidean(points[51], points[57])
		mouth_top_bottom = distance.euclidean(points[62], points[66])
		top_bottom = np.mean([lip_top_bottom, mouth_top_bottom])
		lip_left_right = distance.euclidean(points[48], points[54])
		mouth_left_right = distance.euclidean(points[60], points[64])
		left_right = np.mean([lip_left_right, mouth_left_right])
		mouth_ratio = top_bottom / left_right
		self.purge_from_log(10000, ('mar'+str(face)))
		self.push_to_log(('mar'+str(face)), mouth_ratio)
		return mouth_ratio

	def draw_bounding_box(self, frame, face=0, color=(0, 0, 255), ts_threshold=1500):
		ts = self.get_current_ts() - ts_threshold
		ftop_min = self.log[(self.log.ts > ts) & (self.log.key == ('ftop' + str(face)))]['value'].min()
		fbottom_max = self.log[(self.log.ts > ts) & (self.log.key == ('fbottom' + str(face)))]['value'].max()
		fleft_min = self.log[(self.log.ts > ts) & (self.log.key == ('fleft' + str(face)))]['value'].min()
		fright_max = self.log[(self.log.ts > ts) & (self.log.key == ('fright' + str(face)))]['value'].max()
		bounding_points = np.array([[fleft_min, ftop_min], [fright_max, ftop_min], [fright_max, fbottom_max], [fleft_min, fbottom_max]])
		try:
			faceHull = cv2.convexHull(bounding_points)
			cv2.drawContours(frame, [faceHull], -1, color, 1)
			cv2.putText(frame, "(%.1f, %.1f)" % tuple(np.mean(bounding_points, axis=0)), tuple([fright_max-95,ftop_min+15]), #/"{:.2f}".format(mar)
						cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), thickness=1)
		except Exception as e:
			pass
		return frame

	def draw_mouth(self, frame, points, mar, color):
		mouth_points = points[48:59]
		mouthHull = cv2.convexHull(mouth_points)
		cv2.drawContours(frame, [mouthHull],-1, color, 1)
		cv2.putText(frame, "{:.2f}".format(mar), tuple([points[57][0]-15,points[57][1]+10]),
					cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), thickness=1)
		return frame

	def get_mouth_openess_over_time(self, face=0, ts_threshold=3000, mar_threshold=0.14, talk_threshold=0.12):
		ts = self.get_current_ts() - ts_threshold
		df_mar = self.log[(self.log.ts > ts) & (self.log.key == ('mar'+str(face)))]
		is_talking = False
		count = df_mar['value'].count()
		max_val = None
		mean_val = None
		if count > round(ts_threshold / 200):
			max_val = df_mar['value'].max()
			if max_val > mar_threshold:
				mean_val = df_mar['value'].mean()
				is_talking = mean_val > talk_threshold
		return is_talking, count, max_val, mean_val

	def draw_text(self, frame, text, color, pos=None, face=0, y_offset=-5, x_offset=10, ts_threshold=1500):
		if pos is None:
			ts = self.get_current_ts() - ts_threshold
			fbottom_max = self.log[(self.log.ts > ts) & (self.log.key == ('fbottom' + str(face)))]['value'].max()
			fleft_min = self.log[(self.log.ts > ts) & (self.log.key == ('fleft' + str(face)))]['value'].min()
			pos = (fleft_min + x_offset, fbottom_max + y_offset)
		cv2.putText(frame, text, pos,
		            cv2.FONT_HERSHEY_PLAIN, 0.9, color, thickness=1)
		return frame

	def draw_subtitles(self, frame, text, pos1=None, face=0, bg=(0, 0, 0), fg=(255, 255, 255), alpha=0.7, y_offset=10, x_offset=-50, width=200, height=50, ts_threshold=5000):
		if pos1 is None:
			ts = self.get_current_ts() - ts_threshold
			fbottom_max = self.log[(self.log.ts > ts) & (self.log.key == ('fbottom' + str(face)))]['value'].max()
			fleft_min = self.log[(self.log.ts > ts) & (self.log.key == ('fleft' + str(face)))]['value'].min()
			fright_max = self.log[(self.log.ts > ts) & (self.log.key == ('fright' + str(face)))]['value'].max()
			pos1 = (fleft_min + x_offset, fbottom_max + y_offset)
			width = round((fright_max - fleft_min) + 100)
			#print("%s, %s, %s, %s" % (fbottom_max, fleft_min, fright_max, width))

		pos2 = (pos1[0] + width, pos1[1] + height)
		tpos = (pos1[0] + 10, pos1[1] + 20)
		#print("%s\t->\t%s\t\t%s" % (pos1, pos2, tpos))
		overlay = frame.copy()
		output = frame.copy()
		cv2.rectangle(overlay, pos1, pos2, bg, -1)
		cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
		if type(text) is list:
			for t in text:
				cv2.putText(output, t, tpos, cv2.FONT_HERSHEY_PLAIN, 1.0, fg, 1)
				tpos = (tpos[0], tpos[1] + 25)
		else:
			cv2.putText(output, text, tpos, cv2.FONT_HERSHEY_PLAIN, 1.0, fg, 1)
		return output


	def crop_frame_to_bounding(self, frame, face_loc):
		r = int(math.ceil(face_loc.right()))
		l = int(math.floor(face_loc.left()))
		h = r - l
		b = int(math.ceil(face_loc.bottom()))
		t = int(math.floor(face_loc.top()))
		w = b - t
		if w > h:
			t = int(math.floor(t - ((w - h)/2)))
			if t < 0: t = 0
			b = t + w
			h = w
		elif w < h:
			l = int(math.floor(l - ((h - w)/2)))
			if l < 0: l = 0
			r = l + h
			w = h
		image = frame.copy()
		#image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
		image = image[int(t):int(b), int(l):int(r)]
		image = imutils.resize(image, 121, 121)
		image = image[0:120, 0:120]
		return image

	def length_queues(self):
		return len(self.audio_queue), len(self.pred_queue)

	def add_to_video_queue(self, frame, face_loc, f=0):
		img = self.crop_frame_to_bounding(frame, face_loc)
		if f not in self.video_queue.keys():
			self.video_queue[f] = []
		self.video_queue[f].append(img)

	def add_to_audio_queue(self, wave):
		self.audio_queue.append(wave)

	def decode_results(self, decoded_output, decoded_offsets):
		results = {
			"output": [],
		}

		for b in range(len(decoded_output)):
			for pi in range(min(1, len(decoded_output[b]))):
				result = {'transcription': decoded_output[b][pi]}
				result['offsets'] = decoded_offsets[b][pi]
				results['output'].append(result)
		return results

	def prepare_queue(self, f=0, increase_volume=1, stretch=1):
		frames = np.empty([1, 120, 120, 3])
		for v in range(len(self.video_queue[f])):
			frames = np.append(frames, self.video_queue[f][v].reshape((1, 120, 120, 3)), axis=0)

		wave = None
		for a in range(len(self.audio_queue)):
			if a == 0:
				wave = self.audio_queue[a]
			else:
				wave = np.concatenate((wave, self.audio_queue[a]), axis=0)
		if increase_volume > 1:
			wave = wave * random.uniform(increase_volume, increase_volume*1.133333)
		if stretch != 1:
			wave = librosa.effects.time_stretch(wave.astype('float'), stretch)

		return frames, wave

	def clear_av_queue(self, f=0):
		self.audio_queue = []
		self.video_queue[f] = []

	def clear_pred_queue(self):
		self.pred_queue = []

	def pred_talk(self, frames, wave, f=0, normaudio = True, separateVocals = False):
		window = scipy.signal.hamming
		n_fft = win_length = int(self.audio_conf['sample_rate'] * self.audio_conf['window_size'])
		hop_length = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])
		spectf = librosa.feature.melspectrogram(wave, self.samplerate)
		D = librosa.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
		spect, phase = librosa.magphase(D)
		if separateVocals:
			margin_i, margin_v = 2, 10
			power = 2
			sfilter = librosa.decompose.nn_filter(spect, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=samplerate)))
			sfilter = np.minimum(spect, sfilter)
			mask_i = librosa.util.softmask(sfilter, margin_i * (spect - sfilter), power=power)
			mask_v = librosa.util.softmask(spect - sfilter, margin_v * sfilter, power=power)
			sforeground = mask_v * spect
			#sbackground = mask_i * spect
			spect = librosa.istft(sforeground)
		spect = np.log1p(spect)
		spect = torch.FloatTensor(spect)
		if normaudio:
			mean = spect.mean()
			std = spect.std()
			spect.add_(-mean)
			spect.div_(std)
		spect = spect.view(1, 1, spect.size(0), spect.size(1))
		#print(spect.size())
		#if frames is not None and wave is not None:
		#	frames = self.pad_tensor(frames, spect.size(3), 0)
		out = self.talkmodel(Variable(spect, volatile=False))
		# out, output_sizes = self.talkmodel(spect_std.type('torch.FloatTensor').to(device))
		out = out.transpose(0, 1)  # TxNxH
		#print(out.size())
		decoded_output, decoded_offsets = self.talkdecoder.decode(out.data)
		pred = decoded_output[0][0]
		if len(pred) > 0:
			self.pred_queue.append(pred)
			self.push_to_log(('preds'+str(f)), pred)
		return pred
