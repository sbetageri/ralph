import imutils
import cv2
import sys
import tkinter as tk
from tkinter import messagebox
import librosa
import numpy as np
import sounddevice as sd
import pandas as pd
import talkpredict
import time as tm


class demo1:

	FRAME_WIDTH = 750
	WINDOW_TITLE = "Demo #1: live talk prediction"

	K_ESC = 27
	K_QUIT = ord('q')
	K_POINTS = ord('p')
	K_BOUNDING = ord('b')
	K_MOUTH = ord('m')
	K_NONE = ord('n')
	K_REFRESH = ord('r')
	K_SAVE_LOG = ord('l')
	K_HELP = ord('h')

	LOG_PATH = './transcript_%ts.csv'

	def __init__(self):
		self.rootwin = tk.Tk()
		self.rootwin.withdraw()
		cv2.namedWindow(demo1.WINDOW_TITLE)
		self.show_points = True
		self.show_bounding = True
		self.show_mar = True
		self.samplerate = None
		self.talkpredict = talkpredict.talkPredictor()

	def run(self):

		self.samplerate = sd.query_devices(None, 'input')['default_samplerate']


		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			print("Unable to connect to camera.")
			return

		with sd.InputStream(device=None, channels=1, callback=self.process_audio,
		                    blocksize=int(self.talkpredict.samplerate * 440 / 1000),
		                    samplerate=self.talkpredict.samplerate):
			while self.cap.isOpened():
				self.key_strokes_handler()
				ret, frame = self.cap.read()
				if ret:
					frame = imutils.resize(frame, width=demo1.FRAME_WIDTH)
					frame = self.process_video(frame)
					cv2.imshow(demo1.WINDOW_TITLE, frame)
					cv2.moveWindow(demo1.WINDOW_TITLE, 0, 0)

	def process_audio(self, indata, frames, time, status):
		wave = indata
		#wave, samplerate = librosa.load('./data/1KFUSfbcwfw50001.wav')

		if len(wave.shape) > 1:
			if wave.shape[1] == 1:
				wave = wave.squeeze()
			else:
				wave = wave.mean(axis=1)  # multiple channels, average

		self.talkpredict.add_to_audio_queue(wave)

	def process_video(self, frame=None):

		faces_loc = self.talkpredict.detect_faces(frame, None, True)
		numfaces = len(faces_loc)

		if numfaces > 0:
			for f in range(numfaces):
				face_loc = faces_loc[f]

				self.talkpredict.add_to_video_queue(frame, face_loc, f)

				#Predict coordinates of 68 points of this face using ML trained model
				points = self.talkpredict.pred_points_on_face(frame, face_loc)

				mar = self.talkpredict.est_mouth_openess(points, f)
				is_talking, _, _, _ = self.talkpredict.get_mouth_openess_over_time(f,2500)
				was_talking, _, _, _ = self.talkpredict.get_mouth_openess_over_time(f,5000)

				len_av_queue, len_pred_queue = self.talkpredict.length_queues()
				print("%s: %s" % (self.talkpredict.get_current_ts(), len_pred_queue))

				if is_talking or len_av_queue > 5:
					if is_talking:
						frame = self.talkpredict.draw_text(frame, "TALKING", (0, 0, 255), face=f)
					frames, wave = self.talkpredict.prepare_queue(f)
					pred = self.talkpredict.pred_talk(frames, wave, f)
					print("\t" + pred)
					len_av_queue, len_pred_queue = self.talkpredict.length_queues()
					if len(pred):
						frame = self.talkpredict.draw_subtitles(frame, list(reversed(self.talkpredict.pred_queue)), face=f, bg=(0, 0, 0), fg=(255, 255, 255), alpha=0.7, height=(len_pred_queue*30))
					if len(pred) > 4 or len_av_queue > 10:
						print("\t[CLR AVQ]")
						self.talkpredict.clear_av_queue()

				if self.show_bounding:
					frame = self.talkpredict.draw_bounding_box(frame, f, (0, 0, 255))

				if self.show_points:
					frame = self.talkpredict.draw_points_on_face(frame, points, (0, 0, 255))

				if self.show_mar:
					frame = self.talkpredict.draw_mouth(frame, points, mar, (0, 0, 255))

				if len_pred_queue > 0:
					frame = self.talkpredict.draw_subtitles(frame, list(reversed(self.talkpredict.pred_queue)), face=f, bg=(0, 0, 0), fg=(255, 255, 255), alpha=0.7, height=(len_pred_queue*30))

				if len_pred_queue > 3:
					print("\t[CLR PQ]")
					self.talkpredict.clear_pred_queue()

		return frame

	def key_strokes_handler(self):
		pressed_key = cv2.waitKey(1) & 0xFF

		if pressed_key == demo1.K_ESC or pressed_key == demo1.K_QUIT:
			print('-> QUIT')
			self.cap.release()
			cv2.destroyAllWindows()
			sys.exit(0)

		elif pressed_key == demo1.K_POINTS:
			print('-> SHOW FACIAL LANDMARKS')
			self.show_points = (not self.show_points)
			return None

		elif pressed_key == demo1.K_BOUNDING:
			print('-> SHOW BOUNDING CUBE')
			self.show_bounding = (not self.show_bounding)
			return None

		elif pressed_key == demo1.K_MOUTH:
			print('-> SHOW MOUTH OPENNESS ESTIMATION')
			self.show_mar = (not self.show_mar)
			return None

		elif pressed_key == demo1.K_NONE:
			print('-> SHOW NO ESTIMATIONS')
			self.show_points = False
			self.show_bounding = False
			self.show_mar = False
			return None

		elif pressed_key == demo1.K_REFRESH:
			print('-> RESET SHOW TO DEFAULT')
			self.show_points = False
			self.show_bounding = False
			self.show_mar = True
			return None

		elif pressed_key == demo1.K_SAVE_LOG:
			print('-> SAVE LOG FILE WITH PREDICTIONS')
			preds_log = self.talkpredict.fetch_log('preds0')
			ts = int(round(time.time() * 1000))
			path = (demo1.LOG_PATH).replace('%ts', str(ts))
			print("\t" + path)
			preds_log.to_csv(path)
			return None

		elif pressed_key == demo1.K_HELP:
			tk.messagebox.showinfo("Help",
			                       "'p': Show facial landmarks\r\n'b': Show bounding cube\r\n'm': Show mouth info\r\n'n': Show nothing\r\n'r': Refresh/clear the frame of all info\r\n'l': Save log file\r\n'q': Quit the program")
			return None

		else:
			return None

if __name__ == '__main__':
	demo1 = demo1()
	demo1.run()