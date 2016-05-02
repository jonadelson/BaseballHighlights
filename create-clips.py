import subprocess
import cv2
import pickle
import sys
from get_timestamps import mills_to_hours_mins_secs, scan_cap_ahead

if __name__ == '__main__':
	video_file = sys.argv[1]
	ts_file = sys.argv[2]
	out_folder = sys.argv[3]
	cap = cv2.VideoCapture(video_file)

	starts, ends, ids = pickle.load(open(ts_file))

	for i,s,e in zip(ids, starts, ends):
		print i
		diff = e - s
		hours, mins, secs = mills_to_hours_mins_secs(diff)
		if (mins*60 + secs) > 30:
			e = s + 30*1000
			diff = e - s
			hours, mins, secs = mills_to_hours_mins_secs(diff)

		if hours < 10:
			hours = '0' + str(hours)
		if mins < 10:
			mins = '0' + str(mins)
		if secs < 10:
			secs = '0' + str(secs)

		start_hours, start_mins, start_secs = mills_to_hours_mins_secs(s)
		if start_hours < 10:
			start_hours = '0' + str(start_hours)
		if start_mins < 10:
			start_mins = '0' + str(start_mins)
		if start_secs < 10:
			start_secs = '0' + str(start_secs)
		subprocess.call(['ffmpeg','-ss','{0}:{1}:{2}'.format(start_hours, start_mins, start_secs), '-i', 
			video_file, '-t', '{0}:{1}:{2}'.format(hours, mins, secs), '-c', 'copy', out_folder + '/' + str(i) + '.mp4'])
		        

