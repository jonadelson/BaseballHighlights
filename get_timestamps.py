import cv2
import pickle
import numpy as np
import pickle
import sys
import itertools
import pandas as pd
import os
import pymysql
import warnings
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

def mins_secs_to_mills(mins, secs):
    return mins * 60 * 1000 + secs * 1000

def hours_mins_secs_to_mills(hours, mins, secs):
    return hours * 60 * 60 * 1000 + mins * 60 * 1000 + secs * 1000

def get_hours_mins_secs(sv_id):
    secs = sv_id % 100
    whittle = sv_id // 100
    mins = whittle % 100
    whittle = whittle // 100
    return whittle, mins, secs

def mills_to_hours_mins_secs(mills):
    seconds = int(mills / 1000) % 60
    minutes = int((mills / (1000*60)) % 60)
    hours   = int((mills / (1000*60*60)) % 24)
    return hours, minutes, seconds

def scan_cap_ahead(cap, seconds):
    totalSeconds = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, int(totalSeconds) + seconds * 1000)

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])

def is_pitcher_frame(frame, pitcher_clf, pitcher_kmeans):
    
    gen = sliding_window(cv2.resize(frame,(102, 182), 
                                            interpolation = cv2.INTER_CUBIC), 
                                             stepSize=5, 
                                             windowSize=(8,8))
        
    im_kmeans = np.zeros(20)
    for p in gen:
        if p[2].shape == (8,8,3):
            im_kmeans[pitcher_kmeans.predict(p[2].flatten().reshape(1,-1)[0])] += 1
    
    probs = []
    for clf in pitcher_clf:
        prob = clf.predict_proba(im_kmeans.reshape(1,-1))[0][1]
        probs.append(prob)
    return probs

def get_game_info(date_string, conn):
	columns = ["pitches.sv_id", "pitches.des", "pitches.ball", "pitches.strike", 
	"pitches.on_1b", "pitches.on_2b", "pitches.on_3b", "pitches.pitch_id",
	"atbats.outs"]

	query = ("select {0} "
    		"from pitches "
    		"join atbats " 
         	 "on pitches.ab_id = atbats.ab_id "
        	 "where sv_id is not null and atbats.ab_id in "
        	 "(select ab_id from atbats where game_id in "
         	 "(select game_id "
        	 "from games where date = '{1}' and home = 'tba')) "
        	 "order by sv_id").format(",".join(columns), date_string)

	game_info = pd.read_sql(query, conn)
	game_info = game_info.fillna(0)
	return game_info


def is_scoreboard(frame, template, method, clf, kmeans):        
        
    res = cv2.matchTemplate(frame,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    selected = frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    
    gen = sliding_window(selected, stepSize=5, windowSize=(8,8))
    
    im_kmeans = np.zeros(40)
    
    for p in gen:
        if p[2].shape == (8,8,3):
            im_kmeans[kmeans.predict(p[2].flatten().reshape(1,-1)[0])] += 1

    probs = clf.predict_proba(im_kmeans)
    return probs, top_left, bottom_right


def classify_outs(sum):
    if sum < 3500:
        return 0
    elif sum < 4500:
        return 1
    else:
        return 2

def get_count_area(frame, top_left, bottom_right):
    selected = frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    res = cv2.matchTemplate(selected,score_area,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left2 = max_loc
    bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
    #score = selected[top_left2[1]:bottom_right2[1],top_left2[0]:bottom_right2[0]]
    return top_left2, bottom_right2

def select_count_area(frame, top_left, bottom_right, top_left2, bottom_right2):
    selected = frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    count = selected[top_left2[1]:bottom_right2[1],top_left2[0]:bottom_right2[0]]
    return count

def get_counts_transformed(score, pca, pca2, pca3):       
        
    gray = cv2.cvtColor(score[20:30,18:25],cv2.COLOR_BGR2GRAY).flatten()
    transform = pca.transform(gray).reshape(1,-1)[0]

    gray2 = cv2.cvtColor(score[20:30,27:33],cv2.COLOR_BGR2GRAY).flatten()
    transform2 = pca2.transform(gray2).reshape(1,-1)[0]

    transform3 = pca3.transform(gray).reshape(1,-1)[0]
    return transform, transform2, transform3

def get_outs(score):
    sum = cv2.cvtColor(score[21:29,35:46],cv2.COLOR_BGR2GRAY).sum()
    return classify_outs(sum)

def is_base_occupied(sum):
    return 1 if sum > 2000 else 0

def get_base_status(score):
    first = cv2.cvtColor(score[7:12,40:45],cv2.COLOR_BGR2GRAY).sum()
    second = cv2.cvtColor(score[4:9,35:40],cv2.COLOR_BGR2GRAY).sum()
    third = cv2.cvtColor(score[7:12,32:37],cv2.COLOR_BGR2GRAY).sum()
    return tuple(map(is_base_occupied, [first, second, third]))

def get_count(frame, top_left, bottom_right, top_left2, bottom_right2, pca, pca2, pca3,
             neighs, neighs2, two_three):
    
    score = select_count_area(frame, top_left, bottom_right, top_left2, bottom_right2)
        
    transform, transform2, transform3 = (
        get_counts_transformed(score, pca, pca2, pca3))

    balls = neighs.predict(transform)[0]
    strikes = neighs2.predict(transform2)[0]

    if balls in [2,3]:
        balls = two_three.predict(transform3)[0]

    if balls == 4 and strikes == 4:
        balls = 0
        strikes = 0

    return (balls, strikes)

def scan_ahead_info(next_des, strikes, outs, next_outs, current_bases, next_bases, count, next_count):
    # balls
    # next_des = ball, look for next count
    # next_des = strike
        # if strikes < 2, look for next count
        # elif strikes == 2 and outs < 2, look for out
        # elif strikes == 2 and outs == 2, no scoreboard
    # next_des = out
        # if outs < 2, look for next out situation
        # else look for no scorebox
    # still need to figure out balls in play, look at baserunners
    if next_des == 'hbp':
        if count == (0,0):
            if next_bases == current_bases:
                return 'score_change'
            else:
                return 'bases'
        else:
            return 'ball'
    if next_des == 'ball':
        return 'ball'
    elif next_des == 'strike':
        if strikes < 2:
            return 'strike'
        else:
            if outs < 2:
                return 'next_out'
            else:
                return 'final_out'
    elif next_des == 'foul':
        if strikes < 2:
            return 'strike'
        else:
            if next_count == (0,0):
                if next_outs < 3:
                    return 'next_out'
                else:
                    return 'final_out'
            else:
                return 'foul'
    elif next_des == 'out':
        if next_outs < 3:
            return 'next_out'
        else:
            return 'final_out'
    elif next_des == 'play':
        if next_bases == current_bases:
            return 'score_change'
        else:
            return 'bases' 
    else:
        return 'next_pitch'

des_map = {
    'Ball':'ball',
    'Ball In Dirt':'ball',
    'Called Strike':'strike',
    'Foul':'foul',
    'Foul Bunt':'foul',
    'Foul (Runner Going)':'foul',
    'Foul Tip':'foul',
    'Hit By Pitch':'hbp',
    'In play, no out':'play',
    'In play, no out(s)':'play', 
    'In play, out(s)':'out',
    'In play, run(s)':'play',
    'Intent Ball':'ball',
    'Swinging Strike':'strike',
    'Swinging Strike (Blocked)':'strike'
}

template = pickle.load(open('data/rays-scoreboard'))
score_area = pickle.load(open('data/game-info'))
w,h = template.shape[1], template.shape[0]
w2,h2 = score_area.shape[1], score_area.shape[0]
method = 'cv2.TM_CCOEFF'
method = eval(method)
scoreboard_clf = pickle.load(open('models/scoreboard-clf.pkl'))
scoreboard_kmeans = pickle.load(open('models/scoreboard-kmeans.pkl'))
pca = pickle.load(open('models/count1-pca.pkl'))
neighs = pickle.load(open('models/count1-clf.pkl'))
pca2 = pickle.load(open('models/count2-pca.pkl'))
neighs2 = pickle.load(open('models/count2-clf.pkl'))
pca3 = pickle.load(open('models/two-three-pca.pkl'))
two_three = pickle.load(open('models/two-three-clf.pkl'))
pitcher_kmeans = pickle.load(open('models/pitching-view-kmeans.pkl'))
pitcher_clf = pickle.load(open('models/pitcher-clf.pkl'))
pitcher_clfs = [pitcher_clf]

def get_time_stamps(cap, located_scorebox=False, top_left=None, bottom_right=None, top_left2=None, bottom_right2=None,
	tops=[], bottoms=[], score_tops=[], score_bottoms=[], min_boxes=20, found_one=False, fast_forward=True,
	fast_forward_speed=100, show_video=False, starts=[], ends=[], first_timestamp=0, pitching=False, the_outs=0,
	num_missed=0, prev=0, pitch_ids=[], pitches=0, j=0):

	while True:
	    j += 1

	    if pitches == len(counts):
	   		break

	    ret, frame = cap.read()

	    if not ret:
	        break
	    
	    if fast_forward:
	        totalFrameNumber = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(totalFrameNumber) + fast_forward_speed)   

	    if show_video:    
	        cv2.imshow('frame',frame)
	        
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	    if not located_scorebox:
	        
	        probs, top_left, bottom_right = (
	            is_scoreboard(frame, template, method, scoreboard_clf, scoreboard_kmeans))
	        
	        if not found_one:
	            pitcher_probs = is_pitcher_frame(frame, pitcher_clfs, pitcher_kmeans) 
	            if pitcher_probs[0] > 0.5:
	                found_one = True
	                first_timestamp = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	        
	        if probs[0][1] >= 0.5:
	            
	            
	            tops.append(top_left)
	            bottoms.append(bottom_right)
	            top_left2, bottom_right2 = get_count_area(frame, top_left, bottom_right)
	            score_tops.append(top_left2)
	            score_bottoms.append(bottom_right2)
	                
	            if len(tops) == min_boxes:
	                top_left = sorted(tops)[min_boxes/2]
	                bottom_right = sorted(bottoms)[min_boxes/2]
	                top_left2 = sorted(score_tops)[min_boxes/2]
	                bottom_right2 = sorted(score_bottoms)[min_boxes/2]
	                cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, first_timestamp)
	                located_scorebox = True
	                fast_forward_speed = 10
	                prev = first_timestamp
	                started = True
	                print "located scoreboard"
	                sys.stdout.flush()
	        continue  
	    
	    to_look_for = scan_ahead_info(
	                                  des[pitches], 
	                                  counts[pitches][1], 
	                                  the_outs,
	                                  game_outs[pitches],
	                                  bases[pitches],
	                                  bases[pitches + 1],
	                                  counts[pitches],
	                                  counts[pitches+1]
	                                             )
	    
	    
	    if j % 25 == 0 and located_scorebox:
	        probs, _, _ = is_scoreboard(frame, template, method, scoreboard_clf, scoreboard_kmeans)
	        if probs[0][1] >= 0.5:
	            score = select_count_area(frame, top_left, bottom_right, top_left2, bottom_right2)
	            count = get_count(frame, top_left, bottom_right, top_left2, bottom_right2, pca, pca2, pca3,
	             neighs, neighs2, two_three)
	            #print "read status = {}".format((out,base,count))
	            #print "actual status = {}".format((the_outs, bases[pitches], counts[pitches]))
	            #print pitches
	            if count != counts[pitches]:
	                num_missed += 1
	            else:
	                num_missed = 0
	            print "num_missed = {}".format(num_missed)
	            if num_missed == 5:
	             	pitch_ids.append(pitch_id[pitches])

	                pitches += 1
	                start = first_timestamp + (prev - first_timestamp)
	                if pitches == len(counts):
	                    break
	                end = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	                next_start = hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches]))
	                starts.append(start)
	                ends.append(end)
	                diff = next_start - hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches - 1]))
	                prev += diff
	                cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,start + diff)
	                num_missed = 0
	                continue
	            
	            sys.stdout.flush()

	                
	    
	        print "looking for {}".format(des[pitches])
	        sys.stdout.flush()
	       
	    if to_look_for in ['ball', 'strike']:
	        
	        probs, _, _ = is_scoreboard(frame, template, method, scoreboard_clf, scoreboard_kmeans)
	        if probs[0][1] < 0.3:
	            continue
	            
	        count = get_count(frame, top_left, bottom_right, top_left2, bottom_right2,
	                                   pca, pca2, pca3,
	             neighs, neighs2, two_three)

	        if count == counts[pitches+1]:
	            
	       
	            print "{} found".format(des[pitches])
	            sys.stdout.flush()
	            start = first_timestamp + (prev - first_timestamp)
	            pitch_ids.append(pitch_id[pitches])
	            pitches += 1
	            end = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	            starts.append(start)
	            ends.append(end)
	            next_start = hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches]))
	            diff = next_start - hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches - 1]))
	            prev += diff
	            if pitches == len(counts):
	                    break
	            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,start + diff)
	        continue
	    
	    elif to_look_for == 'next_out':
	        probs, _, _ = is_scoreboard(frame, template, method, scoreboard_clf, scoreboard_kmeans)
	        if probs[0][1] < 0.3:
	            continue
	        
	        score = select_count_area(frame, top_left, bottom_right, top_left2, bottom_right2)
	        outs = get_outs(score)
	        
	        if outs == game_outs[pitches]:

	            print "out found"
	            sys.stdout.flush()
	            the_outs = game_outs[pitches]
	            start = first_timestamp + (prev - first_timestamp)
	            pitch_ids.append(pitch_id[pitches])
	            pitches += 1
	            end = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	            starts.append(start)
	            ends.append(end)
	            next_start = hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches]))
	            diff = next_start - hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches - 1]))
	            prev += diff
	            if pitches == len(counts):
	                    break
	            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,start + diff)

	        continue
	     
	    elif to_look_for == 'bases':
	        probs, _, _ = is_scoreboard(frame, template, method, scoreboard_clf, scoreboard_kmeans)
	        if probs[0][1] < 0.3:
	            continue
	        
	        score = select_count_area(frame, top_left, bottom_right, top_left2, bottom_right2)
	        base_status = get_base_status(score)
	        
	        if base_status == bases[pitches+1]:
	            sys.stdout.flush()
	            
	            print "play found"
	            the_outs = get_outs(score)
	            
	            start = first_timestamp + (prev - first_timestamp)
	            pitch_ids.append(pitch_id[pitches])

	            pitches += 1
	            end = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	            '''
	            k = i
	            while end - start < 8000:
	                k -= 1
	                start = firsts[k]
	            '''
	            starts.append(start)
	            ends.append(end)
	            next_start = hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches]))
	         
	            diff = next_start - hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches - 1]))
	            prev += diff
	            if pitches == len(counts):
	                    break
	            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,start + diff)

	        continue    
	        
	    elif to_look_for == 'score_change':
	        probs, _, _ = is_scoreboard(frame, template, method, scoreboard_clf, scoreboard_kmeans)
	        
	        if probs[0][1] < 0.01:
	            end = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	            start = first_timestamp + (prev - first_timestamp)

	            print "score found"
	            sys.stdout.flush()
	            pitch_ids.append(pitch_id[pitches])

	            pitches += 1
	            
	            starts.append(start)
	            ends.append(end)
	    
	            next_start = hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches]))
	           
	            diff = next_start - hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches - 1]))
	            prev += diff
	            if pitches == len(counts):
	                    break
	            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,start + diff)

	        else:
	            continue    
	    
	    elif to_look_for == 'final_out':        
	        probs, _, _ = is_scoreboard(frame, template, method, scoreboard_clf, scoreboard_kmeans)
	        if probs[0][1] < 0.01:
	            sys.stdout.flush()
	            start = first_timestamp + (prev - first_timestamp)
	            end = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)

	            starts.append(start)
	            ends.append(end)
	            pitch_ids.append(pitch_id[pitches])

	            pitches += 1
	            the_outs = 0
	        
	            next_start = hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches]))

	            diff = next_start - hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches - 1]))
	            prev += diff
	            if pitches == len(counts):
	                    break
	            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,start + diff)
	    
	    else:
	        pitcher_probs = is_pitcher_frame(frame, pitcher_clfs, pitcher_kmeans)
	        if not pitching:
	            if pitcher_probs[0] > 0.2:
	                pitching = True
	            continue
	        if pitcher_probs[0] < 0.01:
	            start = first_timestamp + (prev - first_timestamp)
	            end = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
	            starts.append(start)
	            ends.append(end)
	            pitch_ids.append(pitch_id[pitches])

	            pitches += 1
	            next_start = hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches]))
	            diff = next_start - hours_mins_secs_to_mills(*get_hours_mins_secs(sv_id[pitches - 1]))
	            prev += diff
	            if pitches == len(counts):
	                    break
	            cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC,start + diff)
	            pitching = False
	return starts, ends, pitch_ids

if __name__ == '__main__':
	conn = pymysql.connect(host='localhost', port=3306, user='root', 
                       passwd='password', db='pitchFX2')
	
	video_string = sys.argv[1]
	start_mins = int(sys.argv[2])
	start_secs = int(sys.argv[3])
	date_string = sys.argv[4]
	out_file = sys.argv[5]
	show_video = int(sys.argv[6])

	game_info = get_game_info(date_string, conn)

	game_balls = [int(x) for x in game_info['ball']]
	game_strikes = [int(x) for x in game_info['strike']]
	game_outs = [int(x) for x in game_info.outs]
	on_1b = [1 if x > 0 else 0 for x in game_info.on_1b]
	on_2b = [1 if x > 0 else 0 for x in game_info.on_2b]
	on_3b = [1 if x > 0 else 0 for x in game_info.on_3b]
	counts = zip(game_balls, game_strikes)
	bases = zip(on_1b, on_2b, on_3b)
	sv_id = [int(x.split('_')[1]) for x in game_info.sv_id]
	pitch_id = [int(x) for x in game_info.pitch_id]
	des = [des_map[d] if d in des_map else 'unknown' for d in game_info.des]

	cap = cv2.VideoCapture(video_string)
	start_time = mins_secs_to_mills(start_mins,start_secs)
	cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_time)
	starts, ends, pitch_ids = get_time_stamps(cap=cap, show_video=show_video)

	pickle.dump([starts, ends, pitch_ids], open(out_file, 'wb'))
