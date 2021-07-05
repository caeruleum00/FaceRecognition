import mxnet as mx
import cv2
import sys
import os
import numpy as np
import align_tools
import face_model
import detect
from retinaface_cov import RetinaFaceCoV
from tensorflow.python.keras.models import load_model
import timeit
from PIL import ImageFont, ImageDraw, Image
import datetime

fontpath_s = "fonts/AppleSDGothicNeoR.ttf"
fontpath_l = "fonts/AppleSDGothicNeoH.ttf"

# age, gender
def draw_face(img, bbox, gender, ages):
    font_size = int((bbox[2] - bbox[0])/6)    
    font_ko_s = ImageFont.truetype(fontpath_s, font_size)
    font_ko_l = ImageFont.truetype(fontpath_l, font_size)
    frame = Image.fromarray(img)
    draw = ImageDraw.Draw(frame)
    text = str(int(ages)) + '세 ' + str(gender)               
    draw.text((bbox[0] + 5, bbox[3] + 10 ), text , font = font_ko_l, fill=(0, 0, 0))
    draw.text((bbox[0] + 5, bbox[3] + 10 ), text , font = font_ko_s, fill=(255, 255, 255)) 
    frame = np.array(frame)
    return frame     

def main(args):
    align_t=align_tools.align_tools()
    # cap=cv2.VideoCapture('video/test.webm')
    cap=cv2.VideoCapture(0)
    count = 1
    cap.set(3, 1080) #WIDTH
    cap.set(4, 720) #HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX
    age = [0]
    g = None
    mask_on = None
    CI_r = cv2.imread('fonts/piai_CI.png')
    CI = cv2.resize(CI_r, (0, 0), fx=0.3, fy=0.3)
    CI_w, CI_h, _ = CI.shape
    cnt = 0
    font_ko_o = ImageFont.truetype(fontpath_l, 30)

    while cap.isOpened():
        scales = [640, 1080]
        ret,frame=cap.read()

        ## PIAI 로고삽입 
        cv2.rectangle(frame, (0, frame.shape[0]- CI_w), (frame.shape[1] , frame.shape[0]), (255,255,255), -1)  
        frame[frame.shape[0] - CI_w : frame.shape[0], frame.shape[1] - CI_h : frame.shape[1]] = CI

        start_t = timeit.default_timer()
        if ret is True:
            faces, person_num = detect.detect_person(frame, scales, thresh, detector)
            if faces is not None:
                mask_F=0
                box_list = []
                for i in range(person_num):
                    box, color, text, mask_on = detect.detect_mask(faces, i, mask_thresh)
                    # print(list(box))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

                    box_sizee = (box[2]-box[0]) * (box[3]-box[1])
                    p_size = (box_sizee / (scales[0]*scales[1]))*100

                    if mask_on == False:
                        box_list.append(box)
                        mask_F = 1
                    
                    # 마스크 착용, 미착용 문구
                    font_size = int((box[2] - box[0])/6)
                    font_ko_s = ImageFont.truetype(fontpath_s, font_size)
                    frame = Image.fromarray(frame)
                    draw = ImageDraw.Draw(frame)
                    draw.text((box[0]+font_size/2 , box[1] + font_size/4 ),'%s'%text, font = font_ko_s, fill=color)
                    frame = np.array(frame)
                
                if (mask_F == 1) and (p_size>0.4):
                    bboxes = model_ga.get_box(frame)
                    if (count % 30 == 0) or (count == 1):
                        bboxes = []
                        genderes = []
                        ages = []
                        faces, bboxes = model_ga.get_faces(frame)
                        for i in range(len(faces)):  
                            gender, age = model_ga.get_ga(faces[i])
                            if int(age) < 21:
                                age = 21

                            elif int(age) > 60:
                                age = 60
                            genderes.append(gender)
                            ages.append(age)

                    if len(bboxes) == len(genderes):
                        for i in range(len(bboxes)):
                            bbox = bboxes[i]
                            if genderes[i] == 0 :
                                gender = '여성'
                            else : gender = '남성'
                            frame = draw_face(frame, bbox, gender, ages[i])

                    for box in box_list:
                        detect.detect_emotion(frame, box, classifier, font)
        
            count += 1
            terminate_t = timeit.default_timer()
            FPS = int(1./(terminate_t - start_t))
            print('FPS : ', FPS)

            now = datetime.datetime.now().strftime("%d_%H-%M-%S")
            keycode = cv2.waitKey(1)  # 
            if keycode == ord('q') or keycode == ord('Q'):
                break

            elif keycode ==ord('s') and cnt ==0 :
                cv2.rectangle(frame, (0, 0), (1920, 1080), (255, 255, 255), -1)
                cv2.imshow('frame', frame)
                cnt+=1

            elif keycode ==ord('s') or (cnt >0 and cnt <15):
                cv2.imwrite('screenshot/' + str(now) + ".png", frame)
                text= 'screenshot saved'
                frame = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame)
                draw.text((520, 660),  text, font=font_ko_o, fill=(0,0,0))
                frame = np.array(frame)
                print('='*30,'cnt=',cnt,'='*30)
                cv2.imshow('frame', frame)
                cnt+=1

            elif cnt >=15:
                cnt = 0
                cv2.imshow('frame', frame)
            else:
                cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = detect.get_args()
    thresh = 0.9
    mask_thresh = 0.2
    gpuid = 0
    model = face_model.FaceModel(args)
    detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')
    model_ga = detect.FaceAgeGenderModel(args)
    classifier = load_model('./model/emotion_classification_vgg_5_emotions.h5')
    main(args)
