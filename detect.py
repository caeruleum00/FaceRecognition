import cv2
import numpy as np
from retinaface_cov import RetinaFaceCoV
import argparse
from tensorflow.python.keras.preprocessing.image import img_to_array  # -> mxnet 기반으로 못 바꾸나...?
import mxnet as mx
from PIL import ImageFont, ImageDraw, Image
from utils import face_preprocess
from utils.mtcnn_detector import MtcnnDetector

def get_args():  
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    # parser.add_argument('--model',
    #                     default='model/model,0',
    #                     help='path to load model.')
    parser.add_argument(
        '--det',
        default=0,
        type=int,
        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--model',        default='model/model,200',   help='path to load model.')
    parser.add_argument('--mtcnn_model',  default='mtcnn-model',       help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    args = parser.parse_args()
    return args


def detect_person(frame, scales, thresh, detector) :
    im_shape = frame.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    faces, _ = detector.detect(frame, thresh, scales=scales, do_flip=flip)

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        #print ("faces", faces)
    return faces, faces.shape[0]

def detect_mask(faces, i, mask_thresh) :
    box, color, text, mask_on = [0, 0, 0, 0], (0, 0, 0), ' ', 'True'
    face = faces[i]
    box = face[0:4].astype(np.int)
    mask = face[5]
    
    if mask >= mask_thresh:
        color = (0, 255, 0)
        #text = 'with_mask'
        text = '마스크 착용'
        mask_on = True
    else:
        color = (0, 0, 255)
        #text = 'without_mask'
        text = '마스크 미착용'
        mask_on = False

    return box, color, text, mask_on

def detect_age_gender(frame, box, model_age, model_gender, align_t):
    nimg, _ =align_t.get_intput_cv(frame)
    if nimg is None:
        return [0], None
    nimg = nimg[:, :, ::-1]
    nimg= np.transpose(nimg,(2,0,1))

    input_blob = np.expand_dims(nimg, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]])))
    model_age.forward(db, is_train=False)
    age = model_age.get_outputs()[0].asnumpy()

    model_gender.forward(db,is_train=False)
    gender=model_gender.get_outputs()[0].asnumpy()
    print ('gender model path',model_gender.get_outputs()[0].asnumpy() )

    g='female'
    if gender[0]>0.5:
        g='male'

    return age, g


def detect_age_gender_ko(frame, box, model_age, model_gender, align_t):
    nimg, _ =align_t.get_intput_cv(frame)
    if nimg is None:
        return [0], None
    nimg = nimg[:, :, ::-1]
    nimg= np.transpose(nimg,(2,0,1))

    input_blob = np.expand_dims(nimg, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]])))
    model_age.forward(db, is_train=False)
    age = model_age.get_outputs()[0].asnumpy()

    model_gender.forward(db,is_train=False)
    gender=model_gender.get_outputs()[0].asnumpy()

    g='여성'
    if gender[0]>0.6:
        g='남성'
    
    return age, g


def detect_age_gender_people(frame, box, model_age, model_gender, align_t, font):
    align_t.get_intput_cv_people(frame, model_age, model_gender, box, font)


def detect_emotion(frame, box, classifier, font):
    class_labels = ['화남', '웃음', '무표정', '슬픔', '놀람']
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[box[1]:box[3], box[0]:box[2]]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

    roi = roi_gray.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # make a prediction on the ROI, then lookup the class
    predicts = classifier.predict(roi)[0]
    label = class_labels[predicts.argmax()]

    # Emoji Add
    w = box[2] - box[0]
    d = int((box[2] - box[0])/2)
    d_hf = int(d/2)
    x = box[0] + d_hf
    y = box[1] - d
    frame_w, frame_h = frame.shape[:2]
    emojis_img = cv2.imread('./emojis/%d.png'%predicts.argmax())
    # create masks to be used later
    img_gray=cv2.cvtColor(emojis_img,cv2.COLOR_BGR2GRAY)
    ret,original_mask=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV)
    original_mask_inv=cv2.bitwise_not(original_mask)
    
    if x+d> frame_h or y+d > frame_w: # out of frame
        return
    if x < 0 or y < 0: # out of frame
        return
    
    newFace = cv2.resize(emojis_img,(d,d),cv2.INTER_AREA)
    mask = cv2.resize(original_mask,(d,d),cv2.INTER_AREA)
    mask_inv = cv2.resize(original_mask_inv,(d,d),cv2.INTER_AREA)
    roi=frame[y:y+d,x:x+d]
    frame_bg=cv2.bitwise_and(roi,roi,mask=mask)
    img_fg=cv2.bitwise_and(newFace,newFace,mask=mask_inv)

    # replace the face with the image data and draw a rectangle
    frame[y:y+d,x:x+d]= frame_bg + img_fg
    return label
    

class FaceAgeGenderModel:
    def __init__(self, args):
        self.args = args
        if args.gpu >= 0:
            ctx = mx.gpu(args.gpu)
        else:
            ctx = mx.cpu()
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        if len(args.model) > 0:
            self.model = self.get_model(ctx, image_size, args.model, 'fc1')

        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        self.image_size = image_size
        detector = MtcnnDetector(model_folder=args.mtcnn_model, ctx=ctx, num_worker=1, accurate_landmark=True,
                                 threshold=self.det_threshold)
        self.detector = detector

    # model_loading
    def get_model(self, ctx, image_size, model_str, layer):
        _vec = model_str.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model

    def get_model_gender(ctx, image_size, model_str, layer):
        _vec = model_gender_str.split(',')
        assert len(_vec)==2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading',prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[layer+'_output']
        model = mx.mod.Module(symbol=sym,data_names=('data','stage_num0','stage_num1','stage_num2'),context=ctx, label_names = None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1])),('stage_num0',(1,3)),('stage_num1',(1,3)),('stage_num2',(1,3))])
        model.set_params(arg_params, aux_params)
        return model


    # face recognition
    def get_faces(self, face_img):
        ret = self.detector.detect_face(face_img)
        if ret is None:
            return [], []
        bbox, points = ret
        if bbox.shape[0] == 0:
            return [], []
        bboxes = []
        pointses = []
        faces = []
        for i in range(len(bbox)):
            b = bbox[i, 0:4]
            bboxes.append(b)
            p = points[i, :].reshape((2, 5)).T
            pointses.append(p)
            nimg = face_preprocess.preprocess(face_img, b, p, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2, 0, 1))
            input_blob = np.expand_dims(aligned, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            faces.append(db)
        return faces, bboxes

    # age, gender compute
    def get_ga(self, data):
        self.model.forward(data, is_train=False)
        ret = self.model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age