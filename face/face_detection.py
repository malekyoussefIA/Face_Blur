
from numpy import empty
from face.utils import *
import torch
from retinaface.predict_single import Model

class FaceDetector():
    def __init__(self,thres_confidence,gpu):
        
        '''Confidence threshold for the face detection'''
        self.thresh_confidence=thres_confidence
        
        '''Build a dict. which will contain scores and faces for a given images.'''
        self.faceDict={}

        '''Set the Face detector from RetinaFace'''
        detector_retina= Model(max_size=2048,device=torch.device('cuda' if gpu is not False else 'cpu'))    #Build the model.
        detector_retina.load_state_dict(FaceDetector.get_weights())  #Load pre-trained weights. 
        detector_retina.eval()  #Initiate the face detector. 

        self.detector=detector_retina


    def get_weights():
        """
        To download on S3 the weigth file associated to the RetinaFace algorithm if the file not already exists.
        Returns:
            [torch]: The loaded weight file. 
        """
        path_weight=join(abspath('.'),'face/weights','Resnet50_Final.pth') 
        #Load the weight file 
        pretrained_dict=torch.load(path_weight)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        return pretrained_dict

    def get_facesDict(self,**kwargs):
        """
        Get the complete dictionnary associated to the raw face coordiantes.

        Returns:
            [dict]: Either the dictionnary associated to a specific image name if a key argument is provided or 
                    the whole dictionnary otherwise. 
        """ 
        key = kwargs.get('key', None)
        return self.faceDict[key] if key is not None else self.faceDict

    def isFaceDetected(self):
        """
        Function to test whether or not there is at least one face. 

        Returns:
            [bool]: set to True if there is at least a single face, False otherwise. 
        """
        return self.isFace
    
    def eval(self,img_name,img,**kwargs):
        """
        Main function used to perform the face detection. 
        Both raw face coordinates and confidence scores are saved in a dictionnary 
        as well as the number of face. Optionnaly returns the built dicitonnary

        Args:
            img_name ([str]): Input image path.
            img ([RGB image]): RGB image already opened with OpenCV.

        Returns:
            [dict]: Return the complete face dictionnary if the flag return_dict is set to True. 
        """
        return_dict = kwargs.get('return_dict', False)

        self.faceDict[f'{img_name}']={'Height':img.shape[0],\
                                      'Width':img.shape[1]}
        self.img=img

        #Perform the prediction.
        self.res=self.detector.predict_jsons(img,self.thresh_confidence) 

        self.isFace=False if not self.res else True #Boolean set to True if faces were detected in img, False otherwise
            
        counter_nb_faces=0
        #Main loop to record both face coordinates and associated scores.
        for idx,ann in enumerate(self.res):
            if len(ann['bbox'])!=0:
                x_min, y_min, x_max, y_max=ann['bbox']
                x_min = int(np.clip(x_min, 0, x_max - 1))
                y_min = int(np.clip(y_min, 0, y_max - 1))
                score=ann['score']
                
                self.faceDict[img_name][f'face_{idx}']={}
                self.faceDict[img_name][f'face_{idx}']['raw_coord']=[x_min,y_min,x_max,y_max]
                self.faceDict[img_name][f'face_{idx}']['score']=score
                counter_nb_faces+=1

        self.faceDict[img_name]['nb_faces']=counter_nb_faces+1 if self.isFace else 0

        if return_dict:
            return self.get_facesDict(key=img_name)
        
    def draw_face_pos(self,img_name,img,out_dir):
        """
        Draw 2D bouding boxes over each detected faces. 
        Args:
            img_name ([str]): Input image path.
            img ([RGB image]): RGB image already opened with OpenCV.
            out_dir ([str]): Root directory path where to save results. 
        """
        nb_faces=self.faceDict[img_name]['nb_faces']
        les_faces_coord=[self.faceDict[img_name][f'face_{i}']['raw_coord'] for i in range(nb_faces-1)]

        img_copy=img.copy()
        out_path_img=join(out_dir,img_name)

        for face in les_faces_coord:        #face is a 4-length list with [x_min,y_min,x_max,y_max]
            face = list(map(int, face))
            x_min, y_min, x_max, y_max=face
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imwrite(out_path_img,cv2.cvtColor(img_copy,cv2.COLOR_RGB2BGR))

def run(les_img,args):
    """
    The function perform the main face detection over all the image contained in the les_img list. 

    Args:
        args ([args dict]): Arguments passed through the main.py file. 
        les_img ([list]): Contain the root path list of all the image which are going to be processed by the face detection algorithm. 
    """
    #Init the face detection algorithm.
    detector=FaceDetector(args.thresh_confidence,args.gpu)

    #Run over all available paths within les_img
    for img_path in les_img:

        #Image basename
        basename_img=basename(img_path)

        try: 
            #Read input image and convert it into RGB channels.
            img=cv2.imread(img_path)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f'[CV2 error: {e} \n Issue with the image name: {basename_img} - NO PROCESSING MADE]')
            continue

        #Run inference. 
        faces_res=detector.eval(basename_img,img,return_dict=True)
        print(f"[Image {basename_img} has been processed: {faces_res['nb_faces']} face(s) detected.]") 

    if args.verbose:
        path_out_dir=join(abspath('.'),'results')
        path_out_dir_2DBB=join(path_out_dir,NAME_PLOT_FACE_DETECTION_DIR)
        makedirs(path_out_dir_2DBB,exist_ok=True)   #Create a result folder if it not already exists.

        #Save image with 2D bouding boxes if verbose
        detector.draw_face_pos(basename_img,img,path_out_dir_2DBB)

        #Save in a json file the detected faces coordinates.
        date=datetime.date.today().strftime('%d-%m-%Y')
        
        #Save the generated dict. file in a .json
        with open(join(path_out_dir,f'res_RetinaFace_{date}.json'), 'w') as fp:
            json.dump(detector.get_facesDict(), fp,indent=2)   
   


    

    
   

