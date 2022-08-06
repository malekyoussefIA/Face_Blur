
import argparse
from face.utils import get_list_img,set_environment
from face.utils import isfile,isdir,remove,join,abspath,makedirs
from face.utils import setproctitle,NAME_EXTRA_FILES_DIR

setproctitle.setproctitle('[FaceDetection]')

def parse_args():
    parser=argparse.ArgumentParser('FaceBlurring')
    parser.add_argument('--input','-i',\
                         type=str,default='./data/imgs/test_group.jpg',\
                         help='Input path (either a folder or a single image')
    parser.add_argument('--gpu','-g',\
                        type=bool,default=True,help='Set to True for GPU mode.CPU otherwise.')
    parser.add_argument('--thresh_confidence','-thresh',\
                        type=float,default=0.1,\
                        help='Set a confidence threshold for the detection algorithm.')
    parser.add_argument('--verbose','-v',\
                        type=int,default=1,\
                        help='Set the verbose parameter : 0 : No infos shown at all.\
                                                          1 : Save bounding boxes + dictionaries in .json file')

    return parser.parse_args()



def main(args):
    import face.face_detection as fd
    #Create a folder which is going to contain some important files : weights for RetinaFace etc. 
    extra_dir=join(abspath('.'),NAME_EXTRA_FILES_DIR)
    makedirs(extra_dir,exist_ok=True)

    #Select the best available GPU if args.gpu is not None.
    set_environment(use_gpu=args.gpu)
    
    #Get a list of image path.
    les_img=get_list_img(args.input)
    fd.run(les_img,args)

if __name__=='__main__':
    args=parse_args()
    main(args)
    

        





