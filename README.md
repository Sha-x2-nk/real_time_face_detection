# real_time_face_detection
The project was facial recognition. The approach we follow is:
## TRAINING:
1. place all the folders in data/train/ directory as: <br />
&emsp;      data/<br />
         &emsp;&emsp;train/<br />
          &emsp;&emsp;&emsp;-shashank/<br />
                &emsp;&emsp;&emsp;&emsp;  IMG_name_1.format<br />
               &emsp;&emsp;&emsp;&emsp;&emsp;    .<br />
                &emsp;&emsp;&emsp;&emsp;&emsp;    .<br />
                &emsp;&emsp;&emsp;&emsp;   IMG_name_n.fomrat<br />
               &emsp;&emsp;&emsp;&emsp;.<br />
              &emsp;&emsp;&emsp;&emsp; .<br />
               &emsp;&emsp;&emsp;varun/<br />
                  &emsp;&emsp;&emsp;&emsp;IMG_name_1.format<br />
                   &emsp;&emsp;&emsp;&emsp;&emsp; .<br />
                 &emsp;&emsp;&emsp;&emsp;&emsp;   .<br />
                &emsp;&emsp;&emsp;&emsp;.<br />
              &emsp;&emsp;&emsp;&emsp;  .<br /></p>
2. compile and run train.cpp. The file has support for all major formats. Since opencv does not have HIEC support(upto v4.7) libheif has to be compiled seperately, the library has been used in the .cpp file.
3. Embeddings for each face are egenrated.
4. the embeddings are appended in FAISS HNSW Index. FAISS is a very efficient similarity search librrary developed by google.
5. the ids of each embedding is stored in db against the user name(which happens to be the folder name in our case (eg. shashank, varun).
            
##  LIVE CAMERA FEED.
1. 480x640 resolution camera feed is inputted.
2. face detection
3. largest face is detected.(user can decide to omit it.)
4. embedding generation
5. FAISS Index. FAISS is a efficient similarity search library. user should read about it. The index we use is HNSW Index.
6. the nearest neighbours' embedding ids are feteched from the db. 
7. the similarity criteria we use right now is cosine similarity. threshold is kept at 0.75 and k=5 nearest neighbours are searched for.
8. If all the nearest neighbours are embeddings of the same person then a label is given and the person is labeled as identified. Our use case reuqires such high precision but it can be customised.

## compiling:
if you have all the libraries installed and configured our way, the program can be compiled as<br />
 <code>g+ live.cpp -o3 -o live -lfaiss -lpqxx -lpq `pkg-config --cflags --libs opencv4`"</code><br />
 Please remember to source Intel OneAPI environmental varibales for your program to use them.
 

## DETAILS:
1. the detector used while training is: YOLOV8 face detector. The model was best out of what we saw for accuracy and its runtime was less than MTCNN.
2. the detector used for live video inference is UltraLightFaceDtector which is unmatched in its speed.
3. embedding generator model used is FsceNet.
4. The ulra light model 640rf supports 640x480 feed. If you plan to pass videos of varies sizes, please use 320RF_withoutpreprocessing model. If a lot of overlapping boxes are appearing, you can tune the nms threshold to remove them.

## Requirements
1. OpenCV compiled from source.(We also used optimised libraries like intel MKL, TBB) you can follow the guide here https://medium.com/@shashankrajora2002/unlocking-powerful-performance-harnessing-cpu-capabilities-for-significant-improvements-without-4ad802fb7798
2. FAISS compiled from source. one you've built opencv you should have an idea what to do.
3. any db connector(you can also use file system for storing mappings between ids and labels), we use POSTGRESQL and PQXX connector.
4. Libheif.(if you need support for HEIC format. otherwise you can comment decode_heic section in train.cpp)

## Results:
Platform -> Intel i5 1135g7 
| Detection     | embedding genration |  FAISS Index Search  |
| ------------- | ------------------- | -------------------- |
|   >120 FPS    |       >110 FPS      |       >990 FPS       |

All these results have been benched without GPU. Libraries like Intel MKL and Intel OpenVINO are strongly recommended. You can install them via the opencv guide I provided above.
the live.cpp code can be modified to run inference over photos and videos as well if the user has an idea of what is going on in the code. 
there is a keyword when cv::dnn::Net is initialised - Net.setpreferableBackend. We have set it to INFERENCE_ENGINE_BACKEND to use Intel OpenVINO. Should be set to CUDA if available.
Somehow opencv has this bug where it cannot use onnx models with variable batch size, so we have 2 files, one which accepts batch_size=1 inputs and one which uses batch_size = 2 inputs.
The embedding generator is taken from facenet_pytorch repo by timesler. You can customise your own onnx model while OpenCV guys fix this bug and hardcode the batch_size.

## CITATIONS:
https://github.com/timesler/facenet-pytorch
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
https://github.com/facebookresearch/faiss
https://github.com/jtv/libpqxx
https://github.com/strukturag/libheif
