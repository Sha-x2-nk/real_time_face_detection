// opencv libraries
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/highgui.hpp>
// FAISS Index
#include<faiss/IndexHNSW.h>
// FAISS for reading-writing to disk
#include<faiss/index_io.h>
// fstream for checking if files exist
#include<fstream>
// postgresql 
#include<pqxx/pqxx>
// file directory functions
#include<dirent.h>
// libheif for reading HEIC images
#include<libheif/heif.h>
// standard libraries
#include<vector>
#include<iostream>

//we will use the YOLOv8 detector for training. Since training is a crucial process, we can afford to invest time to this
class YOLOv8_face
{
public:
	YOLOv8_face(std::string modelpath, float confThreshold, float nmsThreshold);
	void detect(cv::Mat& frame, std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector< std::vector<cv::Point>> &landmarks, bool return_largest);
private:
	cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw);
	const bool keep_ratio = true;
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	const int num_class = 1; 
	const int reg_max = 16;
	cv::dnn::Net net;
	void softmax_(const float* x, float* y, int length);
	void generate_proposal(cv::Mat out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector< std::vector<cv::Point>>& landmarks, int imgh, int imgw, float ratioh, float ratiow, int padh, int padw);
	void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<cv::Point> landmark);
};

static inline float sigmoid_x(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

YOLOv8_face::YOLOv8_face(std::string modelpath, float confThreshold, float nmsThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	this->net = cv::dnn::readNet(modelpath);
	// net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
	// net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

cv::Mat YOLOv8_face::resize_image(cv::Mat srcimg, int *newh, int *neww, int *padh, int *padw)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *padw, this->inpWidth - *neww - *padw, cv::BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padh = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *padh, this->inpHeight - *newh - *padh, 0, 0, cv::BORDER_CONSTANT, 0);
		}
	}
	else {
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

void YOLOv8_face::softmax_(const float* x, float* y, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		y[i] = exp(x[i]);
		sum += y[i];
	}
	for (i = 0; i < length; i++)
	{
		y[i] /= sum;
	}
}

void YOLOv8_face::generate_proposal(cv::Mat out, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector< std::vector<cv::Point>>& landmarks, int imgh,int imgw, float ratioh, float ratiow, int padh, int padw)
{
	const int feat_h = out.size[2];
	const int feat_w = out.size[3];
	// std::cout << out.size[1] << "," << out.size[2] << "," << out.size[3] << std::endl;
	const int stride = (int)ceil((float)inpHeight / feat_h);
	const int area = feat_h * feat_w;
	float* ptr = (float*)out.data;
	float* ptr_cls = ptr + area * reg_max * 4;
	float* ptr_kp = ptr + area * (reg_max * 4 + num_class);

	for (int i = 0; i < feat_h; i++)
	{
		for (int j = 0; j < feat_w; j++)
		{
			const int index = i * feat_w + j;
			int cls_id = -1;
			float max_conf = -10000;
			for (int k = 0; k < num_class; k++)
			{
				float conf = ptr_cls[k*area + index];
				if (conf > max_conf)
				{
					max_conf = conf;
					cls_id = k;
				}
			}
			float box_prob = sigmoid_x(max_conf);
			if (box_prob > this->confThreshold)
			{
				float pred_ltrb[4];
				float* dfl_value = new float[reg_max];
				float* dfl_softmax = new float[reg_max];
				for (int k = 0; k < 4; k++)
				{
					for (int n = 0; n < reg_max; n++)
					{
						dfl_value[n] = ptr[(k*reg_max + n)*area + index];
					}
					softmax_(dfl_value, dfl_softmax, reg_max);

					float dis = 0.f;
					for (int n = 0; n < reg_max; n++)
					{
						dis += n * dfl_softmax[n];
					}

					pred_ltrb[k] = dis * stride;
				}
				float cx = (j + 0.5f)*stride;
				float cy = (i + 0.5f)*stride;
				float xmin = std::max((cx - pred_ltrb[0] - padw)*ratiow, 0.f);  ///restore to the original image
				float ymin = std::max((cy - pred_ltrb[1] - padh)*ratioh, 0.f);
				float xmax = std::min((cx + pred_ltrb[2] - padw)*ratiow, float(imgw - 1));
				float ymax = std::min((cy + pred_ltrb[3] - padh)*ratioh, float(imgh - 1));
				cv::Rect box = cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin));
				boxes.push_back(box);
				confidences.push_back(box_prob);

				// the below code is for landmarks and has been commented out
				// vector<Point> kpts(5);
				// for (int k = 0; k < 5; k++)
				// {
				// 	float x = ((ptr_kp[(k * 3)*area + index] * 2 + j)*stride - padw)*ratiow;  ///restore to the original image

				// 	float y = ((ptr_kp[(k * 3 + 1)*area + index] * 2 + i)*stride - padh)*ratioh;
				// 	///float pt_conf = sigmoid_x(ptr_kp[(k * 3 + 2)*area + index]);
				// 	kpts[k] = Point(int(x), int(y));
				// }
				// landmarks.push_back(kpts);
			}
		}
	}
}

void YOLOv8_face::detect(cv::Mat& srcimg, std::vector<cv::Rect> &boxes, std::vector<float> &confidences, std::vector< std::vector<cv::Point>> &landmarks, bool return_largest = false)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	cv::Mat dst = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
	cv::Mat blob;
	cv::dnn::blobFromImage(dst, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	std::vector<cv::Mat> outs;

	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;

	generate_proposal(outs[0], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	generate_proposal(outs[1], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);
	generate_proposal(outs[2], boxes, confidences, landmarks, srcimg.rows, srcimg.cols, ratioh, ratiow, padh, padw);

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
    std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	
	//finding the largest face
	if(boxes.size() > 1 && return_largest){
		int largest_area = 0;
		int largest_idx = 0;
		for(int i=0;i<indices.size();i++)
		{
			int idx = indices[i];
			cv::Rect box = boxes[idx];
			if(box.width*box.height>largest_area)
			{
				largest_area = box.width*box.height;
				largest_idx = idx;
			}
		}
		indices.clear();
		indices.push_back(largest_idx);
        auto largest_box = boxes[largest_idx];
        auto largest_conf = confidences[largest_idx];
        // auto largest_landmark = landmarks[largest_idx];
        boxes.clear();
        confidences.clear();
        // landmarks.clear();
        boxes.push_back(largest_box);
        confidences.push_back(largest_conf);
        // landmarks.push_back(largest_landmark);
	}
}

inline bool is_heic(std::string img_path){
        return (img_path.substr(img_path.size()-5, 5) == ".heic" || img_path.substr(img_path.size()-5, 5) == ".HEIC");
}

cv::Mat decode_heic(std::string filePath){
    heif_context* context = heif_context_alloc();
    heif_context_read_from_file(context, filePath.c_str(), nullptr);

    // Retrieve the primary image (usually the first image in the HEIC file)
    heif_image_handle* imageHandle = nullptr;
    heif_context_get_primary_image_handle(context, &imageHandle);
    // Decode the HEIC image
    heif_image* image = nullptr;
    heif_decode_image(imageHandle, &image, heif_colorspace_RGB, heif_chroma_interleaved_RGB, nullptr);

    // Get image information
    int width = heif_image_get_width(image, heif_channel_interleaved);
    int height = heif_image_get_height(image, heif_channel_interleaved);

    // Create OpenCV Mat without data
	cv::Mat mat(height, width, CV_8UC3);
    // Get the image data and copy it to the OpenCV Mat
    uint8_t* imageData = const_cast<uint8_t*>(heif_image_get_plane_readonly(image, heif_channel_interleaved, 0));
    std::memcpy(mat.data, imageData, width * height * 3);

    // Release libheif resources
    heif_image_release(image);
    heif_image_handle_release(imageHandle);
    heif_context_free(context);

    // Convert from RGB to BGR
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    return mat;
}

faiss::IndexHNSWFlat *generateIndex(int m=128, int efconstruction=64, int efsearch=250){
	faiss::IndexHNSWFlat* index = new faiss::IndexHNSWFlat(512, m);
	index->hnsw.efConstruction = efconstruction;
	index->hnsw.efSearch = efsearch;
	return index;
}

int main(int argc, char *args[]){
    std::string train_dir = "./data/train";
    std::string embed_gen1_path = "./models/inceptionResnetV1.onnx";
    std::string embed_gen20_path = "./models/inceptionResnetV1_20.onnx";
    std::string detector_model_path = "./models/yolov8n-face.onnx";
    std::string faiss_index_path = "./models/faissHNSW.index";
 
    // establishing connection with db
    pqxx::connection conn = pqxx::connection("dbname=IIIT_EEMS user=postgres password=282606 host = localhost, port = 5432");
    if(!conn.is_open()){
        std::cout<<"Error: Could not connect to database\n";
        return -1;
    }
    // transaction object
    pqxx::work *txn = new pqxx::work(conn);
    // setting embeddings table if it does not exist
    txn->exec("CREATE TABLE IF NOT EXISTS face_embeddings (id INT PRIMARY KEY, name VARCHAR(20));");

    // loading the face detector model. nms_threshold is kept very high because precision is of utmost importance here
    YOLOv8_face detector = YOLOv8_face(detector_model_path, 0.5, 0.96);

    // loading the embedding generator model
    cv::dnn::Net embed_gen1 = cv::dnn::readNetFromONNX(embed_gen1_path);
    // setting the backend to openvino. SHOULD BE SWITCHED TO CUDA FOR BETTER PERFORMANCE
    embed_gen1.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    embed_gen1.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // loading the embedding generator model
	cv::dnn::Net embed_gen20 = cv::dnn::readNetFromONNX(embed_gen20_path);
    // setting the backend to openvino. SHOULD BE SWITCHED TO CUDA FOR BETTER PERFORMANCE
	embed_gen1.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    embed_gen1.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // loading the faiss index
    faiss::IndexHNSWFlat* index;
	std::ifstream file(faiss_index_path);
	
	int batch_size = 20;

    bool file_exists = file.good();
    if(file.good()){
		std::cout<<"LOADING SAVED INDEX\n";
        faiss::Index *loaded = faiss::read_index(faiss_index_path.c_str());
        index = dynamic_cast<faiss::IndexHNSWFlat*>(loaded);
    }
    else{
        index = generateIndex();
		std::cout<<"CREATING NEW INDEX\n";
	}
    // start the training process
    DIR* dir = opendir(train_dir.c_str());
    std::vector<std::string> users;
    if(dir){
        dirent* subdir;
        while((subdir = readdir(dir))){
            if(subdir->d_type == DT_DIR){
                std::string name = subdir->d_name;
                if(name != "." && name != ".."){
                    users.push_back(name);
                }
            }
        }
    }

	std::vector<cv::Mat> frame_faces; // for batching the frame inputs
	std::vector<std::string> frame_labels;

    std::cout<<"FOUND "<<users.size()<<" USERS"<<std::endl;
    int total_embeddings = 0;
    // finding images in each user folder
	double start = cv::getTickCount();
    for(auto &user:users){
        std::cout<<"CURRENTLY IN "<<user<<std::endl;
		dir = opendir((train_dir+"/"+user).c_str());
        std::vector<std::string> images;
        if(dir){
            dirent* subdir;
            while((subdir = readdir(dir))){
                if(subdir->d_type == DT_REG){
                    std::string img_name = subdir->d_name;
                    images.push_back(img_name);
                }
            }
        }
		
        std::cout<<"\tFOUND "<<images.size()<<" IMAGES"<<std::endl;
        for(auto &img_name:images){
            cv::Mat img;
            std::string img_path = train_dir+"/"+user+"/"+img_name;
            // reading image
            if(is_heic(img_path))
                img = decode_heic(img_path);   
            else
                img = cv::imread(img_path);

			// if(img.cols>1080 || img.rows > 1080)
			// 	cv::resize(img, img, cv::Size(1080, 1080));
            // detecting face
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            std::vector<std::vector<cv::Point>> landmarks;
            std::vector<int> indices;
            detector.detect(img, boxes, confidences, landmarks, true);
            if(boxes.size() == 0){
                std::cout<<"\t\tNO FACE FOUND IN "<<img_name<<std::endl;
                continue;
            }
			std::cout<<"\t\tFOUND "<<boxes.size()<<" FACE(s)"<<std::endl;
            cv::Mat face = img(boxes[0]);
            img.release();
            cv::resize(face, face, cv::Size(160, 160));
			frame_faces.push_back(face);
			frame_labels.push_back(user);
            cv::imshow("TRAINING..",face);
            cv::waitKey(1);
			if(frame_faces.size() == batch_size){
            	cv::Mat blob = cv::dnn::blobFromImages(frame_faces, 1.0 / 128, cv::Size(160, 160), cv::Scalar(127.5, 127.5, 127.5), true, false);
            	embed_gen20.setInput(blob);
				cv::Mat embed = embed_gen20.forward();
				std::vector<float> embeddings(embed.begin<float>(), embed.end<float>());
				index->add(batch_size, embeddings.data());
				int id = index->ntotal-batch_size;
				for(int i=0; i<batch_size; ++i){
					txn->exec("INSERT INTO face_embeddings (id, name) VALUES ("+std::to_string(id)+", '"+frame_labels[i]+"')");
					++id;
				}
				frame_faces.clear();
				frame_labels.clear();
				total_embeddings += batch_size;
			}
        }
    }
	for(int i=0;i<frame_faces.size(); ++i){
		cv::Mat blob = cv::dnn::blobFromImage(frame_faces[i], 1.0 / 128, cv::Size(160, 160), cv::Scalar(127.5, 127.5, 127.5), true, false);
		embed_gen1.setInput(blob);
		cv::Mat embed = embed_gen1.forward();
		std::vector<float> embedding(embed.begin<float>(), embed.end<float>());
		index->add(1, embedding.data());
		txn->exec("INSERT INTO face_embeddings (id, name) VALUES ("+std::to_string(index->ntotal-1)+", '"+frame_labels[i]+"')");
		embedding.clear();
		++total_embeddings;
	}
	double time_taken = (cv::getTickCount() - start)/cv::getTickFrequency();
    std::cout<<"GENERATED "<<total_embeddings<<" EMBEDDINGS IN "<<time_taken<<" SECONDS."<<std::endl;

	std::cout<<"COMMITING TO DB...\n";
	txn->commit();
	std::cout<<"\tSUCCESSFUL\n";
    // saving the index
    faiss::write_index(index, faiss_index_path.c_str());
    std::cout<<"SAVED INDEX TO "<<faiss_index_path<<std::endl;
	conn.close();
    return 0;
}