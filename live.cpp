// opencv libraries
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/highgui.hpp>
// for loading models
#include<opencv2/dnn.hpp>
// faiss CPU
#include<faiss/IndexHNSW.h>
// faiss loading indexes
#include<faiss/index_io.h>
// postgresql
#include<pqxx/pqxx>
// standard libraries
#include<vector>
#include<iostream>
// for argsort
#include<algorithm>
// for face detector
#include "cv_dnn_ultraface.h"

//detector model
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

UltraFace::UltraFace(const std::string model_path,
	int input_width, int input_length, int num_thread_,
	float score_threshold_, float iou_threshold_, int topk_) {
	num_thread = num_thread_;
	topk = topk_;
	score_threshold = score_threshold_;
	iou_threshold = iou_threshold_;
	in_w = input_width;
	in_h = input_length;
	w_h_list = { in_w, in_h };

	for (auto size : w_h_list) {
		std::vector<float> fm_item;
		for (float stride : strides) {
			fm_item.push_back(ceil(size / stride));
		}
		featuremap_size.push_back(fm_item);
	}

	for (auto size : w_h_list) {
		shrinkage_size.push_back(strides);
	}

	/* generate prior anchors */
	for (int index = 0; index < num_featuremap; index++) {
		float scale_w = in_w / shrinkage_size[0][index];
		float scale_h = in_h / shrinkage_size[1][index];
		for (int j = 0; j < featuremap_size[1][index]; j++) {
			for (int i = 0; i < featuremap_size[0][index]; i++) {
				float x_center = (i + 0.5) / scale_w;
				float y_center = (j + 0.5) / scale_h;

				for (float k : min_boxes[index]) {
					float w = k / in_w;
					float h = k / in_h;
					priors.push_back({ clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1) });
				}
			}
		}
	}
	num_anchors = priors.size();
	/* generate prior anchors finished */

	// ultraface = cv::dnn::readNetFromONNX(model_path + "/version-RFB-320_without_postprocessing.onnx");
	ultraface = cv::dnn::readNetFromONNX(model_path + "/version-RFB-640.onnx");

	ultraface.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
	ultraface.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// UltraFace::~UltraFace() {  }

void UltraFace::detect(cv::Mat &img, std::vector<FaceInfo> &face_list) {
	if (img.empty()) {
		std::cout << "image is empty ,please check!" << std::endl;
		return ;
	}

	cv::resize(img, img, cv::Size(in_w, in_h));
	image_h = img.rows;
	image_w = img.cols;

	//cv::Mat in;

	std::vector<FaceInfo> bbox_collection;
	std::vector<FaceInfo> valid_input;

	cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1.0 / 128, cv::Size(in_w, in_h), cv::Scalar(127, 127, 127), true);
	ultraface.setInput(inputBlob);

	std::vector<cv::String> output_names = { "scores","boxes"};
	std::vector<cv::Mat> out_blobs;
	ultraface.forward(out_blobs, output_names);
	generateBBox(bbox_collection,out_blobs[0], out_blobs[1], score_threshold, num_anchors);
	nms(bbox_collection, face_list);
}

void UltraFace::generateBBox(std::vector<FaceInfo> &bbox_collection, cv::Mat scores, cv::Mat boxes, float score_threshold, int num_anchors) {
	
	float* score_value = (float*)(scores.data);
	float* bbox_value = (float*)(boxes.data);
	for (int i = 0; i < num_anchors; i++) {
		float score = score_value[2 * i + 1];
		if (score_value[2 * i + 1] > score_threshold) {
			FaceInfo rects = {0};
			float x_center = bbox_value[i * 4] * center_variance * priors[i][2] + priors[i][0];
			float y_center = bbox_value[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
			float w = exp(bbox_value[i * 4 + 2] * size_variance) * priors[i][2];
			float h = exp(bbox_value[i * 4 + 3] * size_variance) * priors[i][3];

			rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
			rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
			rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
			rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
			rects.score = clip(score_value[2 * i + 1], 1);
			bbox_collection.push_back(rects);
		}
	}
}

void UltraFace::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
	std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++) {
		if (merged[i])
			continue;
		std::vector<FaceInfo> buf;

		buf.push_back(input[i]);
		merged[i] = 1;

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++) {
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score > iou_threshold) {
				merged[j] = 1;
				buf.push_back(input[j]);
			}
		}
		switch (type) {
		case hard_nms: {
			output.push_back(buf[0]);
			break;
		}
		case blending_nms: {
			float total = 0;
			for (int i = 0; i < buf.size(); i++) {
				total += exp(buf[i].score);
			}
			FaceInfo rects;
			memset(&rects, 0, sizeof(rects));
			for (int i = 0; i < buf.size(); i++) {
				float rate = exp(buf[i].score) / total;
				rects.x1 += buf[i].x1 * rate;
				rects.y1 += buf[i].y1 * rate;
				rects.x2 += buf[i].x2 * rate;
				rects.y2 += buf[i].y2 * rate;
				rects.score += buf[i].score * rate;
			}
			output.push_back(rects);
			break;
		}
		default: {
			printf("wrong type of nms.");
			exit(-1);
		}
		}
	}
}

void select_largest_face(std::vector<FaceInfo> &faces){
	if(faces.size() < 2){
		return;
	}
	int max_area = 0;	
	FaceInfo max_face;
	for(int i = 0; i < faces.size(); i++){
		int area = (faces[i].x2 - faces[i].x1) * (faces[i].y2 - faces[i].y1);
		if(area > max_area){
			max_area = area;
			max_face = faces[i];
		}
	}
	faces.clear();
	faces.push_back(max_face);
}


faiss::IndexHNSWFlat* generate_index(int m, int ef_search, int ef_construction){
	faiss::IndexHNSWFlat* index = new faiss::IndexHNSWFlat(512, m, faiss::METRIC_INNER_PRODUCT);
	index->hnsw.efSearch = ef_search;
	index->hnsw.efConstruction = ef_construction;

	return index;
}
// function for selecting the face with largest area.
template<typename T>
std::vector<size_t> argsort(std::vector<T> &values){
	std::vector<size_t> indices(values.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&values](size_t &i, size_t &j){ return values[i] > values[j];});
	return indices;
}

int main(int argc, char *args[]){
	std::string embed_gen_path = "./models/inceptionResnetV1_30.onnx";
    std::string detector_model_path = "./models";
    // std::string detector_model_path = "./models/yolov8n-face.onnx";

    std::string faiss_index_path = "./models/faissHNSW.index";

	// loading detector model
    // YOLOv8_face detector = YOLOv8_face(detector_model_path, 0.5, 0.96);

	cv::dnn::Net embed_gen = cv::dnn::readNetFromONNX(embed_gen_path);
    // setting the backend to openvino. SHOULD BE SWITCHED TO CUDA FOR BETTER PERFORMANCE
    embed_gen.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    embed_gen.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// faiss::IndexHNSWFlat* index = NULL;
	faiss::Index *index = faiss::read_index(faiss_index_path.c_str());
	// index = dynamic_cast<faiss::IndexHNSWFlat*>(loaded);
	// delete loaded;
	
	pqxx::connection conn = pqxx::connection("dbname=IIIT_EEMS user=postgres password=282606 host = localhost, port = 5432");
	if(!conn.is_open()){
        std::cout<<"Error: Could not connect to database\n";
        return -1;
    }
	// transaction object
    pqxx::work *txn = new pqxx::work(conn);

	// cv::VideoCapture cap("/home/shashank/Downloads/test_01.MOV");
	cv::VideoCapture cap(0);
	if(!cap.isOpened()){
		std::cout << "Error opening video stream." << std::endl;
		return -1;
	}

	// initialising detector according to caputre object resolution
	int width = 640; // int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = 480; //int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	int fps = int(cap.get(cv::CAP_PROP_FPS));
	
	UltraFace detector(detector_model_path, width, height, 4, 0.9);

	const int batch_size = 30;
	const int cosine_dist_threshold = 0.75;

	//test run
	{
		cv::Mat test = cv::imread("/home/shashank/EEMS_CPP/data/train/21bcs100/IMG20230413015026.jpg");
		cv::resize(test, test, cv::Size(640,480));
		std::vector<FaceInfo> faces;
		detector.detect(test, faces);
		cv::resize(test, test, cv::Size(160, 160));
		std::vector<cv::Mat> test_frame;
		for(int i=0;i<batch_size; ++i)
			test_frame.push_back(test);
		cv::Mat blob = cv::dnn::blobFromImages(test_frame, 1.0 / 128, cv::Size(160, 160), cv::Scalar(127.5, 127.5, 127.5), true, false);
		embed_gen.setInput(blob);
		embed_gen.forward();
		test.release();
		blob.release();
	}
	double avg_detect, avg_embed, avg_search, total_avg;
	std::vector<double> detect_times, embed_times, search_times, db_times;
	cv::Mat inp_frame;
	int frame_num = 0;
	double t_start = 0, t_end = 0, detect_start=0, detect_end = 0, db_start = 0, db_end = 0, embed_start = 0, embed_end = 0, search_start = 0, search_end = 0;
	std::vector<cv::Mat> frame_faces;
	std::string label = "No Face";
	bool running = false;
	bool restart = true;
	t_start = cv::getTickCount();
	cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
	for(;;){
		++frame_num;

		//for benchmarking

		if(frame_num == 1000)
			break;
		cap >> inp_frame;
		// if(inp_frame.empty()){
		// 	std::cout << "Error: No frame captured." << std::endl;
		// 	break;
		// }
		// std::vector<FaceInfo> faces;
		// detect_start = (double)cv::getTickCount();
		// detector.detect(inp_frame, faces);
		// detect_end = (double)cv::getTickCount();
		// detect_times.push_back((detect_end - detect_start) / cv::getTickFrequency());

		// // if old face has not disappeared. no need to inference again.
		// if(faces.size() > 0){
		// 	select_largest_face(faces);
			
		// 	cv::rectangle(inp_frame, cv::Point(faces[0].x1, faces[0].y1), cv::Point(faces[0].x2, faces[0].y2), cv::Scalar(0, 255, 0), 2);
		// 	if(label == "No Face" || label == "UNKNOWN")
		// 		label = "IDENTIFYING";
		// 	if(label == "IDENTIFYING"){
		// 		cv::Mat face = inp_frame(cv::Rect(faces[0].x1, faces[0].y1, faces[0].x2 - faces[0].x1, faces[0].y2 - faces[0].y1));
		// 		cv::resize(face, face, cv::Size(160,160));
		// 		frame_faces.push_back(face);
		// 	}
		// }
		// else{
		// 	label = "No Face";
		// 	restart = true;
		// 	frame_faces.clear();
		// }
		// // std::cout<<label<<std::endl;
		// // std::cout<<frame_faces.size()<<std::endl;
		// if(frame_faces.size() == batch_size){
		// 	label = "UNKNOWN";
		// 	cv::Mat inp_blob = cv::dnn::blobFromImages(frame_faces, 1.0 / 128, cv::Size(160, 160), cv::Scalar(127.5, 127.5, 127.5), true);
		// 	embed_gen.setInput(inp_blob);
		// 	embed_start = cv::getTickCount();
		// 	cv::Mat embed = embed_gen.forward();
		// 	embed_end = cv::getTickCount();
		// 	embed_times.push_back((embed_end - embed_start) / cv::getTickFrequency());

		// 	std::vector<float> embed_vec(embed.begin<float>(), embed.end<float>());
		// 	std::vector<long int> labels(batch_size*5);
		// 	std::vector<float> distances(batch_size*5);
		// 	search_start = cv::getTickCount();
		// 	index->search(batch_size, embed_vec.data(), 5, distances.data(), labels.data());
		// 	std::unordered_map<long int, float> scores;

		// 	for(int i=0;i<labels.size();++i){
		// 		if(distances[i]<cosine_dist_threshold)
		// 			continue;
		// 		scores[labels[i]] += distances[i];
		// 	}

		// 	search_end = cv::getTickCount();
		// 	search_times.push_back((search_end - search_start) / cv::getTickFrequency());
		// 	// APPROACH 1. NOT TAKEN. finding max label
		// 	//finding max label
		// 	// float max_score = 0;
		// 	// int key = -1;
		// 	// for(auto &i:scores){
		// 	// 	if(i.second > max_score){
		// 	// 		max_score = i.second;
		// 	// 		key = i.first;
		// 	// 	}
		// 	// }
		// 	// if(key != -1){
		// 	// 	db_start = cv::getTickCount();
		// 	// 	pqxx::result result = txn->exec("SELECT name FROM face_embeddings WHERE id = "+std::to_string(key));
		// 	// 	db_end = cv::getTickCount();
		// 	// 	db_times.push_back((db_end - db_start) / cv::getTickFrequency());	
		// 	// 	if (!result.empty()) 
		// 	// 		label = result[0]["name"].as<std::string>();
		// 	// }

		// 	// APPROACH 2. TAKEN. checking if all the nearest neighbour embeds are of the same person.
		// 	//finding the lables of all indexes
		// 	std::unordered_set<std::string> rolls;

		// 	for(auto &i:scores){
		// 		db_start = cv::getTickCount();
		// 		pqxx::result result = txn->exec("SELECT name FROM face_embeddings WHERE id = "+std::to_string(i.first));
		// 		db_end = cv::getTickCount();
		// 		db_times.push_back((db_end - db_start) / cv::getTickFrequency());
		// 		if (!result.empty()) 
		// 			rolls.insert(result[0]["name"].as<std::string>());
		// 	}
			
		// 	if(rolls.size() == 1)
		// 		label = *rolls.begin();

		// 	frame_faces.clear();
		// 	restart = false;
		// 	if(label != "UNKNOWN")
		// 		label = label + " Marked.";
		// }
		// if(faces.size() > 0)
		// 	cv::putText(inp_frame, label, cv::Point(faces[0].x1, faces[0].y1), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);


		// // std::vector<cv::Rect> boxes;
		// // std::vector<float> confidences;
		// // std::vector<std::vector<cv::Point>> landmarks;
		// // std::vector<int> indices;
		// // detector.detect(inp_frame, boxes, confidences, landmarks, true);

		// cv::imshow("live_feed", inp_frame);
		// cv::waitKey(1);
		writer.write(inp_frame);
	}
	writer.release();
	t_end = cv::getTickCount();
	total_avg = (t_end - t_start) / cv::getTickFrequency();
	double total_fps = frame_num / total_avg;
	std::cout << "Total FPS: " << total_fps << std::endl;
	avg_detect = detect_times.size() / std::accumulate(detect_times.begin(), detect_times.end(), 0.0);
	// double detect_fps = frame_num / avg_detect;
	std::cout << "Detection FPS: " << avg_detect << std::endl;
	avg_embed = embed_times.size() / std::accumulate(embed_times.begin(), embed_times.end(), 0.0);
	// double embed_fps = frame_num / avg_embed;
	std::cout << "Embedding FPS: " << batch_size*avg_embed << std::endl;
	avg_search = search_times.size()/std::accumulate(search_times.begin(), search_times.end(), 0.0) ;
	// double search_fps = frame_num / avg_search;
	std::cout << "Search FPS: " << 20*avg_search << std::endl;
	double avg_db = db_times.size()/std::accumulate(db_times.begin(), db_times.end(), 0.0) ;
	// double db_fps = frame_num / avg_db;
	std::cout << "DB FPS: " << avg_db << std::endl;
	return 0;
}