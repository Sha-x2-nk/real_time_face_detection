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
// dlib
#include<dlib/image_processing/shape_predictor.h>
#include<dlib/opencv/cv_image.h>
#include<dlib/image_processing.h>
#include<dlib/geometry/rectangle.h>
// eigen for norm calculation
#include<eigen3/Eigen/Dense>
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
	ultraface = cv::dnn::readNetFromONNX(model_path);

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

inline double eyeAspectRatio(const Eigen::MatrixXd& eye) {
    // Compute the Euclidean distances between the vertical eye landmarks
    double A = (eye.row(1) - eye.row(5)).norm();
    double B = (eye.row(2) - eye.row(4)).norm();

    // Compute the Euclidean distance between the horizontal eye landmarks
    double C = (eye.row(0) - eye.row(3)).norm();

    // Compute the aspect ratio
    double ear = (A + B) / (2.0 * C);

    return ear;
}

class EEMS{
	private:
	pqxx::work *txn;
	cv::dnn::Net embed_gen;
	UltraFace *detector;
	dlib::shape_predictor sp;
	bool check_liveness;
	faiss::Index *index;
	int batch_size, check_blink_for_frames;
	float cosine_threshold, ear_threshold;
	public:
	EEMS(pqxx::connection &conn, bool liveness = false, int batch_size = 20, float cosine_threshold = 0.75, float ear_threshold = 0.2, int check_blink_for_frames = 1, std::string detector_path = "./models/version-RFB-640.onnx", std::string embedgen_path = "./models/inceptionResnetV1_20.onnx", std::string faiss_index_path = "./models/faissHNSW.index", std::string dlib_shape_predictor = "./models/shape_predictor_68_face_landmarks.dat"){
		detector = new UltraFace(detector_path, 640, 480, 4, 0.9);
		embed_gen = cv::dnn::readNetFromONNX(embedgen_path);
		embed_gen.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
		embed_gen.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		check_liveness = liveness;
		if(check_liveness)
			dlib::deserialize(dlib_shape_predictor) >> sp;
		this->batch_size = batch_size;
		this->cosine_threshold = cosine_threshold;
		this->ear_threshold = ear_threshold;
		this->check_blink_for_frames = check_blink_for_frames;
		index = faiss::read_index(faiss_index_path.c_str());
		txn = new pqxx::work(conn);
	}
	
	void testRun(){
		cv::Mat test = cv::imread("./test.webp");
		cv::resize(test, test, cv::Size(640,480));
		std::vector<FaceInfo> faces;
		detector->detect(test, faces);
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

	void live(){
		testRun();
		cv::VideoCapture cap(0);
		if(!cap.isOpened()){
			std::cout << "Error opening video stream." << std::endl;
			return;
		}
		cv::namedWindow("LIVE FEED", cv::WINDOW_FULLSCREEN);
		cv::Mat inp_frame;
		std::vector<cv::Mat> frame_faces;
		float area;
		int open_count = 0, closed_count = 0;
		bool is_live = false, is_open = false, is_closed = false, come_closer;
		std::string label = "IDENTIFYING..";
		for(;;){
			cap >> inp_frame;
			if(inp_frame.empty()){
				std::cout << "Error: No frame captured." << std::endl;
				break;
			}
			std::vector<FaceInfo> faces;
			detector->detect(inp_frame, faces);
			
			if(faces.size()>0){
				select_largest_face(faces);
				cv::Rect face_box = cv::Rect(faces[0].x1, faces[0].y1, faces[0].x2 - faces[0].x1, faces[0].y2 - faces[0].y1);
				cv::rectangle(inp_frame, face_box, cv::Scalar(0, 255, 0), 2);
				if(label == "IDENTIFYING.."){
					cv::Mat face = inp_frame(face_box);
					cv::resize(face, face, cv::Size(160,160));
					area = 1.0*face_box.area()/inp_frame.size().area();
					if(area < 0.15)
						come_closer = true;
					else{
						come_closer = false;
						// blink checker
						if(check_liveness && !is_live){
							double live_start = cv::getTickCount();
							dlib::array2d<dlib::bgr_pixel> dlibImage;
							dlib::assign_image(dlibImage, dlib::cv_image<dlib::bgr_pixel>(face));
							dlib::rectangle faceRect(0, 0, dlibImage.nc() - 1, dlibImage.nr() - 1);
							dlib::full_object_detection landmarks = sp(dlibImage, faceRect);
							Eigen::MatrixXd left_eye(6, 2), right_eye(6, 2);
							left_eye << landmarks.part(36).x(), landmarks.part(36).y(),
										landmarks.part(37).x(), landmarks.part(37).y(),
										landmarks.part(38).x(), landmarks.part(38).y(),
										landmarks.part(39).x(), landmarks.part(39).y(),
										landmarks.part(40).x(), landmarks.part(40).y(),
										landmarks.part(41).x(), landmarks.part(41).y();
							right_eye << landmarks.part(42).x(), landmarks.part(42).y(),
										landmarks.part(43).x(), landmarks.part(43).y(),
										landmarks.part(44).x(), landmarks.part(44).y(),
										landmarks.part(45).x(), landmarks.part(45).y(),
										landmarks.part(46).x(), landmarks.part(46).y(),
										landmarks.part(47).x(), landmarks.part(47).y();
							double left_ear = eyeAspectRatio(left_eye);
							double right_ear = eyeAspectRatio(right_eye);
							double ear = (left_ear + right_ear) / 2.0;
							// checking blink :  open -> shut -> open
							if(is_open == false && is_closed == false && ear>ear_threshold) { // open
								++open_count;
								if(open_count == check_blink_for_frames)
									is_open = true;
							}
							else if(is_open == true && is_closed == false && ear<ear_threshold){ // shut
								++closed_count;
								if(closed_count == check_blink_for_frames)
									is_open = false, is_closed = true;
							}
							else if(is_open == false && is_closed == true && ear>ear_threshold){ // open
								++open_count;
								if(open_count == check_blink_for_frames)
									is_open = true, is_live = true;
							}
						}
						if(frame_faces.size() < batch_size)
							frame_faces.push_back(face);
					}
				}
			} 
			if(faces.size() == 0){
				label = "IDENTIFYING..";
				is_live = is_open = is_closed = false;
				open_count = closed_count = 0;
				frame_faces.clear();
			}
			if(frame_faces.size() == batch_size && (!check_liveness || is_live)){
				cv::Mat inp_blob = cv::dnn::blobFromImages(frame_faces, 1.0 / 128, cv::Size(160, 160), cv::Scalar(127.5, 127.5, 127.5), true);
				embed_gen.setInput(inp_blob);
				cv::Mat embed = embed_gen.forward();
				std::vector<float> embed_vec(embed.begin<float>(), embed.end<float>());

				std::vector<long int> labels(batch_size*5);
				std::vector<float> distances(batch_size*5);
				index->search(batch_size, embed_vec.data(), 5, distances.data(), labels.data());
				
				std::unordered_set<std::string> rolls;
				for(int i=0;i<labels.size(); ++i){
					if(distances[i] < cosine_threshold)
						continue;
					pqxx::result result = txn->exec("SELECT name FROM face_embeddings WHERE id = " + std::to_string(labels[i]));
					std::string roll = result[0]["name"].as<std::string>();
					rolls.insert(roll);
				}
				if(rolls.size() == 1)
					label = *rolls.begin();
				frame_faces.clear();
			}
			if(faces.size()>0){
				if(come_closer == true)
					cv::putText(inp_frame, "COME CLOSER..", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
				else if(check_liveness){
					if(is_live)
						cv::putText(inp_frame, "WAITING FOR BLINK..", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
					else
						cv::putText(inp_frame, "BLINK CAUGHT", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
				}
				cv::putText(inp_frame, label, cv::Point(faces[0].x1, faces[0].y1), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
			}
			else{
				cv::putText(inp_frame, "No Face Found.", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
			}
			cv::imshow("LIVE FEED", inp_frame);
			cv::waitKey(1);
		}
	}
	
	~EEMS(){
		cv::destroyAllWindows();
	}
};

int main(int argc, char *args[]){
	pqxx::connection conn =pqxx::connection("dbname=IIIT_EEMS user=postgres password=282606 host = localhost, port = 5432");
	EEMS live(conn);
	live.live();
	conn.close();
}