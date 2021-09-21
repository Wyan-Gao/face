#define _CRT_SECURE_NO_WARNINGS
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace InferenceEngine;

static std::vector<std::string>items = {
	"0","1","2","3","4","5","6","7","8","9","<Anhui >",
	"< Beijing >","< Chongqing >","< Fujian >","< Gansu >","< Guangdong >","< Guangxi >","< Guizhou >",
	"< Hainan >","< Hebei >","< Heilongjiang >","< Henan >","< HongKong >","< Hubei >","< Hunan >",
    "< InnerMongolia >","< Jiangsu >","< Jiangxi >","< Jilin >","< Liaoning >","< Macau >","< Ningxia >","< Qinghai >",
    "< Shaanxi >","< Shandong >","< Shanghai >","< Shanxi >","< Sichuan >","< Tianjin >","< Tibet >",
    "< Xinjiang >","< Yunnan >","< Zhejiang >","< police >","A","B","C","D","E","F","G",
    "H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
};
InferenceEngine::InferRequest plate_request;
std::string plate_input_name1;
std::string plate_input_name2;
std::string plate_output_name;
void load_plate_recog_model();
void fetch_plate_text(cv::Mat &image, const cv::Mat &platROI);

int main(int argc, char** argv) {
#pragma warning(disable:4996)
	InferenceEngine::Core ie;
	load_plate_recog_model();
	std::string xml = "E:/models/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml";
	std::string bin = "E:/models/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.bin";

	cv::Mat src = cv::imread("E:/vcworkplace/rad_car.jpg");
	cv::namedWindow("input", cv::WINDOW_FREERATIO);
	int im_h = src.rows;
	int im_w = src.cols;

	InferenceEngine::CNNNetwork network = ie.ReadNetwork(xml, bin);
	InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
	InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();

	std::string input_name = "";
	for (auto item : inputs) {
		input_name = item.first;
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
		// input_data->getPreProcess().setColorFormat(ColorFormat::BGR);
		std::cout << "input name: " << input_name << std::endl;
	}

	std::string output_name = "";
	for (auto item : outputs) {
		output_name = item.first;
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
		std::cout << "output name: " << output_name << std::endl;
	}

	auto executable_network = ie.LoadNetwork(network, "CPU");
	auto infer_request = executable_network.CreateInferRequest();

	auto input = infer_request.GetBlob(input_name);
	size_t num_channels = input->getTensorDesc().getDims()[1];
	size_t h = input->getTensorDesc().getDims()[2];
	size_t w = input->getTensorDesc().getDims()[3];
	size_t image_size = h * w;
	cv::Mat blob_image;
	cv::resize(src, blob_image, cv::Size(w, h));

	// HWC =¡·NCHW
	unsigned char* data = static_cast<unsigned char*>(input->buffer());
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t ch = 0; ch < num_channels; ch++) {
				data[image_size*ch + row * w + col] = blob_image.at<cv::Vec3b>(row, col)[ch];
			}
		}
	}

	infer_request.Infer();

	auto output = infer_request.GetBlob(output_name);
	const float* detection_out = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
	const SizeVector outputDims = output->getTensorDesc().getDims();
	std::cout << outputDims[2] << "x" << outputDims[3] << std::endl;
	const int max_count = outputDims[2];
	const int object_size = outputDims[3];

	int padding = 5;
	for (int n = 0; n < max_count; n++) {
		float label = detection_out[n*object_size + 1];
		float confidence = detection_out[n*object_size + 2];
		float xmin = detection_out[n*object_size + 3] * im_w;
		float ymin = detection_out[n*object_size + 4] * im_h;
		float xmax = detection_out[n*object_size + 5] * im_w;
		float ymax = detection_out[n*object_size + 6] * im_h;
		if (confidence > 0.25) {
			printf("label id : %d \n", static_cast<int>(label));
			cv::Rect box;
			box.x = static_cast<int>(xmin);
			box.y = static_cast<int>(ymin);
			box.width = static_cast<int>(xmax - xmin);
			box.height = static_cast<int>(ymax - ymin);
			if (label == 2) {
				cv::Rect plate_roi;
				plate_roi.x = box.x - padding;
				plate_roi.y = box.y - padding;
				plate_roi.width = box.width + 2* padding;
				plate_roi.height = box.height + 2* padding;
				fetch_plate_text(src, src(plate_roi));
			}
			
			
			cv::rectangle(src, box, cv::Scalar(0, 0, 255), 2, 8, 0);
			cv::putText(src, cv::format("%.2f", confidence), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
		}
	}
	cv::imshow("input", src);
	cv::waitKey(0);
	return 0;
}


void load_plate_recog_model(){
	InferenceEngine::Core ie;
	std::string xml = "E:/vcworkplace/models/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.xml";
	std::string bin = "E:/vcworkplace/models/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.bin";

	

	InferenceEngine::CNNNetwork network = ie.ReadNetwork(xml, bin);
	InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
	InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();

	int cnt = 0;
	for (auto item : inputs) {
		if (cnt == 0) {
			plate_input_name1 = item.first;
			auto input_data = item.second;
			input_data->setPrecision(Precision::U8);
			input_data->setLayout(Layout::NCHW);
			// input_data->getPreProcess().setColorFormat(ColorFormat::BGR);
		}
		if (cnt == 1) {
			plate_input_name2 = item.first;
			auto input_data = item.second;
			input_data->setPrecision(Precision::FP32);

		}
		
		std::cout << "plate output name: " <<(cnt+1)<<":"<< item.first << std::endl;
		cnt++;
	}


	for (auto item : outputs) {
		plate_output_name = item.first;
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
		std::cout << "plate_output_name: " << plate_output_name << std::endl;
	}

	auto executable_network = ie.LoadNetwork(network, "CPU");
	plate_request = executable_network.CreateInferRequest();
}
void fetch_plate_text(cv::Mat &image, const cv::Mat &platROI) {
	auto input1 = plate_request.GetBlob(plate_input_name1);
	size_t num_channels = input1->getTensorDesc().getDims()[1];
	size_t h = input1->getTensorDesc().getDims()[2];
	size_t w = input1->getTensorDesc().getDims()[3];
	size_t image_size = h * w;
	cv::Mat blob_image;
	cv::resize(platROI, blob_image, cv::Size(w, h));

	// HWC =¡·NCHW
	unsigned char* data = static_cast<unsigned char*>(input1->buffer());
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t ch = 0; ch < num_channels; ch++) {
				data[image_size*ch + row * w + col] = blob_image.at<cv::Vec3b>(row, col)[ch];
			}
		}
	}

	auto input2 = plate_request.GetBlob(plate_input_name2);
	int max_sequence = input2->getTensorDesc().getDims()[0];
	float* blob2 = input2->buffer().as<float*>();
	blob2[0] = 0.0f;
	std::fill(blob2 + 1, blob2 + max_sequence, 1.0f);

	plate_request.Infer();
	auto output = plate_request.GetBlob(plate_output_name);
	const float* plate_data = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());//maybe9;03
	std::string result;
	for (int i = 0; i < max_sequence; i++) {
		if (plate_data[i] == -1)
			break;
		result += items[std::size_t(plate_data[i])];//maybe10:37
	}
	std::cout << result << std::endl;
	cv::putText(image, result.c_str(),cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);

}
