#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose2D.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>


#include <caffe/caffe.hpp>

using namespace caffe;


class CaffeAnalyzer {
    public:
        CaffeAnalyzer() {}

        void initialize(const string& model_file,
                const string& mean_file,
                const string& trained_file);

        std::vector<float> run(const cv::Mat& img);

    private:
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);

        void Preprocess(const cv::Mat& img,
                std::vector<cv::Mat>* input_channels);

        void SetMean(const string& mean_file);

    private:
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
};




class FloorPlaneClassifier {
    protected:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        image_transport::Subscriber image_sub_;
        image_transport::Publisher image_pub_;

        std::string model_file_;
        std::string mean_file_;
        std::string deploy_file_;
        int thumb_size_;

        CaffeAnalyzer caffe;

        typedef enum {
            UNUSABLE,
            UNTRAVERSABLE,
            TRAVERSABLE
        } ThumbType;

        ThumbType check_thumb(const cv::Mat_<cv::Vec3b> & thumb) {
            std::vector<float> res = caffe.run(thumb);
            // TODO: interprete res to return a label for this image.
            return UNUSABLE;
        }


    protected: // ROS Callbacks



        void image_callback(const sensor_msgs::ImageConstPtr & img_msg) {

            cv::Mat img(cv_bridge::toCvShare(img_msg,"bgr8")->image);
            printf("Got 1 image\n");

            for (int r=0;r<img.rows;r+=thumb_size_) {
                if (r+thumb_size_>img.rows) { continue; }
                for (int c=0;c<img.cols;c+=thumb_size_) {
                    // printf("r %d c %d\n",r,c);
                    if (c+thumb_size_>img.cols) { continue; }
                    cv::Rect roi(c,r,thumb_size_,thumb_size_);
                    cv::Mat thumb = img(roi);
                    ThumbType type = check_thumb(thumb);
                    // Now for display (could be disabled to save CPU)
                    int mark = 0;
                    switch (type) {
                        case UNTRAVERSABLE: 
                            mark = 2; break;
                        case TRAVERSABLE: 
                            mark = 1; break;
                        case UNUSABLE: 
                        default:
                            mark = 0; break;
                    }
                    for (int tr=0;tr<thumb.rows;tr++) {
                        for (int tc=0;tc<thumb.cols;tc++) {
                            thumb.at<cv::Vec3b>(tr,tc)[mark] = 255;
                        }
                    }
                    // getchar();
                }
            }
            printf("\tClassified\n");
            
            cv_bridge::CvImage br(img_msg->header,"bgr8",img);
            image_pub_.publish(br.toImageMsg());
        }

    public:
        FloorPlaneClassifier() : nh_("~"), it_(nh_) {
            nh_.param("thumb_size",thumb_size_,32);
            nh_.param("model_file",model_file_,std::string("caffenet.caffemodel"));
            nh_.param("mean_file",mean_file_,std::string("mean.binaryproto"));
            nh_.param("deploy_file",deploy_file_,std::string("deploy.prototxt"));
            std::string transport = "raw";
            nh_.param("transport",transport,transport);


            caffe.initialize(deploy_file_,mean_file_,model_file_);

            image_pub_ = it_.advertise("image_label",1);
            image_sub_ = it_.subscribe<FloorPlaneClassifier>("image",1, &FloorPlaneClassifier::image_callback,this,transport);
            
        }

};

int main(int argc, char * argv[]) 
{
    ros::init(argc,argv,"floor_plane_classifier");
    FloorPlaneClassifier fp;

    ros::spin();
    return 0;
}


void CaffeAnalyzer::initialize(const string& model_file,
        const string& mean_file, const string& trained_file) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file);
}


std::vector<float> CaffeAnalyzer::run(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
            input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->ForwardPrefilled();

    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin,end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CaffeAnalyzer::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void CaffeAnalyzer::Preprocess(const cv::Mat& img,
        std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    // cv::Mat sample_normalized;
    // cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
            == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

/* Load the mean file in binaryproto format. */
void CaffeAnalyzer::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
        << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}



