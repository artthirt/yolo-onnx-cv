#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <cstdlib>

#include "AVDecoderVideo.h"

using namespace std;
using namespace cv;

struct BBox{
    Rect2d box;
    float p = 0;
    int   c = 0;
};

class OnnxModel{
public:
    /**
     * tr = Compose([ToTensor(), Normalize(
     *  mean=[0.485, 0.456, 0.406],
     *  std=[0.229, 0.224, 0.225],)])
     */
    cv::dnn::Net model;
    cv::Size image_size = {640, 640};
    std::vector<string> outLs;

    OnnxModel(){

    }

    bool load(string cfg, string weight, cv::Size sz){
        try{
            if(!sz.empty()){
                image_size = sz;
            }
            model = cv::dnn::readNetFromDarknet(
                cfg,
                weight
                );
            model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        }catch(std::exception& e){
            cout << "Error reading model: " << e.what() << endl;
            return false;
        }
        cout << "layers model\n";
        auto ls =  model.getLayerNames();
        for(const auto &it: ls){
            cout << "  " << it << endl;
        }
        for(const auto &it: model.getUnconnectedOutLayers()){
            outLs.push_back(ls[it - 1]);
        }
        return true;
    }

    bool load(string file, cv::Size sz){
//        try
        {
            if(!sz.empty()){
                image_size = sz;
            }
            //cv::dnn::enableModelDiagnostics(true);
            model = cv::dnn::readNetFromONNX(file);
            model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        }
//        catch(std::exception& e){
//            cout << "Error reading model: " << e.what() << endl;
//            return false;
//        }
        cout << "layers model\n";
        auto ls =  model.getLayerNames();
        for(const auto &it: ls){
            cout << "  " << it << endl;
        }
        for(const auto &it: model.getUnconnectedOutLayers()){
            outLs.push_back(ls[it - 1]);
        }
        return true;
    }

    vector<BBox> forward(cv::Mat image, float thresh = 0.2, float div = 1){
        if(image.empty()){
            return {};
        }
        vector<cv::Mat> outB;
        cv::Mat out;
        //cout << "set input\n";
        image.convertTo(image, CV_32F, 1./255);

        auto blob_image = cv::dnn::blobFromImage(image, 1, image_size, {}, true);

        //cout << blob_image.size << endl;

        model.setInput(blob_image);
        //cout << "begin forward\n";
        model.forward(outB, outLs);

        if(outB.size() > 1){
//            for(auto& it: outB){
//                cout << it.size << endl;
//            }
            cv::vconcat(outB, out);
        }else if(outB.size() == 1){
            out = outB[0];
        }
//        for(int i = 0; i < out.size.dims(); ++i){
//            cout << out.size[i] << " ";
//        }
        //cout << out.size << endl;

        auto bb = nms(out, thresh, 0.8, div, true);

        return bb;
    }

    bool inrect(const Rect2f &a, const Rect2f& b) const{
        float xa1 = a.x, ya1 = a.y;
        float xa2 = a.x + a.width, ya2 = a.y + a.height;

        float xb1 = b.x, yb1 = b.y;
        float xb2 = b.x + b.width, yb2 = b.y + b.height;

        return xb1 >= xa1 && xb2 <= xa2 && yb1 >= ya1 && yb2 <= ya2;
    }

    vector<BBox> nms(const cv::Mat& res, float thresh = 0.2, float unionK = 0.9, bool divSize = false, bool use_cv_nms = false)
    {
        int rows = 0, cols = 0;
        if(res.size.dims() < 2){
            return {};
        }
        if(res.size.dims() == 3){
            rows = res.size[1];
            cols = res.size[2];
        }
        if(res.size.dims() == 2){
            rows = res.size[0];
            cols = res.size[1];
        }

        float divX = 1, divY = 1;
        if(divSize){
            divX = image_size.width;
            divY = image_size.height;
        }

        vector<BBox> bbr;
        vector<Rect2d> rrect;
        vector<float> scores;
        vector<int> classes;
        int cs = cols - 5;
        float *fptr = (float*)res.data;
        for(int i = 0; i < rows; ++i){
            const float *fp =  &fptr[i * cols];
            float x = fp[0]/divX;
            float y = fp[1]/divY;
            float w = fp[2]/divX;
            float h = fp[3]/divY;
            float p = fp[4];
            if(p > thresh){
                int idm = 5;
                float m = fp[idm];
                for(int j = idm + 1; j < cols; ++j){
                    if(fp[j] > m){
                        m = fp[j], idm = j;
                    }
                }
                float fc = p * m;
                if(fc > thresh){
                    int c = idm - 5 + 1;
                    //bb.emplace_back(BBox{Rect2f{x - w/2, y - h/2, w, h}, fc, c});
                    rrect.push_back(Rect2f{x - w/2, y - h/2, w, h});
                    scores.push_back(fc);
                    classes.push_back(c);
                }
            }
        }

        set<int> rm;
        if(!use_cv_nms){
            if(!rrect.empty()){
                cout << "<<<<<----------\n";
                for(int i = 0; i < rrect.size() - 1; ++i){
                    const auto &r1 = rrect[i];
                    float p1 = scores[i];
                    float a1 = r1.area();
                    int c1 = classes[i];
                    if(rm.count(i) == 0){
                        for(int j = i + 1; j < rrect.size(); ++j){
                            const auto &r2 = rrect[j];
                            float p2 = scores[j];
                            float a2 = r2.area();
                            int c2 = classes[j];
                            float x1 = fmin(r1.x, r2.x);
                            float x2 = fmax(r1.x + r1.width, r2.x + r2.width);
                            float y1 = fmin(r1.y, r2.y);
                            float y2 = fmax(r1.y + r1.height, r2.y + r2.height);
                            float a0 = (x2 - x1) * (y2 - y1);
                            if(a0 > 0){
                                float k1 = a1/a0, k2 = a2/a0;
                                bool inI = c1 == c2 && inrect(r1, r2);
                                bool inJ = c1 == c2 && inrect(r2, r1);
                                if(k1 > unionK && k2 > unionK){
                                    if(p1 > p2)
                                        rm.insert(j);
                                    else
                                        rm.insert(i);
    //                                printf("1   %.2f:%d   %.2f:%d (%.2f %.2f %.2f %.2f) (%.2f %.2f %.2f %.2f)\n", p1, b1.c, p2, b2.c,
    //                                       r1.x, r2.y, r1.br().x, r1.br().y,
    //                                       r2.x, r2.y, r2.br().x, r2.br().y);
                                }else if(inI){
                                    if(p1 > p2)
                                        rm.insert(j);
    //                                printf("2   %.2f:%d   %.2f:%d\n", p1, b1.c, p2, b2.c);
                                }else if(inJ){
                                    if(p2 > p1)
                                        rm.insert(i);
    //                                printf("3   %.2f:%d   %.2f:%d\n", p1, b1.c, p2, b2.c);
                                }
                            }
                        }
                    }
                }
                //cout << "---------->>>>>\n";
            }
            for(int i = 0; i < rrect.size(); ++i){
                if(rm.count(i) == 0){
                    bbr.emplace_back(BBox{rrect[i], scores[i], classes[i]});
                }
            }
        }else{
            vector<int> indices;
            cv::dnn::NMSBoxes(rrect, scores, thresh, 0.5, indices);
            for(auto i: indices){
                bbr.emplace_back(BBox{rrect[i], scores[i], classes[i]});
            }
        }

        return bbr;
    }
};

class Args{
public:
    string url;
    string yolo_cfg;
    string yolo_weight;
    string yolo_onnx;
    Size yolo_size = {640, 640};
    bool scaled = false;

    bool parse(int argc, char *argv[])
    {
        for(int i = 1; i < argc; ++i){
            string arg = argv[i];
            if(arg == "-url"){
                url = argv[++i];
            }
            if(arg == "-cfg"){
                yolo_cfg = argv[++i];
            }
            if(arg == "-weights"){
                yolo_weight = argv[++i];
            }
            if(arg == "-onnx"){
                yolo_onnx = argv[++i];
            }
            if(arg == "-w"){
                yolo_size.width = stoi(argv[++i]);
            }
            if(arg == "-h"){
                yolo_size.height = stoi(argv[++i]);
            }
            if(arg == "-scaled"){
                scaled = 1;
            }
        }
        return !yolo_size.empty() && (!yolo_onnx.empty() || is_darknet());
    }
    bool is_darknet() const{
        return !yolo_cfg.empty() && !yolo_weight.empty();
    }
    bool is_onnx() const{
        return !yolo_onnx.empty();
    }
};

int main(int argc, char *argv[])
{
    cout << cv::getBuildInformation() << endl;

    Args args;
    if(!args.parse(argc, argv)){
        cout << "wrong arguments\n";
        return 1;
    }

    OnnxModel model;
    bool div = args.scaled;
    if(args.is_darknet()){
        if(!model.load(args.yolo_cfg, args.yolo_weight, args.yolo_size)){
            cout << "model not loaded\n";
            return 1;
        }
    }else if(args.is_onnx()){
        if(!model.load(args.yolo_onnx, args.yolo_size)){
            cout << "model not loaded\n";
            return 1;
        }
    }else{
        cout << "wrong arguments\n";
    }

    AVDecoderVideo cap(args.url);

    cv::namedWindow("output", cv::WINDOW_FREERATIO | cv::WINDOW_NORMAL);

    auto start = chrono::steady_clock::now();
    int frames = 0;
    float fps = 0;

    cv::Mat templ;
    while(cap.read(templ)){

        auto out = model.forward(templ, 0.2, div);
        float w = templ.cols;
        float h = templ.rows;
        //cout << templ.cols << "x" << templ.rows << endl;
        for(auto it: out){
            Rect r(it.box.x * w, it.box.y * h, it.box.width * w, it.box.height * h);
            stringstream ss;
            ss << it.c << " p=" << setprecision(2) << it.p;
            putText(templ, ss.str(), r.tl() + Point(0, 20), 1, 1, {0, 0, 255}, 1);
            rectangle(templ, r, {0, 255, 0}, 2);
        }

        auto delay = 1./1e+3 * chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
        if(delay > 2){
            fps = frames / delay;
            frames = 0;
            start = chrono::steady_clock::now();
        }
        putText(templ, to_string(fps), Point(templ.cols - 150, 30), 1, 2, {0, 0, 255}, 2);

        frames++;

        cv::imshow("output", templ);

        if(cv::waitKey(1) == 27){
            break;
        }
    }

    return 0;
}
