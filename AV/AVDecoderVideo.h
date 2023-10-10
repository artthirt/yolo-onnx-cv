#ifndef AVDECODERVIDEO_H
#define AVDECODERVIDEO_H

#include <opencv2/opencv.hpp>

#include <functional>
#include <thread>

extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

using Image = cv::Mat;
using String = cv::String;

using funimgop = std::function<Image(Image)>;

class AVDecoderVideo: public cv::VideoCapture
{
public:
    AVDecoderVideo();
    AVDecoderVideo(const cv::String& url);
    ~AVDecoderVideo();

    bool isOpened() const;
    bool open(const String& name);
    void close();
    void release();

    void setCanNext(bool val);

    Image getNextFrame();

    bool isPlayed() const;
    void startPlay();
    void stopPlay();

    void setFunImage(funimgop f);

private:
    AVFormatContext *mFmt = nullptr;
    AVCodecContext *mCtx = nullptr;
    const AVCodec* mCdc = nullptr;
    AVStream *mVideo = nullptr;
    std::vector<AVStream*> mAudio;
    bool mIsOpened = false;
    bool mIsPlay = false;
    bool mIsWorking = false;
    bool mIsCanNext = true;
    bool mIsDone = false;
    std::unique_ptr<std::thread> mThr;

    funimgop mImgOp;

    void streaming();
    void waitUntilWork();

    // VideoCapture interface
public:
    bool open(const cv::String &filename, int apiPreference);
    bool open(const cv::String &filename, int apiPreference, const std::vector<int> &params);
    bool read(cv::OutputArray image);
};

#endif // AVDECODERVIDEO_H
