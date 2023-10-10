#include "AVDecoderVideo.h"

#include <iostream>
#include <fstream>

using namespace std;

AVDecoderVideo::AVDecoderVideo()
{
    avformat_network_init();
}

AVDecoderVideo::AVDecoderVideo(const cv::String &url)
{
    avformat_network_init();
    open(url);
}

AVDecoderVideo::~AVDecoderVideo()
{
    release();
}

bool AVDecoderVideo::isOpened() const
{
    return mIsOpened;
}

bool AVDecoderVideo::open(const String &name)
{
    int res = 0;
    close();

    std::string str = name;

    res = avformat_open_input(&mFmt, str.c_str(), nullptr, nullptr);
    if(res < 0){
        printf("Error %d\n", res);
        return false;
    }
    res = avformat_find_stream_info(mFmt, nullptr);

    for(int i = 0; i < mFmt->nb_streams; ++i){
        auto str = mFmt->streams[i];
        if(!str->codecpar){
            continue;
        }
        if(str->codecpar->codec_type == AVMEDIA_TYPE_VIDEO){
            mVideo = str;
        }
        if(str->codecpar->codec_type == AVMEDIA_TYPE_AUDIO){
            mAudio.push_back(str);
        }
    }
    if(mVideo == nullptr){
        close();
        return false;
    }
    mCdc = avcodec_find_decoder(mVideo->codecpar->codec_id);
    if(mCdc == nullptr){
        close();
        return false;
    }
    mCtx = avcodec_alloc_context3(mCdc);
    res = avcodec_parameters_to_context(mCtx, mVideo->codecpar);
    if(res < 0){
        close();
        return false;
    }
    res = avcodec_open2(mCtx, mCdc, nullptr);
    if(res < 0){
        close();
        return false;
    }
    mIsOpened = true;
    return true;
}

void AVDecoderVideo::close()
{
    stopPlay();

    mIsOpened = false;
    if(mCtx){
        avcodec_close(mCtx);
        avcodec_free_context(&mCtx);
        mCtx = nullptr;
    }
    if(mFmt){
        avformat_close_input(&mFmt);
        mFmt = nullptr;
    }
    mVideo = nullptr;
    mAudio.clear();
}

void AVDecoderVideo::release()
{
    mIsDone = true;

    if(mThr){
        mThr->join();
    }
    mThr.reset();

    close();
}

void AVDecoderVideo::setCanNext(bool val)
{
    mIsCanNext = val;
}

inline int Min(int a, int b) {
    return a <= b ? a : b;
}

inline int Max(int a, int b) {
    return a >= b ? a : b;
}

inline int clamp255(int val)
{
    return Min(255, Max(0, val));
}

void ycbcr2rgb(int Y, int U, int V, int& r, int& g, int& b)
{
    int tb = (Y + 1.4075 * (V - 128));;
    // or fast integer computing with a small approximation
    // rTmp = yValue + (351*(vValue-128))>>8;
    int tg = (Y - 0.3455 * (U - 128) - (0.7169 * (V - 128)));
    // gTmp = yValue - (179*(vValue-128) + 86*(uValue-128))>>8;
    int tr = (Y + 1.7790 * (U - 128));
    // bTmp = yValue + (443*(uValue-128))>>8;
    r = clamp255(tr);
    g = clamp255(tg);
    b = clamp255(tb);
}

void RGBfromYUV(uchar& R, uchar& G, uchar& B, int Y, int U, int V)
{
  Y -= 16;
  U -= 128;
  V -= 128;
  R = 1.164 * Y             + 1.596 * V;
  G = 1.164 * Y - 0.392 * U - 0.813 * V;
  B = 1.164 * Y + 2.017 * U;
}

void ycbcr2rgb(int ry, int cr, int cb, uchar *rgb)
{
    int r, g, b;
    //RGBfromYUV(r, g, b, ry, cr, cb);
    ycbcr2rgb(ry, cr, cb, r, g, b);
    rgb[0] = r;
    rgb[1] = g;
    rgb[2] = b;
}

Image decode_frame(AVFrame* frame)
{
    Image ret(frame->height, frame->width, CV_8UC3);

    int C = 3;
    int w = frame->width;
    int h = frame->height;
    if(frame->format == AV_PIX_FMT_YUV420P || frame->format == AV_PIX_FMT_YUVJ420P){
        for(int i = 0; i < h; ++i){
            uchar* yc = ret.ptr<uchar>(i);
            uchar* ls = frame->data[0] + i * frame->linesize[0];
            for(int j = 0; j < w; ++j){
                yc[C * j + 0] = ls[j];
            }
        }
        for(int i = 0; i < h/2; ++i){
            uchar* yc0 = ret.ptr<uchar>(i * 2 + 0);
            uchar* yc1 = ret.ptr<uchar>(i * 2 + 1);
            uchar* ls1 = frame->data[1] + i * frame->linesize[1];
            uchar* ls2 = frame->data[2] + i * frame->linesize[2];
            for(int j = 0; j < w/2; ++j){
                uchar y00 = yc0[C * (j * 2 + 0)];
                uchar y01 = yc0[C * (j * 2 + 1)];
                uchar y10 = yc1[C * (j * 2 + 0)];
                uchar y11 = yc1[C * (j * 2 + 1)];
                uchar cr = ls1[j];
                uchar cb = ls2[j];
                ycbcr2rgb(y00, cr, cb, &yc0[C * (j * 2 + 0)]);
                ycbcr2rgb(y01, cr, cb, &yc0[C * (j * 2 + 1)]);
                ycbcr2rgb(y10, cr, cb, &yc1[C * (j * 2 + 0)]);
                ycbcr2rgb(y11, cr, cb, &yc1[C * (j * 2 + 1)]);
            }
        }
    }
    return ret;
}

Image AVDecoderVideo::getNextFrame()
{
    if(!mCtx || !mFmt){
        return {};
    }
    int res = 0;
    auto pkt = av_packet_alloc();
    auto frame = av_frame_alloc();
    Image ret;
    while(ret.empty()){
        res = av_read_frame(mFmt, pkt);
        if(res < 0){
            return {};
        }
        res = avcodec_send_packet(mCtx, pkt);
        res = avcodec_receive_frame(mCtx, frame);
        if(res >= 0){
            ret = decode_frame(frame);
            break;
        }
        av_packet_unref(pkt);
        av_frame_unref(frame);
    }
    av_packet_free(&pkt);
    av_frame_free(&frame);

    return ret;
}

bool AVDecoderVideo::isPlayed() const
{
    return mIsPlay;
}

void AVDecoderVideo::startPlay()
{
    if(mIsPlay){
        return;
    }
    stopPlay();

    mIsPlay = true;
    mIsWorking = true;
    mIsDone = false;
    if(!mThr){
        try{
            mThr.reset(new std::thread([this](){
                cout << "begin;" << endl;
                streaming();
                cout << "end;" << endl;
            }));
        }catch(std::exception &e){
            cout << e.what() << endl;
        }
    }
}

void AVDecoderVideo::stopPlay()
{
    mIsPlay = false;
    waitUntilWork();
    if(mFmt){
        avformat_seek_file(mFmt, mVideo->index, 0, 0, 0, 0);
    }
 }

void AVDecoderVideo::setFunImage(funimgop f)
{
    mImgOp = f;
}

void AVDecoderVideo::streaming()
{
    if(!mIsPlay || !mIsOpened){
        return;
    }
    mIsWorking = true;

    avformat_seek_file(mFmt, mVideo->index, 0, 0, 0, 0);

    while(!mIsDone){
        if(mIsPlay){
            if(!mIsCanNext){
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            auto img = getNextFrame();
            if(img.empty()){
                mIsPlay = false;
                continue;
            }
            if(mImgOp){
                img = mImgOp(img);
            }
        }else{
            mIsWorking = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    mIsPlay = false;
    mIsWorking = false;
}

void AVDecoderVideo::waitUntilWork()
{
   using namespace std;
   while(mIsWorking){
       std::this_thread::sleep_for(5ms);
   }
}

bool AVDecoderVideo::open(const cv::String &filename, int apiPreference)
{
   return open(filename);
}

bool AVDecoderVideo::open(const cv::String &filename, int apiPreference, const std::vector<int> &params)
{
   return open(filename);
}

bool AVDecoderVideo::read(cv::OutputArray image)
{
   auto im = getNextFrame();
   if(im.empty()){
       return false;
   }
   image.assign(im);
   return true;
}
