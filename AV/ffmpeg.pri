FFMPEG_PATH = $$PWD/../3rd/ffmpeg

INCLUDEPATH += $$FFMPEG_PATH/include $$PWD

LIBS += -L$$FFMPEG_PATH/bin -lavcodec -lavformat -lavutil

SOURCES += $$PWD/AVDecoderVideo.cpp
HEADERS += $$PWD/AVDecoderVideo.h
