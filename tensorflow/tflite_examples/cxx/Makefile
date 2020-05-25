TFPATH := /wrk/xsjhdnobkup6/answery/learn/tensorflow/
MAKEFILE_DIR := $(TFPATH)/tensorflow/lite/tools/make

INCLUDES := \
-I. \
-I$(TFPATH) \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/absl \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/ruy \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(MAKEFILE_DIR)/downloads/fp16/include 


LIBS := -lstdc++  -lpthread -lm -lz  -ldl -ltensorflow-lite

all: imghelper label_image
	$(CXX) -g imghelper.o label_image.o $(INCLUDES) -o imgcls  $(LIBS)

imghelper:
	$(CXX) -O2 -g -c bitmap_helpers.cc $(INCLUDES) -o imghelper.o  $(LIBS)

label_image:
	$(CXX) -O2 -g -c label_image.cc $(INCLUDES) -o label_image.o  $(LIBS)

clean:
	rm -rf *.o imgcls