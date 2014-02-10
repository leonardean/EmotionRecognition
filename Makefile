# Paths
#OPENCV_PATH=/usr/local
OPENCV_PATH=/opt/local

# Programs
CC=
CXX=g++

# Flags
ARCH_FLAGS=-arch x86_64
CFLAGS=-Wextra -Wall -pedantic-errors $(ARCH_FLAGS) -O3 -fopenmp
LDFLAGS=$(ARCH_FLAGS)
DEFINES=
INCLUDES=-I$(OPENCV_PATH)/include/opencv -Isrc/lib
LIBRARIES=-L$(OPENCV_PATH)/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lgomp

# Files which require compiling
SOURCE_FILES=\
	src/lib/IO.cc\
	src/lib/PDM.cc\
	src/lib/Patch.cc\
	src/lib/CLM.cc\
	src/lib/FDet.cc\
	src/lib/PAW.cc\
	src/lib/FCheck.cc\
	src/lib/Tracker.cc\
	src/lib/svm.cc

# Source files which contain a int main(..) function
SOURCE_FILES_WITH_MAIN=src/exe/emotion_recognizer.cc

# End Configuration
SOURCE_OBJECTS=$(patsubst %.cc,%.o,$(SOURCE_FILES))

ALL_OBJECTS=\
	$(SOURCE_OBJECTS) \
	$(patsubst %.cc,%.o,$(SOURCE_FILES_WITH_MAIN))

DEPENDENCY_FILES=\
	$(patsubst %.o,%.d,$(ALL_OBJECTS)) 

all: bin/emotion_recognizer

%.o: %.cc Makefile
	@# Make dependecy file
	$(CXX) -MM -MT $@ -MF $(patsubst %.cc,%.d,$<) $(CFLAGS) $(DEFINES) $(INCLUDES) $<
	@# Compile
	$(CXX) $(CFLAGS) $(DEFINES) $(INCLUDES) -c -o $@ $< 

-include $(DEPENDENCY_FILES)

bin/emotion_recognizer: $(ALL_OBJECTS)
	$(CXX) $(LDFLAGS) $(LIBRARIES) -o $@ $(ALL_OBJECTS)

.PHONY: clean
clean:
	@echo "Cleaning"
	@for pattern in '*~' '*.o' '*.d' ; do \
		find . -name "$$pattern" | xargs rm ; \
	done
