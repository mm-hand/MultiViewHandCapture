CXX ?= g++
SDK := ManusSDK_v3.1.1
TARGET := libmanus_sdk_bridge.so

.PHONY: all clean
all: $(TARGET)

$(TARGET): manus_sdk_bridge.cpp $(SDK)/include/ManusSDK.h $(SDK)/lib/libManusSDK_Integrated.so
	$(CXX) -std=c++17 -O2 -fPIC -shared \
		-I$(SDK)/include manus_sdk_bridge.cpp \
		-L$(SDK)/lib -lManusSDK_Integrated -pthread \
		-Wl,-rpath,'$$ORIGIN/$(SDK)/lib' -o $@

clean:
	rm -f $(TARGET)
