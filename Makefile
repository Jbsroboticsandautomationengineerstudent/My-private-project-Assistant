THEOS_PACKAGE_SCHEME = rootless
TARGET = iphone:clang:latest:17.0
ARCHS = arm64 arm64e
INSTALL_TARGET_PROCESSES = Chess

# Optional checksum override when the Hugging Face metadata API is unavailable.
export MAIA79_ONNX_SHA256

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = ChessAssistantPro
ChessAssistantPro_FILES = Tweak.xm
ChessAssistantPro_FRAMEWORKS = Foundation UIKit CoreML QuartzCore
ChessAssistantPro_LIBRARIES = substrate
ChessAssistantPro_CFLAGS = -fobjc-arc -fblocks -I$(CURDIR)/.asset-cache/ort/ios-arm64/onnxruntime.framework/Headers
ChessAssistantPro_CCFLAGS = -std=c++17
ChessAssistantPro_LDFLAGS = -F$(CURDIR)/.asset-cache/ort/ios-arm64 -framework onnxruntime

include $(THEOS_MAKE_PATH)/tweak.mk

.PHONY: assets
assets:
	@python3 prepare_assets.py

# Download before compilation so the ONNX Runtime headers and library exist.
# Theos prefixes layout/ with /var/jb for rootless packages.
before-all::
	@python3 prepare_assets.py
