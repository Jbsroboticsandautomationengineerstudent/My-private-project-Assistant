TARGET = iphone:clang:latest:17.0
# The official ONNX Runtime iOS framework contains arm64 only. arm64e
# devices can run the arm64 slice; linking a separate arm64e slice fails.
ARCHS = arm64
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
ChessAssistantPro_LDFLAGS = -F$(CURDIR)/.asset-cache/ort/ios-arm64 -framework onnxruntime -Xlinker -rpath -Xlinker @executable_path/Frameworks

include $(THEOS_MAKE_PATH)/tweak.mk

.PHONY: assets
assets:
	@python3 prepare_assets.py
	@bash build_stockfish19.sh
	@bash build_triumviratus.sh

# Download before compilation so the ONNX Runtime headers and library exist.
before-all::
	@python3 prepare_assets.py
	@bash build_stockfish19.sh
	@bash build_triumviratus.sh

# After linking, rewrite the dylib's ORT reference from the build-time
# @rpath to @executable_path/Frameworks/ so it resolves inside the app
# bundle when sideloaded (no jailbreak runtime paths).
after-stage::
	@install_name_tool -change @rpath/onnxruntime.framework/onnxruntime \
		@executable_path/Frameworks/onnxruntime.framework/onnxruntime \
		$(THEOS_STAGING_DIR)/Library/MobileSubstrate/DynamicLibraries/ChessAssistantPro.dylib
	@echo "Rewritten ORT load path to @executable_path/Frameworks/"
