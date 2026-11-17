export VULKAN_SDK=/opt/vulkan-sdk/x86_64
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Use lavapipe (llvmpipe) only
VK_DRIVER_FILES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json 2>/dev/null | head -n1 || true)
export VK_DRIVER_FILES
