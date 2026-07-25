set -ex

CANN_PATH=/usr/local/Ascend/cann-9.0.0
VENDOR_DIR=${CANN_PATH}/opp/vendors/custom_nz_pass/custom_fusion_passes

g++ -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -o libnz_weight_pass.so nz_weight_pass.cpp \
    -I${CANN_PATH}/aarch64-linux/include \
    -I${CANN_PATH}/opp/built-in/op_graph/inc \
    -L${CANN_PATH}/aarch64-linux/lib64 \
    -L${CANN_PATH}/opp/built-in/op_graph/lib \
    -lgraph -lregister

mkdir -p ${VENDOR_DIR}
cp libnz_weight_pass.so ${VENDOR_DIR}/
chmod 755 ${VENDOR_DIR}/libnz_weight_pass.so
