set -e

TARGET_SOC="rk3588"

GCC_COMPILER=/media/lee/data_new/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

dir=build-aarch64-linux-gnu
mkdir -p $dir
cd $dir

if [ ! -f alsa-lib/src/.libs/libasound.so ]; then
  echo "Start to cross-compile alsa-lib"
  if [ ! -d alsa-lib ]; then
    git clone --depth 1 https://github.com/alsa-project/alsa-lib
  fi

  
  pushd alsa-lib
  CC=${GCC_COMPILER}-gcc ./gitcompile --host=aarch64-linux-gnu
  popd
  echo "Finish cross-compiling alsa-lib"
fi
export CPLUS_INCLUDE_PATH=$PWD/alsa-lib/include:$CPLUS_INCLUDE_PATH
export SHERPA_ONNX_ALSA_LIB_DIR=$PWD/alsa-lib/src/.libs


# build
 
cmake \
  -DRK3588=true \
  -DTARGET_SOC=${TARGET_SOC} \
  -DTARGET_ARCH=aarch \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DCMAKE_TOOLCHAIN_FILE=aarch64-linux-gnu.toolchain.cmake \
  -DSPEAK_ID_NPU=ON \
  ..

make -j4
make install/strip
cd -
adb push ./build-aarch64-linux-gnu/install/ /userdata/sherpa-onnx/