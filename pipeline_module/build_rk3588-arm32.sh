set -e

TARGET_SOC="rk3588"


dir=build-arm32-linux-gnu
mkdir -p $dir
cd $dir

if [ ! -f alsa-lib/src/.libs/libasound.so ]; then
  echo "Start to cross-compile alsa-lib"
  if [ ! -d alsa-lib ]; then
    git clone --depth 1 https://github.com/alsa-project/alsa-lib
  fi
  # If it shows:
  #  ./gitcompile: line 79: libtoolize: command not found
  # Please use:
  #  sudo apt-get install libtool m4 automake
  #
  pushd alsa-lib
  CC=arm-linux-gnueabihf-gcc ./gitcompile --host=arm-linux-gnueabihf
  popd
  echo "Finish cross-compiling alsa-lib"
fi
export CPLUS_INCLUDE_PATH=$PWD/alsa-lib/include:$CPLUS_INCLUDE_PATH
export SHERPA_ONNX_ALSA_LIB_DIR=$PWD/alsa-lib/src/.libs


# build
 
cmake \
  -DRK3588=true \
  -DTARGET_SOC=${TARGET_SOC} \
  -DTARGET_ARCH=arm32 \
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
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm-linux-gnueabihf.toolchain.cmake  \
  ..

make -j4
make install/strip
cd -
# scp push ./build-arm32-linux-gnu/install/lib/ root@192.168.50.159:/bin/speaker/lib/
