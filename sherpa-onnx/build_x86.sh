TARGET_SOC="pc"
set -e

dir=build
mkdir -p $dir
cd $dir
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTARGET_SOC=${TARGET_SOC} \
  -DTARGET_ARCH=x86 \
  -DRK3588=false \
  -DSPEAK_ID_NPU=OFF \
  ..
make -j4
# make install/strip
cp bin/vad_stt_tts ../
