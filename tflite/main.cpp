
#include <tensorflow/lite/micro/all_ops_resolver.h>

#include "../decoder.cpp"
#include "../encoder.cpp"

int
main()
{
  auto encoder = tflite::GetModel(encoder_tflite);
  auto decoder = tflite::GetModel(decoder_tflite);

  tflite::AllOpsResolver resolver;

  // tflite::MicroInterpreter interpreter;

  return 0;
}
