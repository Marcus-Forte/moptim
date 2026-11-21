#include <gtest/gtest.h>

void checkTuModelsA();
void checkTuModelsB();

// Both orders, and repeated, so that a kernel mixed up by name would show whichever way it resolves.
TEST(KernelNames, SameNamedModelsInTwoTranslationUnits) {
  checkTuModelsA();
  checkTuModelsB();
  checkTuModelsB();
  checkTuModelsA();
}
