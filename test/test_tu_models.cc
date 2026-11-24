#include <gtest/gtest.h>

void checkTuModelsA();
void checkTuModelsB();

// Both orders, and repeated, so that a template instantiation mixed up between the units would show whichever way it
// resolves.
TEST(KernelNames, SameNamedModelsInTwoTranslationUnits) {
  checkTuModelsA();
  checkTuModelsB();
  checkTuModelsB();
  checkTuModelsA();
}
