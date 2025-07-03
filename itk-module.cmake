# the top-level README is used for describing this module, just
# re-used it for documentation here
get_filename_component(MY_CURRENT_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(READ "${MY_CURRENT_DIR}/README.md" DOCUMENTATION)

# itk_module() defines the module dependencies in Metrics
# Metrics depends on ITKCommon and ITKMetricsv4
# By convention those modules outside of ITK are not prefixed with
# ITK.

# define the dependencies of the include module and the tests
itk_module(Metrics
  DEPENDS
    ITKCommon
    ITKMetricsv4
    ITKSmoothing
    ITKTransform
  COMPILE_DEPENDS
    ITKImageSources
    ITKSmoothing
    ITKTestKernel
    ITKTransform
  TEST_DEPENDS
    ITKCommon
    ITKMetricsv4
    ITKOptimizersv4
    ITKSmoothing
    ITKTransform
  DESCRIPTION
    "${DOCUMENTATION}"
  EXCLUDE_FROM_DEFAULT
  ENABLE_SHARED
)
