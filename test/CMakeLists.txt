itk_module_test()

set(MetricsTests
  itkProductImageToImageMetricv4Test.cxx
  itkMatchCardinalityImageToImageMetricv4Test.cxx
  )

CreateTestDriver(Metrics "${Metrics-Test_LIBRARIES}" "${MetricsTests}")

itk_add_test(NAME itkProductImageToImageMetricv4Test
  COMMAND MetricsTestDriver itkProductImageToImageMetricv4Test
  )
