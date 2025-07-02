set(MetricsTests
  itkProductSumImageToImageMetricv4Test.cxx
  )

CreateTestDriver(Metrics "${Metrics-Test_LIBRARIES}" "${MetricsTests}")


itk_add_test(NAME itkProductSumImageToImageMetricv4Test
  COMMAND MetricsTestDriver itkProductSumImageToImageMetricv4Test
  )
