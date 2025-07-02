set(MetricsTests
  #itkEntropyImageToImageMetricv4Test.cxx
  itkProductSumImageToImageMetricv4Test.cxx
  )

CreateTestDriver(Metrics "${Metrics-Test_LIBRARIES}" "${MetricsTests}")


#itk_add_test(NAME itkEntropyImageToImageMetricv4Test
#  COMMAND MetricsTestDriver itkEntropyImageToImageMetricv4Test
#  )

itk_add_test(NAME itkProductSumImageToImageMetricv4Test
  COMMAND MetricsTestDriver itkProductSumImageToImageMetricv4Test
  )
