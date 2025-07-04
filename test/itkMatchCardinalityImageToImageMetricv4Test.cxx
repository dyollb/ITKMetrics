/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "itkMatchCardinalityImageToImageMetricv4.h"
#include "itkTranslationTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkTestingMacros.h"

int
itkMatchCardinalityImageToImageMetricv4Test(int argc, char * argv[])
{
  constexpr unsigned int ImageDimension = 2;
  using PixelType = float; // required because of concept checking in ImageToImageMetricv4
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using MetricType = itk::MatchCardinalityImageToImageMetricv4<ImageType, ImageType>;

  // Create two simple test images with labels
  auto fixedImage = ImageType::New();
  auto movingImage = ImageType::New();

  // Set up image properties
  ImageType::SizeType size;
  size.Fill(10);

  ImageType::IndexType index;
  index.Fill(0);

  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(index);

  fixedImage->SetRegions(region);
  fixedImage->Allocate();

  movingImage->SetRegions(region);
  movingImage->Allocate();

  // Fill images with simple label patterns
  itk::ImageRegionIteratorWithIndex<ImageType> fixedIt(fixedImage, region);
  itk::ImageRegionIteratorWithIndex<ImageType> movingIt(movingImage, region);

  for (fixedIt.GoToBegin(), movingIt.GoToBegin(); !fixedIt.IsAtEnd(); ++fixedIt, ++movingIt)
  {
    ImageType::IndexType currentIndex = fixedIt.GetIndex();

    // Create a simple checkerboard pattern for fixed image
    PixelType fixedLabel = ((currentIndex[0] + currentIndex[1]) % 2 == 0) ? 1 : 2;
    fixedIt.Set(fixedLabel);

    // Create a slightly shifted pattern for moving image
    PixelType movingLabel = ((currentIndex[0] + currentIndex[1] + 1) % 2 == 0) ? 1 : 2;
    movingIt.Set(movingLabel);
  }

  // Create the metric
  auto metric = MetricType::New();

  ITK_EXERCISE_BASIC_OBJECT_METHODS(metric, MatchCardinalityImageToImageMetricv4, ImageToImageMetricv4);

  // Set up the metric
  metric->SetFixedImage(fixedImage);
  metric->SetMovingImage(movingImage);

  // Set up transform
  using TransformType = itk::TranslationTransform<double, ImageDimension>;
  auto transform = TransformType::New();
  transform->SetIdentity();
  metric->SetMovingTransform(transform);

  // Set up interpolator
  using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double>;
  auto interpolator = InterpolatorType::New();
  metric->SetMovingInterpolator(interpolator);

  // Initialize the metric
  try
  {
    metric->Initialize();
  }
  catch (const itk::ExceptionObject & e)
  {
    std::cerr << "Exception caught during metric initialization: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Test getting value
  MetricType::MeasureType value;
  try
  {
    value = metric->GetValue();
    std::cout << "Metric value: " << value << std::endl;
  }
  catch (const itk::ExceptionObject & e)
  {
    std::cerr << "Exception caught during GetValue: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Test getting derivative
  MetricType::DerivativeType derivative;
  try
  {
    metric->GetDerivative(derivative);
    std::cout << "Derivative size: " << derivative.Size() << std::endl;
  }
  catch (const itk::ExceptionObject & e)
  {
    std::cerr << "Exception caught during GetDerivative: " << e << std::endl;
    return EXIT_FAILURE;
  }

  // Test getting value and derivative
  try
  {
    metric->GetValueAndDerivative(value, derivative);
    std::cout << "Value and derivative computed successfully" << std::endl;
    std::cout << "Value: " << value << std::endl;
    std::cout << "Derivative size: " << derivative.Size() << std::endl;
  }
  catch (const itk::ExceptionObject & e)
  {
    std::cerr << "Exception caught during GetValueAndDerivative: " << e << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Test completed successfully!" << std::endl;
  return EXIT_SUCCESS;
}
