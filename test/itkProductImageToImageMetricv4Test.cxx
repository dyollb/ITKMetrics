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

#include "itkProductImageToImageMetricv4.h"

#include "itkCommand.h"
#include "itkPointSet.h"
#include "itkTestingMacros.h"
#include "itkTranslationTransform.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMath.h"
#include <cmath>

/* Helper function to create test images with known patterns */
template <typename TImageType>
typename TImageType::Pointer
CreateTestImage(typename TImageType::SizeType size, double baseValue = 100.0, double multiplier = 1.0)
{
  using ImageType = TImageType;
  using IndexType = typename ImageType::IndexType;

  auto image = ImageType::New();

  IndexType startIndex;
  startIndex.Fill(0);

  typename ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(startIndex);

  typename ImageType::SpacingType spacing;
  spacing.Fill(1.0);

  typename ImageType::PointType origin;
  origin.Fill(0.0);

  typename ImageType::DirectionType direction;
  direction.SetIdentity();

  image->SetRegions(region);
  image->SetSpacing(spacing);
  image->SetOrigin(origin);
  image->SetDirection(direction);
  image->Allocate();

  // Fill with a simple linear pattern
  itk::ImageRegionIteratorWithIndex<ImageType> iterator(image, region);
  iterator.GoToBegin();

  while (!iterator.IsAtEnd())
  {
    IndexType index = iterator.GetIndex();
    double    value = baseValue + multiplier * index[0];
    iterator.Set(static_cast<typename ImageType::PixelType>(value));
    ++iterator;
  }

  return image;
}

int
itkProductImageToImageMetricv4Test(int argc, char * argv[])
{
  constexpr unsigned int Dimension = 2;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;

  // Create test image parameters
  constexpr unsigned int imageSize = 10;
  ImageType::SizeType    size;
  size.Fill(imageSize);

  // Create test images
  std::cout << "Creating test images..." << std::endl;
  auto fixedImage = CreateTestImage<ImageType>(size, 100.0, 1.0); // Values from 100 to 118
  auto movingImage = CreateTestImage<ImageType>(size, 50.0, 2.0); // Values from 50 to 86

  // Create transforms
  using TransformType = itk::TranslationTransform<double, Dimension>;
  auto fixedTransform = TransformType::New();
  auto movingTransform = TransformType::New();

  fixedTransform->SetIdentity();
  movingTransform->SetIdentity();

  // Create the product sum metric
  using MetricType = itk::ProductImageToImageMetricv4<ImageType, ImageType, ImageType>;
  auto metric = MetricType::New();

  // Test basic object methods
  ITK_EXERCISE_BASIC_OBJECT_METHODS(metric, ProductImageToImageMetricv4, ImageToImageMetricv4);

  // Set up the metric
  std::cout << "Setting up metric..." << std::endl;
  metric->SetFixedImage(fixedImage);
  metric->SetMovingImage(movingImage);
  metric->SetFixedTransform(fixedTransform);
  metric->SetMovingTransform(movingTransform);

  int result = EXIT_SUCCESS;

  // Test initialization and computation
  try
  {
    std::cout << "Initializing metric..." << std::endl;
    metric->Initialize();

    std::cout << "Computing value and derivative..." << std::endl;
    MetricType::MeasureType    value;
    MetricType::DerivativeType derivative;
    metric->GetValueAndDerivative(value, derivative);

    std::cout << "Metric value: " << value << std::endl;
    std::cout << "Derivative magnitude: " << derivative.magnitude() << std::endl;
    std::cout << "Number of valid points: " << metric->GetNumberOfValidPoints() << std::endl;

    // Test that we get a reasonable result
    if (std::isnan(value) || std::isinf(value))
    {
      std::cerr << "ERROR: Metric produced invalid value: " << value << std::endl;
      result = EXIT_FAILURE;
    }

    // For our test images, we expect a positive value since both images have positive values
    if (value <= 0)
    {
      std::cerr << "WARNING: Expected positive value for product sum of positive images, got: " << value << std::endl;
    }

    // Test with different thread counts
    std::cout << "\nTesting with different thread counts..." << std::endl;

    metric->SetMaximumNumberOfWorkUnits(1);
    MetricType::MeasureType    value1;
    MetricType::DerivativeType derivative1;
    metric->GetValueAndDerivative(value1, derivative1);

    metric->SetMaximumNumberOfWorkUnits(4);
    MetricType::MeasureType    value2;
    MetricType::DerivativeType derivative2;
    metric->GetValueAndDerivative(value2, derivative2);

    const double tolerance = 1e-10;
    if (std::abs(value1 - value2) > tolerance)
    {
      std::cerr << "ERROR: Different values with different thread counts:" << std::endl;
      std::cerr << "  1 thread:  " << value1 << std::endl;
      std::cerr << "  4 threads: " << value2 << std::endl;
      result = EXIT_FAILURE;
    }

    auto derivativeDifference = derivative1 - derivative2;
    if (derivativeDifference.magnitude() > tolerance)
    {
      std::cerr << "ERROR: Different derivatives with different thread counts" << std::endl;
      std::cerr << "  Difference magnitude: " << derivativeDifference.magnitude() << std::endl;
      result = EXIT_FAILURE;
    }

    // Test with constant images to verify computation
    std::cout << "\nTesting with constant images..." << std::endl;
    auto constantFixed = CreateTestImage<ImageType>(size, 5.0, 0.0);  // All pixels = 5
    auto constantMoving = CreateTestImage<ImageType>(size, 3.0, 0.0); // All pixels = 3

    metric->SetFixedImage(constantFixed);
    metric->SetMovingImage(constantMoving);
    metric->Initialize();

    MetricType::MeasureType    constantValue;
    MetricType::DerivativeType constantDerivative;
    metric->GetValueAndDerivative(constantValue, constantDerivative);

    // Expected value should be 5 * 3 = 15
    // The metric framework normalizes the sum
    double expectedValue = 5.0 * 3.0;
    std::cout << "Constant images - Expected: " << expectedValue << ", Got: " << constantValue << std::endl;

    if (std::abs(constantValue - expectedValue) > 1e-6)
    {
      std::cerr << "ERROR: Constant images test failed" << std::endl;
      std::cerr << "  Expected: " << expectedValue << std::endl;
      std::cerr << "  Got: " << constantValue << std::endl;
      result = EXIT_FAILURE;
    }

    // Test derivative computation with constant and gradient images
    std::cout << "\nTesting derivative with constant and gradient images..." << std::endl;
    auto constantImage = CreateTestImage<ImageType>(size, 4.0, 0.0); // All pixels = 4
    auto gradientImage = CreateTestImage<ImageType>(size, 0.0, 1.0); // Gradient along x: f(x,y) = x + y

    metric->SetFixedImage(constantImage);
    metric->SetMovingImage(gradientImage);
    metric->Initialize();

    MetricType::MeasureType    gradientValue;
    MetricType::DerivativeType gradientDerivative;
    metric->GetValueAndDerivative(gradientValue, gradientDerivative);

    std::cout << "Gradient test - Value: " << gradientValue
              << ", Derivative magnitude: " << gradientDerivative.magnitude() << std::endl;

    // For product metric: d/dx(fixed * moving) = fixed * d(moving)/dx
    // With constant fixed image (4.0) and gradient moving image (x + y),
    // the derivative should be non-zero and proportional to the constant value
    if (gradientDerivative.magnitude() < 1e-6)
    {
      std::cerr << "ERROR: Expected non-zero derivative for gradient image, got magnitude: "
                << gradientDerivative.magnitude() << std::endl;
      result = EXIT_FAILURE;
    }

    // Test with swapped images (gradient as fixed, constant as moving)
    // Note: In ITK metrics, derivatives are computed w.r.t. moving image transform parameters
    // When constant image is moving, derivative should be zero since moving image is uniform
    metric->SetFixedImage(gradientImage);
    metric->SetMovingImage(constantImage);
    metric->Initialize();

    MetricType::MeasureType    swappedValue;
    MetricType::DerivativeType swappedDerivative;
    metric->GetValueAndDerivative(swappedValue, swappedDerivative);

    // Values should be the same (product is commutative)
    if (std::abs(gradientValue - swappedValue) > 1e-6)
    {
      std::cerr << "ERROR: Product should be commutative, got values: " << gradientValue << " vs " << swappedValue
                << std::endl;
      result = EXIT_FAILURE;
    }

    // When moving image is constant, derivative should be zero or very small
    // This is expected behavior since d/dx(gradient * constant) = gradient * d(constant)/dx = gradient * 0 = 0
    std::cout << "Swapped test - Value: " << swappedValue << ", Derivative magnitude: " << swappedDerivative.magnitude()
              << std::endl;
    if (swappedDerivative.magnitude() > 1e-6)
    {
      std::cerr << "WARNING: Expected near-zero derivative when moving image is constant, got magnitude: "
                << swappedDerivative.magnitude() << std::endl;
    }
  }
  catch (const itk::ExceptionObject & exc)
  {
    std::cerr << "ERROR: Exception during metric computation: " << exc << std::endl;
    result = EXIT_FAILURE;
  }

  if (result == EXIT_SUCCESS)
  {
    std::cout << "\nTest passed successfully!" << std::endl;
  }
  else
  {
    std::cerr << "\nTest failed." << std::endl;
  }

  return result;
}
