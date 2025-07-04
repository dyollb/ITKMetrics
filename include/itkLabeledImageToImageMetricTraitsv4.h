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
#ifndef itkLabeledImageToImageMetricTraitsv4_h
#define itkLabeledImageToImageMetricTraitsv4_h

#include "itkImage.h"
#include "itkVector.h"
#include "itkVariableLengthVector.h"
#include "itkCovariantVector.h"
#include "itkCentralDifferenceImageFunction.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkObjectToObjectMetricBase.h"
#include "itkVectorImage.h"
#include "itkComposeImageFilter.h"
#include "itkMultiLabelImageGradientCalculator.h"
#include "itkMultiLabelImageGradientFilter.h"

namespace itk
{

/** \class LabeledImageToImageMetricTraitsv4
 * \brief A traits class for labeled images in ImageToImageMetricv4
 *
 * This class provides type information for ImageToImageMetricv4 classes
 * that work with labeled images. It handles the special case where
 * gradients need to be computed for each label separately as binary masks.
 *
 * For labeled images, the gradient is computed as follows:
 * - For each label L, create a binary mask where pixels with label L = 1, others = 0
 * - Compute gradient of each binary mask
 * - The final gradient is a vector containing all label gradients
 *
 * \ingroup Metrics
 */
template <typename TFixedImageType, typename TMovingImageType, typename TVirtualImageType, typename TCoordRep = double>
class LabeledImageToImageMetricTraitsv4
{
public:
  /** Standard class type aliases. */
  using Self = LabeledImageToImageMetricTraitsv4;

  using FixedImageType = TFixedImageType;
  using MovingImageType = TMovingImageType;
  using VirtualImageType = TVirtualImageType;

  using FixedImagePixelType = typename FixedImageType::PixelType;
  using MovingImagePixelType = typename MovingImageType::PixelType;

  using CoordinateRepresentationType = TCoordRep;

  /* Image dimension accessors */
  using ImageDimensionType = unsigned int;
  static constexpr ImageDimensionType FixedImageDimension = FixedImageType::ImageDimension;
  static constexpr ImageDimensionType MovingImageDimension = MovingImageType::ImageDimension;
  static constexpr ImageDimensionType VirtualImageDimension = VirtualImageType::ImageDimension;

  // For labeled images, we need to support gradients for multiple labels
  // Use VariableLengthVector for dynamic number of components based on actual labels
  using FixedImageGradientType = VariableLengthVector<CoordinateRepresentationType>;
  using MovingImageGradientType = VariableLengthVector<CoordinateRepresentationType>;
  using VirtualImageGradientType = VariableLengthVector<CoordinateRepresentationType>;

  using FixedImageComponentGradientType = FixedImageGradientType;
  using MovingImageComponentGradientType = MovingImageGradientType;
  using VirtualImageComponentGradientType = VirtualImageGradientType;

  using FixedImageGradientConvertType = DefaultConvertPixelTraits<FixedImageGradientType>;
  using MovingImageGradientConvertType = DefaultConvertPixelTraits<MovingImageGradientType>;

  /** Type of the filter used to calculate the gradients. */
  // For labeled images, gradients should always be floating point regardless of input type
  using FixedRealType = float;
  using MovingRealType = float;

  // Gradient pixel types for compatibility with ITK's metric system
  using FixedGradientPixelType = VariableLengthVector<FixedRealType>;
  using MovingGradientPixelType = VariableLengthVector<MovingRealType>;

  // For labeled images, we use a vector image to store gradients for all labels
  using FixedImageGradientImageType = VectorImage<FixedRealType, Self::FixedImageDimension>;
  using MovingImageGradientImageType = VectorImage<MovingRealType, Self::MovingImageDimension>;

  using FixedImageGradientFilterType = ImageToImageFilter<FixedImageType, FixedImageGradientImageType>;
  using MovingImageGradientFilterType = ImageToImageFilter<MovingImageType, MovingImageGradientImageType>;

  /** Default image gradient filter types for labeled images */
  using DefaultFixedImageGradientFilter = MultiLabelImageGradientFilter<FixedImageType, FixedImageGradientImageType>;
  using DefaultMovingImageGradientFilter = MultiLabelImageGradientFilter<MovingImageType, MovingImageGradientImageType>;

  /** Image gradient calculator types for labeled images */
  using FixedImageGradientCalculatorType =
    ImageFunction<FixedImageType, FixedImageGradientType, CoordinateRepresentationType>;
  using MovingImageGradientCalculatorType =
    ImageFunction<MovingImageType, MovingImageGradientType, CoordinateRepresentationType>;

  using DefaultFixedImageGradientCalculator =
    MultiLabelImageGradientCalculator<FixedImageType, CoordinateRepresentationType>;
  using DefaultMovingImageGradientCalculator =
    MultiLabelImageGradientCalculator<MovingImageType, CoordinateRepresentationType>;

  /** Note: This traits class is designed for labeled (integer) images,
   * but the gradient computations use floating point arithmetic. */
};

} // end namespace itk

#endif
