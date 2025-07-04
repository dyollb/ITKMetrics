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
#ifndef itkMatchCardinalityImageToImageMetricv4_h
#define itkMatchCardinalityImageToImageMetricv4_h

#include "itkImageToImageMetricv4.h"
#include "itkMatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader.h"
#include "itkLabeledImageToImageMetricTraitsv4.h"

namespace itk
{

/** \class MatchCardinalityImageToImageMetricv4
 *
 *  \brief Class implementing a match cardinality metric for labeled images.
 *
 *  This metric computes the match cardinality between two labeled images.
 *  The metric is defined as:
 *  - If label(fixed_image) == label(moving_image), then the error is 0
 *  - Otherwise, the error is 1
 *
 *  The metric value is the mean error across all valid pixels.
 *
 *  **Important Notes:**
 *  - This metric expects labeled images as input (labels 0, 1, 2, 3, etc.)
 *  - The gradient computation is handled specially for labeled images
 *  - Uses LabeledImageToImageMetricTraitsv4 for proper gradient computation
 *
 *  **Gradient Computation:**
 *  For labeled images, the gradient is computed by treating each label as a
 *  separate binary mask and computing the gradient of each mask. The final
 *  gradient is assembled from all label gradients.
 *
 *  See MatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader::ProcessPoint
 *  for algorithm implementation.
 *
 * \ingroup Metrics
 */
template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage = TFixedImage,
          typename TInternalComputationValueType = double,
          typename TMetricTraits =
            LabeledImageToImageMetricTraitsv4<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType>>
class ITK_TEMPLATE_EXPORT MatchCardinalityImageToImageMetricv4
  : public ImageToImageMetricv4<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MatchCardinalityImageToImageMetricv4);

  /** Standard class type aliases. */
  using Self = MatchCardinalityImageToImageMetricv4;
  using Superclass =
    ImageToImageMetricv4<TFixedImage, TMovingImage, TVirtualImage, TInternalComputationValueType, TMetricTraits>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MatchCardinalityImageToImageMetricv4, ImageToImageMetricv4);

  using typename Superclass::DerivativeType;

  using typename Superclass::FixedImagePointType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::FixedImageGradientType;

  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageGradientType;

  using typename Superclass::MovingTransformType;
  using typename Superclass::JacobianType;
  using VirtualImageType = typename Superclass::VirtualImageType;
  using typename Superclass::VirtualIndexType;
  using typename Superclass::VirtualPointType;
  using typename Superclass::VirtualPointSetType;

  /* Image dimension accessors */
  static constexpr typename TVirtualImage::ImageDimensionType VirtualImageDimension = TVirtualImage::ImageDimension;
  static constexpr typename TFixedImage::ImageDimensionType   FixedImageDimension = TFixedImage::ImageDimension;
  static constexpr typename TMovingImage::ImageDimensionType  MovingImageDimension = TMovingImage::ImageDimension;

protected:
  MatchCardinalityImageToImageMetricv4();
  ~MatchCardinalityImageToImageMetricv4() override = default;

  friend class MatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader<
    ThreadedImageRegionPartitioner<Superclass::VirtualImageDimension>,
    Superclass,
    Self>;
  friend class MatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader<ThreadedIndexedContainerPartitioner,
                                                                                 Superclass,
                                                                                 Self>;
  using MatchCardinalityDenseGetValueAndDerivativeThreaderType =
    MatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader<
      ThreadedImageRegionPartitioner<Superclass::VirtualImageDimension>,
      Superclass,
      Self>;
  using MatchCardinalitySparseGetValueAndDerivativeThreaderType =
    MatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader<ThreadedIndexedContainerPartitioner,
                                                                      Superclass,
                                                                      Self>;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMatchCardinalityImageToImageMetricv4.hxx"
#endif

#endif
