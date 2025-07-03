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
#ifndef itkProductImageToImageMetricv4GetValueAndDerivativeThreader_h
#define itkProductImageToImageMetricv4GetValueAndDerivativeThreader_h

#include "itkImageToImageMetricv4GetValueAndDerivativeThreader.h"

namespace itk
{

/** \class ProductImageToImageMetricv4GetValueAndDerivativeThreader
 * \brief Processes points for ProductImageToImageMetricv4 \c
 * GetValueAndDerivative.
 *
 * \ingroup Metrics
 */
template <typename TDomainPartitioner, typename TImageToImageMetric, typename TProductMetric>
class ITK_TEMPLATE_EXPORT ProductImageToImageMetricv4GetValueAndDerivativeThreader
  : public ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ProductImageToImageMetricv4GetValueAndDerivativeThreader);

  /** Standard class type aliases. */
  using Self = ProductImageToImageMetricv4GetValueAndDerivativeThreader;
  using Superclass = ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkTypeMacro(ProductImageToImageMetricv4GetValueAndDerivativeThreader,
               ImageToImageMetricv4GetValueAndDerivativeThreader);

  itkNewMacro(Self);

  using typename Superclass::DomainType;
  using typename Superclass::AssociateType;

  using ImageToImageMetricv4Type = typename Superclass::ImageToImageMetricv4Type;
  using typename Superclass::VirtualPointType;
  using typename Superclass::VirtualIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::FixedImageGradientType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageGradientType;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::NumberOfParametersType;

protected:
  ProductImageToImageMetricv4GetValueAndDerivativeThreader() = default;

  /** This function computes the local voxel-wise contribution of
   *  the metric to the global integral of the metric/derivative.
   */
  bool
  ProcessPoint(const VirtualIndexType &        virtualIndex,
               const VirtualPointType &        virtualPoint,
               const FixedImagePointType &     mappedFixedPoint,
               const FixedImagePixelType &     fixedImageValue,
               const FixedImageGradientType &  mappedFixedImageGradient,
               const MovingImagePointType &    mappedMovingPoint,
               const MovingImagePixelType &    movingImageValue,
               const MovingImageGradientType & movingImageGradient,
               MeasureType &                   metricValueReturn,
               DerivativeType &                localDerivativeReturn,
               const ThreadIdType              threadId) const override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkProductImageToImageMetricv4GetValueAndDerivativeThreader.hxx"
#endif

#endif
