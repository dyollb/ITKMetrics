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
#ifndef itkProductImageToImageMetricv4GetValueAndDerivativeThreader_hxx
#define itkProductImageToImageMetricv4GetValueAndDerivativeThreader_hxx

#include "itkDefaultConvertPixelTraits.h"

namespace itk
{

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TProductMetric>
bool
ProductImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric, TProductMetric>::
  ProcessPoint(const VirtualIndexType &,
               const VirtualPointType & virtualPoint,
               const FixedImagePointType &,
               const FixedImagePixelType & fixedImageValue,
               const FixedImageGradientType &,
               const MovingImagePointType &,
               const MovingImagePixelType &    movingImageValue,
               const MovingImageGradientType & movingImageGradient,
               MeasureType &                   metricValueReturn,
               DerivativeType &                localDerivativeReturn,
               const ThreadIdType              threadId) const
{
  /** Compute the product of fixed and moving image values. */
  const unsigned int nComponents = NumericTraits<FixedImagePixelType>::GetLength(fixedImageValue);
  metricValueReturn = NumericTraits<MeasureType>::ZeroValue();

  for (unsigned int nc = 0; nc < nComponents; ++nc)
  {
    MeasureType fixedC = DefaultConvertPixelTraits<FixedImagePixelType>::GetNthComponent(nc, fixedImageValue);
    MeasureType movingC = DefaultConvertPixelTraits<MovingImagePixelType>::GetNthComponent(nc, movingImageValue);
    metricValueReturn += fixedC * movingC;
  }

  if (!this->GetComputeDerivative())
  {
    return true;
  }

  /* Use a pre-allocated jacobian object for efficiency */
  using JacobianReferenceType = typename TImageToImageMetric::JacobianType &;
  JacobianReferenceType jacobian = this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobian;
  JacobianReferenceType jacobianPositional =
    this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobianPositional;

  /** For dense transforms, this returns identity */
  this->m_Associate->GetMovingTransform()->ComputeJacobianWithRespectToParametersCachedTemporaries(
    virtualPoint, jacobian, jacobianPositional);

  for (unsigned int par = 0; par < this->GetCachedNumberOfLocalParameters(); ++par)
  {
    localDerivativeReturn[par] = NumericTraits<DerivativeValueType>::ZeroValue();
    for (unsigned int nc = 0; nc < nComponents; ++nc)
    {
      MeasureType fixedValue = DefaultConvertPixelTraits<FixedImagePixelType>::GetNthComponent(nc, fixedImageValue);
      for (SizeValueType dim = 0; dim < ImageToImageMetricv4Type::MovingImageDimension; ++dim)
      {
        // Derivative of (fixed * moving) with respect to moving transform parameters
        // d/dp(fixed * moving) = fixed * d(moving)/dp
        // where d(moving)/dp = gradient * jacobian
        localDerivativeReturn[par] += fixedValue * jacobian(dim, par) *
                                      DefaultConvertPixelTraits<MovingImageGradientType>::GetNthComponent(
                                        ImageToImageMetricv4Type::FixedImageDimension * nc + dim, movingImageGradient);
      }
    }
  }
  return true;
}

} // end namespace itk

#endif
