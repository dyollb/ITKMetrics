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
#ifndef itkMatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader_hxx
#define itkMatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader_hxx

#include "itkDefaultConvertPixelTraits.h"

namespace itk
{

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TMatchCardinalityMetric>
unsigned int
MatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader<
  TDomainPartitioner,
  TImageToImageMetric,
  TMatchCardinalityMetric>::GetLabelIndex(const MovingImagePixelType &    label,
                                          const MovingImageGradientType & gradient) const
{
  // For labeled images, we need to determine which label corresponds to the current pixel
  // This is a simplified approach - in practice, you might want to maintain a label map
  // For now, we'll use the label value directly as the index (assuming labels are 0, 1, 2, ...)

  // Find the non-zero gradient component to determine the active label
  constexpr unsigned int ImageDimension = ImageToImageMetricv4Type::MovingImageDimension;
  constexpr unsigned int MaxLabels = 256; // This should match the traits class

  // The gradient vector contains gradients for all labels
  // Find which label gradient is active (non-zero)
  for (unsigned int labelIdx = 0; labelIdx < MaxLabels; ++labelIdx)
  {
    bool hasNonZeroGradient = false;
    for (unsigned int dim = 0; dim < ImageDimension; ++dim)
    {
      unsigned int gradientIdx = labelIdx * ImageDimension + dim;
      if (gradientIdx < gradient.Size() && std::abs(gradient[gradientIdx]) > 1e-12)
      {
        hasNonZeroGradient = true;
        break;
      }
    }
    if (hasNonZeroGradient)
    {
      return labelIdx;
    }
  }

  // If no non-zero gradient found, return the label value directly
  return static_cast<unsigned int>(label);
}

template <typename TDomainPartitioner, typename TImageToImageMetric, typename TMatchCardinalityMetric>
bool
MatchCardinalityImageToImageMetricv4GetValueAndDerivativeThreader<
  TDomainPartitioner,
  TImageToImageMetric,
  TMatchCardinalityMetric>::ProcessPoint(const VirtualIndexType &,
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
  /** Compute the match cardinality metric */
  // If labels match, error is 0, otherwise error is 1
  if (fixedImageValue == movingImageValue)
  {
    metricValueReturn = 0.0;
  }
  else
  {
    metricValueReturn = 1.0;
  }

  if (!this->GetComputeDerivative())
  {
    return true;
  }

  /** Compute the derivative */
  /* Use a pre-allocated jacobian object for efficiency */
  using JacobianReferenceType = typename TImageToImageMetric::JacobianType &;
  JacobianReferenceType jacobian = this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobian;
  JacobianReferenceType jacobianPositional =
    this->m_GetValueAndDerivativePerThreadVariables[threadId].MovingTransformJacobianPositional;

  /** For dense transforms, this returns identity */
  this->m_Associate->GetMovingTransform()->ComputeJacobianWithRespectToParametersCachedTemporaries(
    virtualPoint, jacobian, jacobianPositional);

  constexpr unsigned int ImageDimension = ImageToImageMetricv4Type::MovingImageDimension;
  constexpr unsigned int MaxLabels = 256; // This should match the traits class

  // Initialize derivative to zero
  for (unsigned int par = 0; par < this->GetCachedNumberOfLocalParameters(); ++par)
  {
    localDerivativeReturn[par] = NumericTraits<DerivativeValueType>::ZeroValue();
  }

  // For the match cardinality metric, we need to compute the derivative
  // The derivative is non-zero only when the labels don't match
  if (fixedImageValue != movingImageValue)
  {
    // Find which label's gradient to use
    // TODO: instead, iterate over labels and add gradient. More than one gradient could be non-zero
    unsigned int movingLabelIndex = this->GetLabelIndex(movingImageValue, movingImageGradient);

    // The derivative of the match cardinality metric is:
    // d/dp(match_cardinality) = d/dp(1) when labels don't match
    // Since the metric is discontinuous, we use the gradient of the moving image mask

    for (unsigned int par = 0; par < this->GetCachedNumberOfLocalParameters(); ++par)
    {
      for (SizeValueType dim = 0; dim < ImageDimension; ++dim)
      {
        // Get the gradient for the current moving label
        unsigned int gradientIdx = movingLabelIndex * ImageDimension + dim;
        if (gradientIdx < movingImageGradient.Size())
        {
          // The derivative is the gradient of the moving image mask
          // Since we want to minimize the metric (reduce mismatches),
          // we use the negative gradient
          localDerivativeReturn[par] -= jacobian(dim, par) * movingImageGradient[gradientIdx];
        }
      }
    }
  }

  return true;
}

} // end namespace itk

#endif
