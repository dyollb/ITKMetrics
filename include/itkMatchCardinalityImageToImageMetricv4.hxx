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
#ifndef itkMatchCardinalityImageToImageMetricv4_hxx
#define itkMatchCardinalityImageToImageMetricv4_hxx

namespace itk
{

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
MatchCardinalityImageToImageMetricv4<TFixedImage,
                                     TMovingImage,
                                     TVirtualImage,
                                     TInternalComputationValueType,
                                     TMetricTraits>::MatchCardinalityImageToImageMetricv4()
{
  // Initialize the threaders
  this->m_DenseGetValueAndDerivativeThreader = MatchCardinalityDenseGetValueAndDerivativeThreaderType::New();
  this->m_SparseGetValueAndDerivativeThreader = MatchCardinalitySparseGetValueAndDerivativeThreaderType::New();
}

template <typename TFixedImage,
          typename TMovingImage,
          typename TVirtualImage,
          typename TInternalComputationValueType,
          typename TMetricTraits>
void
MatchCardinalityImageToImageMetricv4<TFixedImage,
                                     TMovingImage,
                                     TVirtualImage,
                                     TInternalComputationValueType,
                                     TMetricTraits>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "MatchCardinalityImageToImageMetricv4" << std::endl;
}

} // end namespace itk

#endif
