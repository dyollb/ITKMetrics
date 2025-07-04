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
#ifndef itkMultiLabelImageGradientCalculator_hxx
#define itkMultiLabelImageGradientCalculator_hxx

#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkOffset.h"
#include <set>

namespace itk
{

template <typename TInputImage, typename TCoordRep>
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::MultiLabelImageGradientCalculator()
{
  this->m_UseImageDirection = true;
  this->m_LabelsAnalyzed = false;
}

template <typename TInputImage, typename TCoordRep>
void
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::SetInputImage(const InputImageType * ptr)
{
  Superclass::SetInputImage(ptr);
  this->m_LabelsAnalyzed = false;
  this->m_LabelToIndexMap.clear();
}

template <typename TInputImage, typename TCoordRep>
void
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::FindUniqueLabels() const
{
  if (this->m_LabelsAnalyzed)
  {
    return;
  }

  if (!this->GetInputImage())
  {
    return;
  }

  std::set<InputPixelType> uniqueLabels;

  // Iterate through the image to find unique labels
  ImageRegionConstIterator<InputImageType> it(this->GetInputImage(), this->GetInputImage()->GetBufferedRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    uniqueLabels.insert(it.Get());
  }

  // Create mapping from label to index
  this->m_LabelToIndexMap.clear();
  unsigned int index = 0;
  for (const auto & label : uniqueLabels)
  {
    this->m_LabelToIndexMap[label] = index++;
  }

  this->m_LabelsAnalyzed = true;
}

template <typename TInputImage, typename TCoordRep>
void
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::ComputeLabelGradient(const IndexType & index,
                                                                                InputPixelType    label,
                                                                                OutputType &      gradient) const
{
  // For each spatial dimension, compute the gradient using central differences
  for (unsigned int dim = 0; dim < ImageDimension; ++dim)
  {
    IndexType backwardIndex = index;
    IndexType forwardIndex = index;

    // Check boundaries
    if (index[dim] == 0)
    {
      // Use forward difference
      forwardIndex[dim] = index[dim] + 1;
      if (this->GetInputImage()->GetBufferedRegion().IsInside(forwardIndex))
      {
        InputPixelType forwardPixel = this->GetInputImage()->GetPixel(forwardIndex);
        InputPixelType centerPixel = this->GetInputImage()->GetPixel(index);

        // Binary mask gradient: 1 if label matches, 0 otherwise
        TCoordRep forwardValue = (forwardPixel == label) ? 1.0 : 0.0;
        TCoordRep centerValue = (centerPixel == label) ? 1.0 : 0.0;

        gradient[dim] = forwardValue - centerValue;
      }
    }
    else if (index[dim] == this->GetInputImage()->GetBufferedRegion().GetSize()[dim] - 1)
    {
      // Use backward difference
      backwardIndex[dim] = index[dim] - 1;
      if (this->GetInputImage()->GetBufferedRegion().IsInside(backwardIndex))
      {
        InputPixelType backwardPixel = this->GetInputImage()->GetPixel(backwardIndex);
        InputPixelType centerPixel = this->GetInputImage()->GetPixel(index);

        TCoordRep backwardValue = (backwardPixel == label) ? 1.0 : 0.0;
        TCoordRep centerValue = (centerPixel == label) ? 1.0 : 0.0;

        gradient[dim] = centerValue - backwardValue;
      }
    }
    else
    {
      // Use central difference
      backwardIndex[dim] = index[dim] - 1;
      forwardIndex[dim] = index[dim] + 1;

      if (this->GetInputImage()->GetBufferedRegion().IsInside(backwardIndex) &&
          this->GetInputImage()->GetBufferedRegion().IsInside(forwardIndex))
      {
        InputPixelType forwardPixel = this->GetInputImage()->GetPixel(forwardIndex);
        InputPixelType backwardPixel = this->GetInputImage()->GetPixel(backwardIndex);

        TCoordRep forwardValue = (forwardPixel == label) ? 1.0 : 0.0;
        TCoordRep backwardValue = (backwardPixel == label) ? 1.0 : 0.0;

        gradient[dim] = (forwardValue - backwardValue) / 2.0;
      }
    }
  }

  // Apply spacing correction if using image direction
  if (this->m_UseImageDirection)
  {
    const typename InputImageType::SpacingType &   spacing = this->GetInputImage()->GetSpacing();
    const typename InputImageType::DirectionType & direction = this->GetInputImage()->GetDirection();

    // Create a vector for the gradient
    Vector<TCoordRep, ImageDimension> gradientVector;
    for (unsigned int i = 0; i < ImageDimension; ++i)
    {
      gradientVector[i] = gradient[i] / spacing[i];
    }

    // Apply direction matrix
    Vector<TCoordRep, ImageDimension> physicalGradient = direction * gradientVector;

    // Copy back to output
    for (unsigned int i = 0; i < ImageDimension; ++i)
    {
      gradient[i] = physicalGradient[i];
    }
  }
}

template <typename TInputImage, typename TCoordRep>
auto
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::EvaluateAtIndex(const IndexType & index) const -> OutputType
{
  if (!this->GetInputImage())
  {
    OutputType gradient;
    gradient.SetSize(0);
    return gradient;
  }

  // Make sure we have analyzed the labels
  this->FindUniqueLabels();

  const size_t numLabels = this->m_LabelToIndexMap.size();
  const size_t numComponents = numLabels * ImageDimension;

  OutputType gradient;
  gradient.SetSize(numComponents);
  gradient.Fill(0.0);

  // Get the current pixel value
  InputPixelType currentLabel = this->GetInputImage()->GetPixel(index);

  // Compute gradient for each label
  for (const auto & labelPair : this->m_LabelToIndexMap)
  {
    InputPixelType label = labelPair.first;
    unsigned int   labelIndex = labelPair.second;

    // Compute gradient for this label using central differences
    Vector<TCoordRep, ImageDimension> labelGradient;
    labelGradient.Fill(0.0);

    for (unsigned int dim = 0; dim < ImageDimension; ++dim)
    {
      IndexType backwardIndex = index;
      IndexType forwardIndex = index;

      // Check boundaries
      if (index[dim] == 0)
      {
        // Use forward difference
        forwardIndex[dim] = index[dim] + 1;
        if (this->GetInputImage()->GetBufferedRegion().IsInside(forwardIndex))
        {
          InputPixelType forwardPixel = this->GetInputImage()->GetPixel(forwardIndex);
          TCoordRep      forwardValue = (forwardPixel == label) ? 1.0 : 0.0;
          TCoordRep      centerValue = (currentLabel == label) ? 1.0 : 0.0;

          labelGradient[dim] = forwardValue - centerValue;
        }
      }
      else if (index[dim] == this->GetInputImage()->GetBufferedRegion().GetSize()[dim] - 1)
      {
        // Use backward difference
        backwardIndex[dim] = index[dim] - 1;
        if (this->GetInputImage()->GetBufferedRegion().IsInside(backwardIndex))
        {
          InputPixelType backwardPixel = this->GetInputImage()->GetPixel(backwardIndex);
          TCoordRep      backwardValue = (backwardPixel == label) ? 1.0 : 0.0;
          TCoordRep      centerValue = (currentLabel == label) ? 1.0 : 0.0;

          labelGradient[dim] = centerValue - backwardValue;
        }
      }
      else
      {
        // Use central difference
        backwardIndex[dim] = index[dim] - 1;
        forwardIndex[dim] = index[dim] + 1;

        if (this->GetInputImage()->GetBufferedRegion().IsInside(backwardIndex) &&
            this->GetInputImage()->GetBufferedRegion().IsInside(forwardIndex))
        {
          InputPixelType forwardPixel = this->GetInputImage()->GetPixel(forwardIndex);
          InputPixelType backwardPixel = this->GetInputImage()->GetPixel(backwardIndex);

          TCoordRep forwardValue = (forwardPixel == label) ? 1.0 : 0.0;
          TCoordRep backwardValue = (backwardPixel == label) ? 1.0 : 0.0;

          labelGradient[dim] = (forwardValue - backwardValue) / 2.0;
        }
      }
    }

    // Apply spacing correction if using image direction
    if (this->m_UseImageDirection)
    {
      const typename InputImageType::SpacingType &   spacing = this->GetInputImage()->GetSpacing();
      const typename InputImageType::DirectionType & direction = this->GetInputImage()->GetDirection();

      // Apply spacing correction
      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        labelGradient[i] /= spacing[i];
      }

      // Apply direction matrix
      labelGradient = direction * labelGradient;
    }

    // Store the gradient for this label in the correct position
    unsigned int baseIndex = labelIndex * ImageDimension;
    for (unsigned int i = 0; i < ImageDimension; ++i)
    {
      if (baseIndex + i < numComponents)
      {
        gradient[baseIndex + i] = labelGradient[i];
      }
    }
  }

  return gradient;
}

template <typename TInputImage, typename TCoordRep>
auto
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::Evaluate(const PointType & point) const -> OutputType
{
  ContinuousIndexType index;
  bool                isInside = this->GetInputImage()->TransformPhysicalPointToContinuousIndex(point, index);
  if (!isInside)
  {
    OutputType gradient;
    gradient.SetSize(0);
    return gradient;
  }
  return this->EvaluateAtContinuousIndex(index);
}

template <typename TInputImage, typename TCoordRep>
auto
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::EvaluateAtContinuousIndex(
  const ContinuousIndexType & index) const -> OutputType
{
  IndexType baseIndex;
  baseIndex.CopyWithRound(index);
  return this->EvaluateAtIndex(baseIndex);
}

template <typename TInputImage, typename TCoordRep>
void
MultiLabelImageGradientCalculator<TInputImage, TCoordRep>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "UseImageDirection: " << this->m_UseImageDirection << std::endl;
  os << indent << "LabelsAnalyzed: " << this->m_LabelsAnalyzed << std::endl;
  os << indent << "Number of unique labels: " << this->m_LabelToIndexMap.size() << std::endl;
}

} // end namespace itk

#endif
