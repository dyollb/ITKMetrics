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
#ifndef itkMultiLabelImageGradientFilter_hxx
#define itkMultiLabelImageGradientFilter_hxx

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include <set>

namespace itk
{

template <typename TInputImage, typename TOutputImage>
MultiLabelImageGradientFilter<TInputImage, TOutputImage>::MultiLabelImageGradientFilter()
{
  this->SetNumberOfWorkUnits(4);
  this->m_UseImageDirection = true;
  this->m_LabelsAnalyzed = false;
}

template <typename TInputImage, typename TOutputImage>
void
MultiLabelImageGradientFilter<TInputImage, TOutputImage>::FindUniqueLabels()
{
  if (this->m_LabelsAnalyzed)
  {
    return;
  }

  const InputImageType * inputImage = this->GetInput();
  if (!inputImage)
  {
    return;
  }

  std::set<InputPixelType> uniqueLabels;

  // Iterate through the image to find unique labels
  ImageRegionConstIterator<InputImageType> it(inputImage, inputImage->GetBufferedRegion());
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

template <typename TInputImage, typename TOutputImage>
typename MultiLabelImageGradientFilter<TInputImage, TOutputImage>::GradientImageType::Pointer
MultiLabelImageGradientFilter<TInputImage, TOutputImage>::ComputeLabelGradient(InputPixelType label)
{
  const InputImageType * inputImage = this->GetInput();

  // Create binary mask for this label
  auto thresholdFilter = BinaryThresholdFilterType::New();
  thresholdFilter->SetInput(inputImage);
  thresholdFilter->SetLowerThreshold(label);
  thresholdFilter->SetUpperThreshold(label);
  thresholdFilter->SetInsideValue(1.0);
  thresholdFilter->SetOutsideValue(0.0);

  // Compute gradient of the binary mask
  auto gradientFilter = GradientFilterType::New();
  gradientFilter->SetInput(thresholdFilter->GetOutput());
  gradientFilter->SetUseImageDirection(this->m_UseImageDirection);
  gradientFilter->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());

  try
  {
    gradientFilter->Update();
  }
  catch (const ExceptionObject & e)
  {
    itkExceptionMacro("Error computing gradient for label " << static_cast<int>(label) << ": " << e.GetDescription());
  }

  return gradientFilter->GetOutput();
}

template <typename TInputImage, typename TOutputImage>
void
MultiLabelImageGradientFilter<TInputImage, TOutputImage>::GenerateData()
{
  // Make sure we have analyzed the labels
  this->FindUniqueLabels();

  const InputImageType * inputImage = this->GetInput();
  OutputImageType *      outputImage = this->GetOutput();

  // Allocate output image
  outputImage->SetRegions(inputImage->GetBufferedRegion());
  outputImage->SetSpacing(inputImage->GetSpacing());
  outputImage->SetOrigin(inputImage->GetOrigin());
  outputImage->SetDirection(inputImage->GetDirection());

  // Set number of components based on number of labels * dimension
  const size_t numLabels = this->m_LabelToIndexMap.size();
  const size_t numComponents = numLabels * InputImageDimension;
  outputImage->SetNumberOfComponentsPerPixel(numComponents);
  outputImage->Allocate();

  // Initialize output to zero
  OutputPixelType zeroPixel;
  zeroPixel.SetSize(numComponents);
  zeroPixel.Fill(0.0);
  outputImage->FillBuffer(zeroPixel);

  // Compute gradients for each label
  std::map<InputPixelType, typename GradientImageType::Pointer> labelGradients;

  size_t labelCount = 0;

  for (const auto & labelPair : this->m_LabelToIndexMap)
  {
    InputPixelType label = labelPair.first;
    unsigned int   labelIndex = labelPair.second;

    // Compute gradient for this label
    typename GradientImageType::Pointer labelGradient = this->ComputeLabelGradient(label);
    labelGradients[label] = labelGradient;

    // Report progress using the proper ITK mechanism
    ++labelCount;
    this->UpdateProgress(static_cast<float>(labelCount) /
                         (numLabels * 2.0f)); // Use half progress for gradient computation
  }

  // Assemble the output image
  ImageRegionIterator<OutputImageType>     outputIt(outputImage, outputImage->GetBufferedRegion());
  ImageRegionConstIterator<InputImageType> inputIt(inputImage, inputImage->GetBufferedRegion());

  const size_t totalPixels = outputImage->GetBufferedRegion().GetNumberOfPixels();
  size_t       pixelCount = 0;

  for (outputIt.GoToBegin(), inputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt, ++inputIt)
  {
    InputPixelType  currentLabel = inputIt.Get();
    OutputPixelType outputPixel(numComponents);
    outputPixel.Fill(0.0);

    // Find the corresponding gradient for this label
    auto labelIt = this->m_LabelToIndexMap.find(currentLabel);
    if (labelIt != this->m_LabelToIndexMap.end())
    {
      unsigned int labelIndex = labelIt->second;

      // Get the gradient for this label
      auto gradientIt = labelGradients.find(currentLabel);
      if (gradientIt != labelGradients.end())
      {
        typename GradientImageType::IndexType          currentIndex = inputIt.GetIndex();
        CovariantVector<RealType, InputImageDimension> labelGradient = gradientIt->second->GetPixel(currentIndex);

        // Store the gradient in the correct position in the output vector
        unsigned int baseIndex = labelIndex * InputImageDimension;
        for (unsigned int i = 0; i < InputImageDimension; ++i)
        {
          if (baseIndex + i < numComponents)
          {
            outputPixel[baseIndex + i] = labelGradient[i];
          }
        }
      }
    }

    outputIt.Set(outputPixel);

    // Report progress for assembly phase (use remaining half of progress)
    ++pixelCount;
    if (pixelCount % 1000 == 0 || pixelCount == totalPixels) // Report every 1000 pixels or at the end
    {
      this->UpdateProgress(0.5f + 0.5f * static_cast<float>(pixelCount) / totalPixels);
    }
  }
}

template <typename TInputImage, typename TOutputImage>
void
MultiLabelImageGradientFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "UseImageDirection: " << this->m_UseImageDirection << std::endl;
  os << indent << "LabelsAnalyzed: " << this->m_LabelsAnalyzed << std::endl;
  os << indent << "Number of unique labels: " << this->m_LabelToIndexMap.size() << std::endl;
}

} // end namespace itk

#endif
