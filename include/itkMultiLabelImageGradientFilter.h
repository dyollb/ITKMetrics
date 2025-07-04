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
#ifndef itkMultiLabelImageGradientFilter_h
#define itkMultiLabelImageGradientFilter_h

#include "itkImageToImageFilter.h"
#include "itkVector.h"
#include "itkCovariantVector.h"
#include "itkImage.h"
#include "itkGradientImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkComposeImageFilter.h"
#include <map>
#include <set>

namespace itk
{

/** \class MultiLabelImageGradientFilter
 * \brief Computes gradients for labeled images by treating each label as a separate mask
 *
 * This filter computes image gradients for labeled images by:
 * 1. Finding all unique labels in the input image
 * 2. For each label, creating a binary mask where pixels with that label = 1, others = 0
 * 3. Computing the gradient of each binary mask
 * 4. Assembling the gradients into a vector image containing all label gradients
 *
 * The output is a vector image where each pixel contains a vector of gradients
 * for all labels at that location.
 *
 * \ingroup Metrics
 */
template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT MultiLabelImageGradientFilter : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiLabelImageGradientFilter);

  /** Standard class type aliases. */
  using Self = MultiLabelImageGradientFilter;
  using Superclass = ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiLabelImageGradientFilter, ImageToImageFilter);

  /** Extract dimension from input image. */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;

  /** Convenient type alias for simplifying declarations. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using InputPixelType = typename InputImageType::PixelType;
  using OutputPixelType = typename OutputImageType::PixelType;
  using InputImagePointer = typename InputImageType::Pointer;
  using OutputImagePointer = typename OutputImageType::Pointer;

  /** Internal types for gradient computation */
  using RealType = float;
  using BinaryImageType = Image<RealType, InputImageDimension>;
  using GradientImageType = Image<CovariantVector<RealType, InputImageDimension>, InputImageDimension>;

  using BinaryThresholdFilterType = BinaryThresholdImageFilter<InputImageType, BinaryImageType>;
  using GradientFilterType = GradientImageFilter<BinaryImageType, RealType, RealType>;

  /** Set/Get whether to use image direction in gradient computation */
  itkSetMacro(UseImageDirection, bool);
  itkGetConstMacro(UseImageDirection, bool);
  itkBooleanMacro(UseImageDirection);

  /** Set/Get sigma for gradient computation (for compatibility) */
  itkSetMacro(Sigma, double);
  itkGetConstMacro(Sigma, double);

  /** Set/Get normalize across scale (for compatibility) */
  itkSetMacro(NormalizeAcrossScale, bool);
  itkGetConstMacro(NormalizeAcrossScale, bool);
  itkBooleanMacro(NormalizeAcrossScale);

protected:
  MultiLabelImageGradientFilter();
  ~MultiLabelImageGradientFilter() override = default;

  void
  GenerateData() override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Find all unique labels in the input image */
  void
  FindUniqueLabels();

  /** Compute gradient for a specific label */
  typename GradientImageType::Pointer
  ComputeLabelGradient(InputPixelType label);

  /** Map of unique labels found in the image */
  std::map<InputPixelType, unsigned int> m_LabelToIndexMap;

  /** Whether to use image direction in gradient computation */
  bool m_UseImageDirection{ true };

  /** Whether labels have been analyzed */
  bool m_LabelsAnalyzed{ false };

  /** Sigma for gradient computation (for compatibility) */
  double m_Sigma{ 1.0 };

  /** Whether to normalize across scale (for compatibility) */
  bool m_NormalizeAcrossScale{ true };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiLabelImageGradientFilter.hxx"
#endif

#endif
