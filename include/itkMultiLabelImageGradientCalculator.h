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
#ifndef itkMultiLabelImageGradientCalculator_h
#define itkMultiLabelImageGradientCalculator_h

#include "itkImageFunction.h"
#include "itkCentralDifferenceImageFunction.h"
#include "itkVector.h"
#include "itkVariableLengthVector.h"
#include "itkContinuousIndex.h"
#include <map>

namespace itk
{

/** \class MultiLabelImageGradientCalculator
 * \brief Calculate gradients for labeled images by treating each label as a separate mask
 *
 * This class computes image gradients for labeled images by:
 * 1. For each unique label in the image, creating a binary mask
 * 2. Computing the gradient of each binary mask
 * 3. Assembling the gradients into a vector containing all label gradients
 *
 * The gradient at each point is a vector where:
 * - Elements [0 to ImageDimension-1] are the gradient for label 0
 * - Elements [ImageDimension to 2*ImageDimension-1] are the gradient for label 1
 * - And so on...
 *
 * \ingroup Metrics
 */
template <typename TInputImage, typename TCoordRep = double>
class ITK_TEMPLATE_EXPORT MultiLabelImageGradientCalculator
  : public ImageFunction<TInputImage, VariableLengthVector<TCoordRep>, TCoordRep>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiLabelImageGradientCalculator);

  /** Standard class type aliases. */
  using Self = MultiLabelImageGradientCalculator;
  using Superclass = ImageFunction<TInputImage, VariableLengthVector<TCoordRep>, TCoordRep>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiLabelImageGradientCalculator, ImageFunction);

  /** InputImageType type alias support */
  using InputImageType = TInputImage;
  using InputPixelType = typename InputImageType::PixelType;
  using IndexType = typename InputImageType::IndexType;
  using ContinuousIndexType = ContinuousIndex<TCoordRep, TInputImage::ImageDimension>;

  /** Dimension underlying input image. */
  static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

  /** Point type alias support */
  using PointType = typename Superclass::PointType;

  /** OutputType type alias support */
  using OutputType = VariableLengthVector<TCoordRep>;
  using CoordinateType = TCoordRep;

  /** Evaluate the function at specified Point */
  OutputType
  Evaluate(const PointType & point) const override;

  /** Evaluate the function at specified Index */
  OutputType
  EvaluateAtIndex(const IndexType & index) const override;

  /** Evaluate the function at specified ContinuousIndex */
  OutputType
  EvaluateAtContinuousIndex(const ContinuousIndexType & index) const override;

  /** Set the input image. */
  void
  SetInputImage(const InputImageType * ptr) override;

  /** Set/Get whether to use image direction in gradient computation */
  itkSetMacro(UseImageDirection, bool);
  itkGetConstMacro(UseImageDirection, bool);
  itkBooleanMacro(UseImageDirection);

protected:
  MultiLabelImageGradientCalculator();
  ~MultiLabelImageGradientCalculator() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Compute gradients for a specific label at a given index */
  void
  ComputeLabelGradient(const IndexType & index, InputPixelType label, OutputType & gradient) const;

  /** Find all unique labels in the image */
  void
  FindUniqueLabels() const;

  /** Map of unique labels found in the image */
  mutable std::map<InputPixelType, unsigned int> m_LabelToIndexMap;

  /** Whether to use image direction in gradient computation */
  bool m_UseImageDirection{ true };

  /** Whether labels have been analyzed */
  mutable bool m_LabelsAnalyzed{ false };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiLabelImageGradientCalculator.hxx"
#endif

#endif
