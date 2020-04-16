/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __itkTransformixInputPointFileReaderBIN_h
#define __itkTransformixInputPointFileReaderBIN_h

#include "itkMeshFileReaderBase.h"

#include <fstream>

namespace itk
{

/** \class TransformixInputPointFileReaderBIN
 *
 * \brief A reader that understands transformix input point files
 *
 * A reader that understands transformix input binary point files.
 *
 **/

template< class TOutputMesh >
class TransformixInputPointFileReaderBIN : public MeshFileReaderBase< TOutputMesh >
{
public:

  /** Standard class typedefs. */
  typedef TransformixInputPointFileReaderBIN Self;
  typedef MeshFileReaderBase< TOutputMesh >  Superclass;
  typedef SmartPointer< Self >               Pointer;
  typedef SmartPointer< const Self >         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TransformixInputPointFileReaderBIN, MeshFileReaderBase );

  /** Some convenient typedefs. */
  typedef typename Superclass::DataObjectPointer DatabObjectPointer;
  typedef typename Superclass::OutputMeshType    OutputMeshType;
  typedef typename Superclass::OutputMeshPointer OutputMeshPointer;

  /** Get whether the read points are indices; actually we should store this as a kind
   * of meta data in the output, but i don't understand this concept yet...
   */
  itkGetConstMacro( PointsAreIndices, bool );

  /** Get the number of points that are defined in the file.
   * In fact we also should store this somehow in the output dataobject,
   * but that would mean resizing the point container, while still filled with
   * invalid data (since the GetNumberOfPoints method in a PointSet returns the
   * size of the point container. Storing as metadata would be another option.
   * For now leave it like this. This is a little similar to the ImageIO classes.
   * They also contain information about the image that they will read.
   * For the Mesh readers I didn't choose for a MeshIO-design, but for a
   * MeshReaderBase class and inheriting classes, so somehow it
   * seems logic to store this kind of data in the inheriting reader classes.
   */
  itkGetConstMacro( NumberOfPoints, unsigned long );

  /** Prepare the allocation of the output mesh during the first back
   * propagation of the pipeline. Updates the PointsAreIndices and NumberOfPoints.
   */
  virtual void GenerateOutputInformation( void );

protected:

  TransformixInputPointFileReaderBIN();
  virtual ~TransformixInputPointFileReaderBIN();

  /** Fill the point container of the output. */
  virtual void GenerateData( void );

  unsigned long m_NumberOfPoints;
  bool          m_PointsAreIndices;

  std::ifstream m_Reader;

private:

  TransformixInputPointFileReaderBIN( const Self & ); // purposely not implemented
  void operator=( const Self & );                     // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTransformixInputPointFileReaderBIN.hxx"
#endif

#endif
