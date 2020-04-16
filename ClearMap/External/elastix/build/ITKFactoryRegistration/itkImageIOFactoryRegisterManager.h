/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef itkImageIOFactoryRegisterManager_h
#define itkImageIOFactoryRegisterManager_h

namespace itk {

class ImageIOFactoryRegisterManager
{
  public:
  ImageIOFactoryRegisterManager(void (*list[])(void))
    {
    for(;*list; ++list)
      {
      (*list)();
      }
    }
};


//
//  The following code is intended to be expanded at the end of the
//  itkImageFileReader.h and itkImageFileWriter.h files.
//
void  BMPImageIOFactoryRegister__Private();void  BioRadImageIOFactoryRegister__Private();void  Bruker2dseqImageIOFactoryRegister__Private();void  GDCMImageIOFactoryRegister__Private();void  GE4ImageIOFactoryRegister__Private();void  GE5ImageIOFactoryRegister__Private();void  GiplImageIOFactoryRegister__Private();void  HDF5ImageIOFactoryRegister__Private();void  JPEGImageIOFactoryRegister__Private();void  LSMImageIOFactoryRegister__Private();void  MINCImageIOFactoryRegister__Private();void  MRCImageIOFactoryRegister__Private();void  MetaImageIOFactoryRegister__Private();void  NiftiImageIOFactoryRegister__Private();void  NrrdImageIOFactoryRegister__Private();void  PNGImageIOFactoryRegister__Private();void  StimulateImageIOFactoryRegister__Private();void  TIFFImageIOFactoryRegister__Private();void  VTKImageIOFactoryRegister__Private();

//
// The code below registers available IO helpers using static initialization in
// application translation units. Note that this code will be expanded in the
// ITK-based applications and not in ITK itself.
//
namespace {

  void (*ImageIOFactoryRegisterRegisterList[])(void) = {
    BMPImageIOFactoryRegister__Private,BioRadImageIOFactoryRegister__Private,Bruker2dseqImageIOFactoryRegister__Private,GDCMImageIOFactoryRegister__Private,GE4ImageIOFactoryRegister__Private,GE5ImageIOFactoryRegister__Private,GiplImageIOFactoryRegister__Private,HDF5ImageIOFactoryRegister__Private,JPEGImageIOFactoryRegister__Private,LSMImageIOFactoryRegister__Private,MINCImageIOFactoryRegister__Private,MRCImageIOFactoryRegister__Private,MetaImageIOFactoryRegister__Private,NiftiImageIOFactoryRegister__Private,NrrdImageIOFactoryRegister__Private,PNGImageIOFactoryRegister__Private,StimulateImageIOFactoryRegister__Private,TIFFImageIOFactoryRegister__Private,VTKImageIOFactoryRegister__Private,
    0};
  ImageIOFactoryRegisterManager ImageIOFactoryRegisterManagerInstance(ImageIOFactoryRegisterRegisterList);

}

}

#endif
