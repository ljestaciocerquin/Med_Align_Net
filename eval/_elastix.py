import numpy as np
import SimpleITK as sitk 

'''def elastix_pred(fixed, moving, seg2):
    """return warped, flow, and w_seg2 """
    warps   = []
    w_seg2s = []
    flows   = []
    
    for i in range(fixed.shape[0]):
        # Convert images to SimpleITK format
        im_fixed  = sitk.GetImageFromArray(fixed.cpu().numpy()[i, 0]) # 1 x 1 x 192 x 192 x 208 -----> 192 x 192 x 208 -----> 208 x 192 x 192
        im_moving = sitk.GetImageFromArray(moving.cpu().numpy()[i, 0])
        im_seg2   = sitk.GetImageFromArray(seg2.cpu().numpy()[i, 0])
        
        print('Fixed: ', im_fixed.GetSize())
        print('Moving: ', im_moving.GetSize())
        print('Segm: ', im_seg2.GetSize())

        # Setup ElastixImageFilter
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(im_fixed)
        elastixImageFilter.SetMovingImage(im_moving)

        # Set parameter map for non-rigid registration
        parameterMap = sitk.GetDefaultParameterMap("bspline")
        elastixImageFilter.SetParameterMap(parameterMap)
        
        # Execute registration
        elastixImageFilter.Execute()
        
        # Get the result image (warped image)
        warped = elastixImageFilter.GetResultImage()
        warps.append(sitk.GetArrayFromImage(warped)) # 208 x 192 x 192 -----> 192 x 192 x 208  
        
        # Apply the same transformation to the segmentation map
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetMovingImage(im_seg2)
        transformixImageFilter.Execute()
        w_seg2_image = transformixImageFilter.GetResultImage()
        w_seg2s.append(sitk.GetArrayFromImage(w_seg2_image))
        #import pdb; pdb.set_trace()
        
        # Get the deformation field
        transform = elastixImageFilter.GetTransformParameterMap()[0]
        displacement_field = sitk.TransformToDisplacementField(sitk.Transform(transform), 
                                                               referenceImage=im_fixed, 
                                                               outputPixelType=sitk.sitkVectorFloat64)
        displacement_field_array = sitk.GetArrayFromImage(displacement_field)
        flows.append(displacement_field_array)
        #import pdb; pdb.set_trace()
    
    # Convert results to numpy arrays
    w_seg2 = np.array(w_seg2s)[:, None]
    warped = np.array(warps)[:, None]
    flows = np.array(flows)[:, None]
    
    print('w_seg2: ', type(w_seg2), '     w_seg2 shape: ', w_seg2.shape)
    print('warped: ', type(warped), '     warped shape: ', warped.shape)
    print('flows: ', type(flows), '     flows shape: ', flows.shape)
    
    return {
        "warped": warped,
        "w_seg2": w_seg2,
        "flow": flows
    }'''
    
def elastix_pred(fixed, moving, seg2, elastix_output_dir='./elastix/'):
    """return warped, w_seg2, and deformation field"""
    warps   = []
    w_seg2s = []
    flows   = []

    for i in range(fixed.shape[0]):
        # Convert PyTorch tensors to SimpleITK images
        im_fixed = sitk.GetImageFromArray(fixed.cpu().numpy()[i,0])
        im_moving = sitk.GetImageFromArray(moving.cpu().numpy()[i,0])
        
        # Create elastixImageFilter object
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(im_fixed)
        elastixImageFilter.SetMovingImage(im_moving)
        
        # Use default parameter map (elastix configuration for non-rigid registration)
        parameterMapVector = sitk.VectorOfParameterMap()
        defaultParameterMap = sitk.GetDefaultParameterMap("bspline")
        parameterMapVector.append(defaultParameterMap)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        
        # Set output directory
        elastixImageFilter.SetOutputDirectory(elastix_output_dir)
        
        # Perform the registration
        elastixImageFilter.Execute()
        
        # Get the warped image
        warped = elastixImageFilter.GetResultImage()
        warps.append(sitk.GetArrayFromImage(warped))
        
        # Apply the transform to seg2
        im_seg2 = sitk.GetImageFromArray(seg2.cpu().numpy()[i,0])
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.SetMovingImage(im_seg2)
        transformixImageFilter.SetOutputDirectory(elastix_output_dir)
        transformixImageFilter.Execute()
        
        w_seg2 = transformixImageFilter.GetResultImage()
        w_seg2s.append(sitk.GetArrayFromImage(w_seg2))
        
        # Compute the deformation field using Transformix
        transformixImageFilter.SetComputeDeformationField(True)
        transformixImageFilter.Execute()
        flow = transformixImageFilter.GetDeformationField()
        flows.append(sitk.GetArrayFromImage(flow))
        import pdb; pdb.set_trace()
    
    w_seg2 = np.array(w_seg2s)[:, None]
    warped = np.array(warps)[:, None]
    flows  = np.transpose(np.array(flows), (0, 4, 1, 2, 3)) 
    import pdb; pdb.set_trace()
    print('w_seg2: ', type(w_seg2), '     w_seg2 shape: ', w_seg2.shape)
    print('warped: ', type(warped), '     warped shape: ', warped.shape)
    print('flows: ', type(flows), '     flows shape: ', flows.shape)
    
    return {
        "warped": warped,
        "w_seg2": w_seg2,
        "deformation_field": flows,
    }
