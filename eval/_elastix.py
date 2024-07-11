import numpy as np
import SimpleITK as sitk 

def elastix_pred(fixed, moving, seg2, elastix_output_dir='./elastix/'):
    """return warped, w_seg2, and deformation field"""
    warps   = []
    w_seg2s = []
    flows   = []

    for i in range(fixed.shape[0]):
        # Convert PyTorch tensors to SimpleITK images
        im_fixed = sitk.GetImageFromArray(fixed.cpu().numpy()[i,0]) # 1 x 1 x 192 x 192 x 208 -----> 192 x 192 x 208 -----> 208 x 192 x 192
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
        warps.append(sitk.GetArrayFromImage(warped)) # 208 x 192 x 192 -----> 192 x 192 x 208  
        
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
        #import pdb; pdb.set_trace()
    
    w_seg2 = np.array(w_seg2s)[:, None]                         # 1 x 1 x 192 x 192 x 208
    warped = np.array(warps)[:, None]                           # 1 x 1 x 192 x 192 x 208
    flows  = np.transpose(np.array(flows), (0, 4, 1, 2, 3))     # 1 x 3 x 192 x 192 x 208
    
    return {
        "warped": warped,
        "w_seg2": w_seg2,
        "flow": flows,
    }