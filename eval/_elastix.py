import ants
import numpy as np
import SimpleITK as sitk 

def ants_pred(fixed, moving, seg2):
    """return warped and w_seg2 """
    warps   = []
    w_seg2s = []
    flows   = []
    
    for i in range(fixed.shape[0]):
        im_fixed  = ants.from_numpy(fixed.cpu().numpy()[i,0])
        im_moving = ants.from_numpy(moving.cpu().numpy()[i,0])
        reg       = ants.registration(fixed=im_fixed, moving=im_moving, type_of_transform='SyN')
        warped    = ants.apply_transforms(fixed=im_fixed, moving=im_moving, transformlist=reg['fwdtransforms'])
        warps.append(warped.numpy())
        w_seg2    = ants.apply_transforms(fixed=im_fixed, moving=ants.from_numpy(seg2.cpu().numpy()[i,0]), transformlist=reg['fwdtransforms'])
        w_seg2s.append(w_seg2.numpy())
        flow      = ants.image_read(reg['fwdtransforms'][0])
        flows.append(flow.numpy())
    
    w_seg2 = np.array(w_seg2s)[:, None]
    warped = np.array(warps)[:, None]
    flow   = np.transpose(np.array(flows), (0, 4, 1, 2, 3))
    
    return {
        "warped": warped,
        "w_seg2": w_seg2,
        "flow":   flow,
    }


def elastix_pred(fixed, moving, seg2):
    """return warped, flow, and w_seg2 """
    warps   = []
    w_seg2s = []
    flows   = []
    
    for i in range(fixed.shape[0]):
        # Convert images to SimpleITK format
        
        print('T Fixed: ',  fixed.shape)
        print('T Moving: ', moving.shape)
        print('T Segm: ',   seg2.shape)
        
        im_fixed  = sitk.GetImageFromArray(fixed.cpu().numpy()[i, 0])
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
        result_image = elastixImageFilter.GetResultImage()
        warps.append(np.moveaxis(sitk.GetArrayFromImage(result_image), 0, -1))

        # Apply the same transformation to the segmentation map
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(im_seg2)
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.Execute()
        w_seg2_image = transformixImageFilter.GetResultImage()
        w_seg2s.append( np.moveaxis(sitk.GetArrayFromImage(w_seg2_image), 0, -1))

        # Get the deformation field
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()
        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.Execute()
        flow = transformixImageFilter.GetDeformationField()
        flows.append(flow)#(np.moveaxis(sitk.GetArrayFromImage(flow), 0, -1))
    
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
    }