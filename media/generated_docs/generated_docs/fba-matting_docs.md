# Project Documentation

## File: `dataloader.py`

### Classes

#### `PredDataset` (Line 7)
Reads image and trimap pairs from folder.

    

**Methods:**
- `__init__(self, img_dir, trimap_dir)`  
  *Line 12: No docstring*

- `__len__(self)`  
  *Line 16: No docstring*

- `__getitem__(self, idx)`  
  *Line 19: No docstring*

## File: `demo.py`

### Functions

#### `np_to_torch(x)` (Line 16)
No docstring

#### `scale_input(x, scale, scale_type)` (Line 20)
Scales inputs to multiple of 8. 

#### `predict_fba_folder(model, args)` (Line 29)
No docstring

#### `pred(image_np, trimap_np, model)` (Line 46)
Predict alpha, foreground and background.
Parameters:
image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
Returns:
fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
alpha: alpha matte image between 0 and 1. Dimensions: (h, w)

## File: `generate_trimap.py`

### Functions

#### `trimap(probs, size, conf_threshold)` (Line 17)
Creates a trimap based on a simple dilation algorithm.

#### `browse_and_process_image(target_class, conf_threshold, output_dir)` (Line 30)
Allows users to select an image file, processes it, and generates a trimap.

#### `select_output_directory()` (Line 83)
Opens a dialog to select the output directory.

#### `main()` (Line 92)
No docstring

## File: `gen_trimap.py`

### Functions

#### `trimap(prob_map, kernel_size, conf_threshold)` (Line 19)
Generate a trimap from the probability map.

#### `browse_and_process_image(target_class, conf_threshold, output_dir)` (Line 48)
Allows users to upload an image, processes it, and generates a trimap.

## File: `layers_WS.py`

### Classes

#### `Conv2d` (Line 6)
No docstring

**Methods:**
- `__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)`  
  *Line 8: No docstring*

- `forward(self, x)`  
  *Line 13: No docstring*

## File: `models.py`

### Classes

#### `MattingModule` (Line 29)
No docstring

**Methods:**
- `__init__(self, net_enc, net_dec)`  
  *Line 30: No docstring*

- `forward(self, image, two_chan_trimap, image_n, trimap_transformed)`  
  *Line 35: No docstring*

#### `ModelBuilder` (Line 41)
No docstring

**Methods:**
- `build_encoder(self, arch)`  
  *Line 42: No docstring*

- `build_decoder(self, arch, batch_norm)`  
  *Line 75: No docstring*

#### `ResnetDilatedBN` (Line 82)
No docstring

**Methods:**
- `__init__(self, orig_resnet, dilate_scale)`  
  *Line 83: No docstring*

- `_nostride_dilate(self, m, dilate)`  
  *Line 112: No docstring*

- `forward(self, x, return_feature_maps)`  
  *Line 127: No docstring*

#### `Resnet` (Line 148)
No docstring

**Methods:**
- `__init__(self, orig_resnet)`  
  *Line 149: No docstring*

- `forward(self, x, return_feature_maps)`  
  *Line 168: No docstring*

#### `ResnetDilated` (Line 191)
No docstring

**Methods:**
- `__init__(self, orig_resnet, dilate_scale)`  
  *Line 192: No docstring*

- `_nostride_dilate(self, m, dilate)`  
  *Line 215: No docstring*

- `forward(self, x, return_feature_maps)`  
  *Line 230: No docstring*

#### `fba_decoder` (Line 268)
No docstring

**Methods:**
- `__init__(self, batch_norm)`  
  *Line 269: No docstring*

- `forward(self, conv_out, img, indices, two_chan_trimap)`  
  *Line 326: No docstring*

## File: `resnet_bn.py`

### Classes

#### `BasicBlock` (Line 14)
No docstring

**Methods:**
- `__init__(self, inplanes, planes, stride, downsample)`  
  *Line 17: No docstring*

- `forward(self, x)`  
  *Line 27: No docstring*

#### `Bottleneck` (Line 46)
No docstring

**Methods:**
- `__init__(self, inplanes, planes, stride, downsample)`  
  *Line 49: No docstring*

- `forward(self, x)`  
  *Line 62: No docstring*

#### `ResNet` (Line 85)
No docstring

**Methods:**
- `__init__(self, block, layers, num_classes)`  
  *Line 87: No docstring*

- `_make_layer(self, block, planes, blocks, stride)`  
  *Line 116: No docstring*

- `forward(self, x)`  
  *Line 133: No docstring*

## File: `resnet_GN_WS.py`

### Classes

#### `BasicBlock` (Line 18)
No docstring

**Methods:**
- `__init__(self, inplanes, planes, stride, downsample)`  
  *Line 21: No docstring*

- `forward(self, x)`  
  *Line 31: No docstring*

#### `Bottleneck` (Line 50)
No docstring

**Methods:**
- `__init__(self, inplanes, planes, stride, downsample)`  
  *Line 53: No docstring*

- `forward(self, x)`  
  *Line 65: No docstring*

#### `ResNet` (Line 88)
No docstring

**Methods:**
- `__init__(self, block, layers, num_classes)`  
  *Line 90: No docstring*

- `_make_layer(self, block, planes, blocks, stride)`  
  *Line 105: No docstring*

- `forward(self, x)`  
  *Line 121: No docstring*

## File: `transforms.py`

### Functions

#### `dt(a)` (Line 7)
No docstring

#### `trimap_transform(trimap)` (Line 11)
No docstring

#### `groupnorm_normalise_image(img, format)` (Line 31)
Accept rgb in range 0,1

#### `groupnorm_denormalise_image(img, format)` (Line 45)
Accept rgb, normalised, return in range 0,1

## File: `__init__.py`

