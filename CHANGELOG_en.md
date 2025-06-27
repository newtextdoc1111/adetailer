# Changelog (English)

## 2025-06-27

### New Features

- **Class name filter for YOLO models**: Added ability to filter detection results by specific class names for standard YOLO models (not just YOLOWorld)
- **Class-specific prompts support**: Added support for using different prompts for different detected classes using `[CLASS=name]` syntax
- **Bounding box mask option**: Added option to use bounding box as mask for segmentation models instead of using the segmentation mask

### UI Improvements

- Class filter UI is now always visible
- Added "Use bounding box as mask" checkbox option in "Detection" section

## Class-Specific Prompts Usage

The new `[CLASS=name]` syntax allows you to specify different prompts for different detected object classes. This is useful when you want to apply different processing to different types of objects in the same image.

By using this function, you can reduce the chance of accidentally drawing another object after inpainting.

### Syntax

```
[CLASS=face]
1girl, smile, red hair,

[CLASS=eyes]
red eyes, close-up, <lora:MyLora:1>

[CLASS=hands]
heart hands
```

### How it works

1. When ADetailer detects objects in an image, it identifies the class name of each detected object
2. If a class-specific prompt is defined using `[CLASS=classname]`, that prompt will be used for objects of that class
3. If no class-specific prompt is found, the general prompt (text before any `[CLASS=]` tags) will be used as fallback
4. This allows fine-tuned control over how different types of objects are processed

### Notes

- Class names must match the model's class names exactly (e.g., "person", "car", "dog", etc.)
- You can apply different LoRAs for each class
- This feature works with both positive and negative prompts