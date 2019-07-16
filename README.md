# CoreML-in-ARKit
This is a project designed to use machine learning, specifically the YOLO v3 databaseto create an object recognition and localization app for the blind and visually impaired. The App uses the YOLO v3 model to identify objects. Then it uses ARKit to place 3d Labels into the scene at the locations of the identified objects.

Combining MLCore and ARKit was based off of this example project, [http://machinethink.net/blog/object-detection-with-yolo/](https://github.com/hanleyweng/CoreML-in-ARKit), which could place 3d labels on the objects returned from an ML model.

The application of YOLOv3 in this project was based off of the implementation of YOLOv3 in Swift here: [https://github.com/Ma-Dan/YOLOv3-CoreML](https://github.com/Ma-Dan/YOLOv3-CoreML)

For a detailed description of how the YOLO method works plsease see either his blog post: [http://machinethink.net/blog/object-detection-with-yolo/](http://machinethink.net/blog/object-detection-with-yolo/) or the orginal creators of YOLOv3 here: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)

This app was designed to provide an automated alternattive for the Olin Occam Lab's (View share program)[] and their objectLocator program()[https://github.com/occamLab/AutomaticObjectDetection](https://github.com/occamLab/ObjectLocator)

## Instructions For Developers

If you want to use this code to develop your own project: clone or fork this repo
You'll have to download "YOLOv3" from [Apple's Machine Learning page](https://developer.apple.com/machine-learning/models/), and copy it into your XCode project.
If you're having issues, double check that the model is part of a target [(source: stackoverflow)](https://stackoverflow.com/questions/45884085/model-is-not-part-of-any-target-add-the-model-to-a-target-to-enable-generation).
