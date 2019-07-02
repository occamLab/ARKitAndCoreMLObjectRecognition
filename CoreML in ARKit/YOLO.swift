import Foundation
import UIKit
import CoreML

class YOLO {
    public static let inputWidth = 416
    public static let inputHeight = 416
    public static let maxBoundingBoxes = 10
    
    // Tweak these values to get more or fewer predictions.
    let confidenceThreshold: Double = 0.6
    let iouThreshold: Double = 0.5
    
    struct Prediction {
        let classIndex: Int
        let score: Float
        let rect: CGRect
    }
    
    //import the neural network model to use
    let model = YOLOv3()
    
    public init() { }

    //try to create a prediction from passing an image to the model
    public func predict(image: CVPixelBuffer) throws -> [Prediction] {
        if let output = try? model.prediction(image: image, iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold) {
            return computeBoundingBoxes(features: output)
        } else {
            return []
        }
    }
    
    public func computeBoundingBoxes(features: YOLOv3Output) -> [Prediction] {
        
        var predictions = [Prediction]()
        
        print(features.confidence[[0, 0]])
        
        let dim0size = Int(features.confidence.shape[0])
        let dim1size = Int(features.confidence.shape[1])

        for i in 0..<dim0size {
            
            var maximumConfidence = 0
            var maximumIndex = 0
            
            for j in 0..<dim1size {
                
                if Int(features.confidence[[NSNumber(value: i), NSNumber(value: j)]]) > maximumConfidence{
                    maximumConfidence = Int(features.confidence[[NSNumber(value: i), NSNumber(value: j)]])
                    maximumIndex = j
                }
                
                print(features.confidence[[NSNumber(value: i), NSNumber(value: j)]])
                print(features.coordinates[[NSNumber(value: i), NSNumber(value: j)]])
            }
            if Double(maximumConfidence) >= confidenceThreshold{
                //make a prediction with a bounding box, typeformat, and confidence
                predictions.append(Prediction(classIndex: maximumIndex,
                                              score: Float(maximumConfidence),
                                              rect: CGRect(x: CGFloat(features.coordinates[[NSNumber(value:i),NSNumber(value:0)]]), y: CGFloat(features.coordinates[[NSNumber(value:i),NSNumber(value:1)]]), width: CGFloat(features.coordinates[[NSNumber(value:i),NSNumber(value:2)]]), height: CGFloat(features.coordinates[[NSNumber(value:i),NSNumber(value:3)]]))))
            }
        }
        // We already filtered out any bounding boxes that have very low scores,
        // but there still may be boxes that overlap too much with others. We'll
        // use "non-maximum suppression" to prune those duplicate bounding boxes.
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: Float(iouThreshold))
    }
}
