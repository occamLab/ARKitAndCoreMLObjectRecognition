//
//  ViewController.swift
//  CoreML in ARKit
//
//  Created by Hanley Weng on 14/7/17.
//  Copyright © 2017 CompanyName. All rights reserved.
//

import UIKit
import SceneKit
import ARKit

import AVFoundation
import CoreMedia
import Vision

class SCNIdentifiedObject: SCNNode {
    var objectName: NSString?

    init(objectName: NSString) {
        super.init()
        self.objectName = objectName
    }
    
    /// Encode the identified object node.
    ///
    /// - Parameter aCoder: the encoder
    override func encode(with aCoder: NSCoder) {
        super.encode(with: aCoder)
        aCoder.encode(objectName, forKey: "objectName")
    }
    
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        objectName = aDecoder.decodeObject(of: NSString.self, forKey: "objectName")
    }
}


class ViewController: UIViewController, ARSCNViewDelegate {
    
    // Line 1. Create an instance of AVSpeechSynthesizer.
    var speechSynthesizer = AVSpeechSynthesizer()
    
    // true: use Vision to drive Core ML, false: use plain Core ML
    let useVision = false
    
    //Dictates the radius in which nodes will be combined distance in in meteres
    let recombiningThreshold: Double = 0.5
    
    // How many predictions we can do concurrently.
    static let maxInflightBuffers = 3
    
    //creates an instance of the functions that controll the bahavior of the YOLO nural network
    let yolo = YOLO()
    
    //MARK: Private variables
    var requests = [VNCoreMLRequest]()
    var startTimes: [CFTimeInterval] = []

    //creates an instance of a structure which can store a static image for processing
    let ciContext = CIContext()
    var resizedPixelBuffers: [CVPixelBuffer?] = []
    
    var framesDone = 0
    var frameCapturingStartTime = CACurrentMediaTime()
    
    var inflightBuffer = 0
    
    //the distance from the user in Metres that objects will be announced from.
    let queryRange = 1.5
    
    //can be either "crop" (lower area of effect) to perform processing on a selection of the area from the center of the screen or "scaleFit" to scale the whole image into the proper size and perform image processing on the whole image (lower resoulution)
    let imageProcessingSetting = "scaleFit"
    
    
    //MARK: Outlet
    // SCENE
    @IBOutlet var sceneView: ARSCNView!
    let bubbleDepth : Float = 0.01 // the 'depth' of 3D text
    var latestPrediction : String = "…" // a variable containing the latest CoreML prediction
    
    @IBOutlet weak var debugImageView: UIImageView!
    @IBOutlet weak var queryEnvButton: UIButton!
    
    @IBAction func queryEnvironment(_ sender: Any) {
        //iterate throguh the nodes in the scene
        for child:SCNNode in sceneView.scene.rootNode.childNodes{
            
            if let child = child as? SCNIdentifiedObject, let pov = sceneView.pointOfView {
                //creates a node at the location o f the camera foruse in distance calcualtions
                //let selfNode = SCNNode()
                //sceneView.pointOfView?.addChildNode(selfNode)
                
                
                //if the node is close enough to the user
                if d3Distance(pov.position, child.position) <= Double(queryRange){
                    // Line 2. Create an instance of AVSpeechUtterance and pass in a String to be spoken.
                    let speechUtterance: AVSpeechUtterance = AVSpeechUtterance(string: "\(child.objectName!) accurate. Distance: \(round(10*d3Distance(pov.position, child.position))/10) meters.")
                    //Line 3. Specify the speech utterance rate. 1 = speaking extremely slowly The higher the values the slower speech patterns. The default rate, AVSpeechUtteranceDefaultSpeechRate is 0.5
                    speechUtterance.rate = 0.5
                    // Line 4. Specify the voice. It is explicitly set to English here, but it will use the device default if not specified.
                    speechUtterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                    // Line 5. Pass in the urrerance to the synthesizer to actually speak.
                    speechSynthesizer.speak(speechUtterance)
                }
                
                //deletes the node created at the camera
                //selfNode.removeFromParentNode()
            }
        }
    }
    
    // COREML
    let dispatchQueueML = DispatchQueue(label: "com.hw.dispatchqueueml") // A Serial Queue
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        sceneView.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        // Create a new scene
        let scene = SCNScene()
        
        // Set the scene to the view
        sceneView.scene = scene
        
        // Enable Default Lighting - makes the 3D text a bit poppier.
        sceneView.autoenablesDefaultLighting = true
        
        //////////////////////////////////////////////////
        setUpCoreImage()
        setUpVision()
        
        // Begin Loop to Update CoreML
        loopCoreMLUpdate()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        // Enable plane detection
        configuration.planeDetection = [.horizontal,.vertical]
        
        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }
    
    // MARK: - Initialization
    func setUpCoreImage() {
        // Since we might be running several requests in parallel, we also need
        // to do the resizing in different pixel buffers or we might overwrite a
        // pixel buffer that's already in use.
        // one pixel buffer will hold one image section which we are going to be checking. thius one pixel buffer is an 'image' or image section representing one object
        for _ in 0..<YOLO.maxBoundingBoxes {
            //CVPixel Buffer is an image stored as pixels
            var emptyPixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(nil, YOLO.inputWidth, YOLO.inputHeight,
                                             kCVPixelFormatType_32BGRA, nil,
                                             &emptyPixelBuffer)
            //error handeling
            if status != kCVReturnSuccess {
                print("Error: could not create resized pixel buffer", status)
            }
            //add the pixel buffer/image
            resizedPixelBuffers.append(emptyPixelBuffer)
        }
    }
    
    func setUpVision() {
        //attempts to create the yolo neural network
        guard let selectedModel = try? VNCoreMLModel(for: yolo.model.model) else {
            print("Error: could not create Vision model")
            return
        }
        
        for _ in 0..<ViewController.maxInflightBuffers {
            //sets up a bunch of requests to send to the neural network
            let request = VNCoreMLRequest(model: selectedModel, completionHandler: visionRequestDidComplete)
            
            // NOTE: If you choose another crop/scale option, then you must also
            // change how the BoundingBox objects get scaled when they are drawn.
            // Currently they assume the full input image is used.
            request.imageCropAndScaleOption = .scaleFill
            requests.append(request)
        }
    }
    
    // MARK: - ARSCNViewDelegate
    
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        DispatchQueue.main.async {
            // Do any desired updates to SceneKit here.
        }
    }
    
    ////////////////////////////
    
    // MARK: - UI stuff
    
    //arrainges views on the screen
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    //sets the status bar style
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    //sets the video from the camera to take up all of its allocated space
    func resizePreviewLayer() {
        sceneView.layer.frame = sceneView.bounds
    }
    
    //////////////////////////////////////
    
    
    // MARK: - Status Bar: Hide
    override var prefersStatusBarHidden : Bool {
        return true
    }
    
    // MARK: - Utility Functions
    func createNewBubbleParentNode(_ text : String) -> SCNNode {
        // Warning: Creating 3D Text is susceptible to crashing. To reduce chances of crashing; reduce number of polygons, letters, smoothness, etc.
        
        // TEXT BILLBOARD CONSTRAINT
        let billboardConstraint = SCNBillboardConstraint()
        billboardConstraint.freeAxes = SCNBillboardAxis.Y
        
        // BUBBLE-TEXT
        let bubble = SCNText(string: text, extrusionDepth: CGFloat(bubbleDepth))
        let font = UIFont(name: "Futura", size: 0.15)
        bubble.font = font
        bubble.alignmentMode = kCAAlignmentCenter
        bubble.firstMaterial?.diffuse.contents = UIColor.orange
        bubble.firstMaterial?.specular.contents = UIColor.white
        bubble.firstMaterial?.isDoubleSided = true
        // bubble.flatness // setting this too low can cause crashes.
        bubble.chamferRadius = CGFloat(bubbleDepth)
        
        // BUBBLE NODE
        let (minBound, maxBound) = bubble.boundingBox
        let bubbleNode = SCNNode(geometry: bubble)
        // Centre Node - to Centre-Bottom point
        bubbleNode.pivot = SCNMatrix4MakeTranslation( (maxBound.x - minBound.x)/2, minBound.y, bubbleDepth/2)
        // Reduce default text size
        bubbleNode.scale = SCNVector3Make(0.2, 0.2, 0.2)
        
        // CENTRE POINT NODE
        let sphere = SCNSphere(radius: 0.005)
        sphere.firstMaterial?.diffuse.contents = UIColor.cyan
        let sphereNode = SCNNode(geometry: sphere)
        
        // BUBBLE PARENT NODE
        let bubbleNodeParent = SCNIdentifiedObject(objectName: NSString(string: text))
        bubbleNodeParent.addChildNode(bubbleNode)
        bubbleNodeParent.addChildNode(sphereNode)
        bubbleNodeParent.constraints = [billboardConstraint]
        
        return bubbleNodeParent
    }
    
    //Add a 3d marker at the given location with the given label
    func add3dLabel(label: String, certanty : Float, point : CGPoint){
        //print(point)
        //let arRaycast = sceneView.session.raycastQuery(from: point, allowing: .estimatedPlane, alignment: .any)
        let arHitTestResults : [ARHitTestResult] = sceneView.hitTest(point, types: [.featurePoint]) // Alternatively, we could use '.existingPlaneUsingExtent' for more grounded hit-test-points.
        if let closestResult = arHitTestResults.first {
            
            // Get Coordinates of the neares hit point in world space
            let transform : matrix_float4x4 = closestResult.worldTransform
            let worldCoord : SCNVector3 = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
            
            //variables holding the sum of the positions of the nearby nodes and the number of nodes referencing the same object. these are used for calculating averages
            var nodePositionSum: SCNVector3 = worldCoord
            var nodeNumber: Int = 1
            
            // Create 3D Text
            
            for child:SCNNode in sceneView.scene.rootNode.childNodes{
                if let child = child as? SCNIdentifiedObject, let objectName = child.objectName, d3Distance(child.position,worldCoord) <= recombiningThreshold && String(objectName).contains(label) {
                    
                    //increments the sum and total number of nodes used (useful for computing averages)
                    nodePositionSum = SCNVector3(nodePositionSum.x+child.position.x,nodePositionSum.y+child.position.y,nodePositionSum.z+child.position.z)
                    nodeNumber += 1
                    
                    //removes the node so it does not display extra labels
                    child.removeFromParentNode()
                }
            }
            
            //creates the new node to have tha average position of all of the found nodes
            let node : SCNNode = createNewBubbleParentNode("\(label): \(round(1000*certanty)/10)%")
            sceneView.scene.rootNode.addChildNode(node)
            //seet the position to be equal to the average node position
            node.position = SCNVector3(nodePositionSum.x/Float(nodeNumber),nodePositionSum.y/Float(nodeNumber),nodePositionSum.z/Float(nodeNumber))
            print("label \(label) certanty \(round(1000*certanty)/10) world \(worldCoord) actual \(node.position)")
        }
    }
    
    //find the 3dDistance between two world points
    func d3Distance(_ point1 : SCNVector3, _ point2: SCNVector3) -> Double{
        return Double(sqrt(Double(pow((point1.x-point2.x),2))+Double(pow((point1.y-point2.y),2))+Double(pow((point1.z-point2.z),2))))
    }
    
    //displays a CVPixelBuffer to the image debugger
    func displayImage(buffer: CVPixelBuffer, debugImageView: UIImageView) -> Int{
        
        //convert the buffer to a CIImage
        let ciimage = CIImage(cvImageBuffer: buffer)
        
        //Convert into a UIImage
        let image = UIImage(ciImage: ciimage)
        
        //display the image
        debugImageView.image = image
        
        return 0
    }
    
    //calculate the cordinates for the a given point in the small image in the chordiantes of the view
    func calculatePointCords(yoloPoint: CGPoint, view: UIView, sceneView: ARSCNView, imageProcessingSetting: String) -> CGPoint? {
        let viewShort = view.bounds.width
        let viewLong = view.bounds.height
        
        //displayImage(buffer: (sceneView.session.currentFrame?.capturedImage)!, debugImageView: debugImageView)
        
        // note: image is rotated 90 degrees (or maybe 270 degrees?) with respect to the view
        let imageLong = CVPixelBufferGetWidth((sceneView.session.currentFrame?.capturedImage)!)
        let imageShort = CVPixelBufferGetHeight((sceneView.session.currentFrame?.capturedImage)!)
        
        let YOLOSize = YOLO.inputHeight
        
        
        switch imageProcessingSetting{
        case "crop":
            
            let imagePixelLocationShort = yoloPoint.x * CGFloat(YOLOSize) + CGFloat((imageShort-YOLOSize)/2)
            let imagePixelLocationLong = yoloPoint.y * CGFloat(YOLOSize) + CGFloat((imageLong-YOLOSize)/2)
            
            let viewPixelLocationShort = CGFloat(viewShort)/CGFloat(imageShort) * imagePixelLocationShort
            
            let viewPixelLocationLong = CGFloat(viewLong)/CGFloat(imageLong) * imagePixelLocationLong
            
            print("x \(viewPixelLocationShort), y \(viewPixelLocationLong)")
            
            return CGPoint(x: viewPixelLocationShort, y: viewPixelLocationLong)
        case "scaleFit":
            let imagePixelLocationShort = yoloPoint.x * CGFloat(YOLOSize) * CGFloat(imageLong/YOLOSize)
            let imagePixelLocationLong = yoloPoint.y * CGFloat(YOLOSize) * CGFloat(imageLong/YOLOSize)

            let viewPixelLocationShort = CGFloat(viewShort)/CGFloat(imageShort) * imagePixelLocationShort
            
            let viewPixelLocationLong = CGFloat(viewLong)/CGFloat(imageLong) * imagePixelLocationLong
            
            //print("x \(viewPixelLocationShort), y \(viewPixelLocationLong)")
            
            return CGPoint(x: viewPixelLocationShort, y: viewPixelLocationLong)
            
            
            
        default:
            print("passed improper scaling option to calculatePointCoords")
            return nil
        }
        
        
    }
    
    // MARK: - CoreML Vision Handling
    
    func loopCoreMLUpdate() {
        // Continuously run CoreML whenever it's ready. (Preventing 'hiccups' in Frame Rate)
        
        dispatchQueueML.async {
            // 1. Run Update.
            self.updateCoreML()
            
            // 2. Loop this function.
            self.loopCoreMLUpdate()
        }
        
    }

    func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int, imageProcessingSetting:String) {
        // Measure how long it takes to predict a single video frame.
        let startTime = CACurrentMediaTime()
        
        // This is an alternative way to resize the image (using vImage):
        //if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
        //                                              width: YOLO.inputWidth,
        //                                              height: YOLO.inputHeight) {
        
        // Resize the input with Core Image to 416x416.
        if let resizedPixelBuffer = resizedPixelBuffers[inflightIndex] {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            var orientedImage: CIImage?
            switch imageProcessingSetting {
            case "crop":
                let croppedImage = ciImage.cropped(to: CGRect(x: (ciImage.extent.size.width/2)-CGFloat(YOLO.inputHeight/2), y: (ciImage.extent.size.height/2)-CGFloat(YOLO.inputWidth/2), width: CGFloat(YOLO.inputHeight), height: CGFloat(YOLO.inputWidth)))
                orientedImage = croppedImage.oriented(CGImagePropertyOrientation.right)
            case "scaleFit" :
                let sy = CGFloat(YOLO.inputHeight) / CGFloat(ciImage.extent.size.width)
                let sx = sy
                let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
                let scaledImage = ciImage.transformed(by: scaleTransform)
                orientedImage = scaledImage.oriented(CGImagePropertyOrientation.right)

            default :
                orientedImage = nil
                print("improper imageProcessingSetting in Predict(pixelBuffer: ...")
                return
            }

            
            ciContext.render(orientedImage!, to: resizedPixelBuffer)
            //ciContext.render(ciImage, to: resizedPixelBuffer)
            
            // Give the resized input to our model.
            if let result = try? yolo.predict(image: resizedPixelBuffer){
                let elapsed = CACurrentMediaTime() - startTime
                showOnMainThread(result, elapsed)
                displayImage(buffer: resizedPixelBuffer, debugImageView: debugImageView)
            } else {
                //if the model could not find anything
                print("BOGUS")
            }
        }
    }
    
    func predictUsingVision(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
        // Measure how long it takes to predict a single video frame. Note that
        // predict() can be called on the next frame while the previous one is
        // still being processed. Hence the need to queue up the start times.
        startTimes.append(CACurrentMediaTime())
        
        // Vision will automatically resize the input image.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        let request = requests[inflightIndex]
        
        // Because perform() will block until after the request completes, we
        // run it on a concurrent background queue, so that the next frame can
        // be scheduled in parallel with this one.
        DispatchQueue.global().async {
            try? handler.perform([request])
        }
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        //print(request.results)
        
        //breakpoint
        
    }
    
    func showOnMainThread(_ predictions: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
        //redraw the bounding boxes
        DispatchQueue.main.async {
            //show the predictions in the view
            self.show(predictions: predictions)
        }
    }
    
    func show(predictions: [YOLO.Prediction]) {
        //iterate through all of the bounding boxes
        for i in 0..<predictions.count {
                let prediction = predictions[i]
                
                // The predicted bounding box is in the coordinate space of the input
                // image, which is a square image of 416x416 pixels. We want to show it
                // on the video preview, which is as wide as the screen and has a 16:9
                // aspect ratio. The video preview also may be letterboxed at the top
                // and bottom.
            
                //corrects the origin of the bounding box which was given by YOLO
            let origin = calculatePointCords(yoloPoint: CGPoint(x: prediction.rect.origin.x, y: prediction.rect.origin.y), view: view, sceneView: sceneView, imageProcessingSetting: imageProcessingSetting)
            
                //finds the location of reletive corners of the bounding box so that the actual cordinates of the bounding box can be found
                let minXminY = calculatePointCords(yoloPoint: CGPoint(x: prediction.rect.minX, y: prediction.rect.minY), view: view, sceneView: sceneView, imageProcessingSetting: self.imageProcessingSetting)
            let maxXminY = calculatePointCords(yoloPoint: CGPoint(x: prediction.rect.maxX, y: prediction.rect.minY), view: view, sceneView: sceneView, imageProcessingSetting: self.imageProcessingSetting)
            let minXmaxY = calculatePointCords(yoloPoint: CGPoint(x: prediction.rect.minX, y: prediction.rect.maxY), view: view, sceneView: sceneView, imageProcessingSetting: self.imageProcessingSetting)
            
                //calculates the width and height of the bounding box.
            let correctedBoundingBoxWidth = maxXminY!.x - minXminY!.x
            let correctedBoundingBoxHeight = minXmaxY!.y - minXminY!.y

            
                //adds a label to the found object.
            add3dLabel(label: String(labels[prediction.classIndex]), certanty: prediction.score, point: CGPoint(x: CGFloat(origin!.x+(correctedBoundingBoxWidth/2)), y: CGFloat(origin!.y+(correctedBoundingBoxHeight/2))))

        }
    }
    
    func updateCoreML() {
        ///////////////////////////
        // Get Camera Image as RGB
        let pixbuff : CVPixelBuffer? = (sceneView.session.currentFrame?.capturedImage)
        if pixbuff == nil { return }
        
        
        // For better throughput, we want to schedule multiple prediction requests
        // in parallel. These need to be separate instances, and inflightBuffer is
        // the index of the current request.
        let inflightIndex = inflightBuffer
        inflightBuffer += 1
        if inflightBuffer >= ViewController.maxInflightBuffers {
            inflightBuffer = 0
        }
        
        if useVision {
            // This method should always be called from the same thread!
            // Ain't nobody likes race conditions and crashes.
            self.predictUsingVision(pixelBuffer: pixbuff!, inflightIndex: inflightIndex)
        } else {
            // For better throughput, perform the prediction on a concurrent
            // background queue instead of on the serial VideoCapture queue.
            DispatchQueue.global().async {
                self.predict(pixelBuffer: pixbuff!, inflightIndex: inflightIndex, imageProcessingSetting: self.imageProcessingSetting)
            }
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixbuff!)
        
        
        
        // Note: Not entirely sure if the ciImage is being interpreted as RGB, but for now it works with the Inception model.
        // Note2: Also uncertain if the pixelBuffer should be rotated before handing off to Vision (VNImageRequestHandler) - regardless, for now, it still works well with the Inception model.
        
        ///////////////////////////
        // Prepare CoreML/Vision Request
        let imageRequestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        // let imageRequestHandler = VNImageRequestHandler(cgImage: cgImage!, orientation: myOrientation, options: [:]) // Alternatively; we can convert the above to an RGB CGImage and use that. Also UIInterfaceOrientation can inform orientation values.
        
        ///////////////////////////
        // Run Image Request
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
}
